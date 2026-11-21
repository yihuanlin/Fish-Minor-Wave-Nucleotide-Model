#!/usr/bin/env python3
"""
scripts/task_manager.py
=======================
Persistent Background Task Manager for both local background execution
and SLURM cluster scheduling.

Features:
- Decoupled execution: Local jobs use subprocess.Popen(start_new_session=True)
  so tasks survive notebook restarts or terminal disconnects.
- SLURM support: Generates .sbatch scripts, submits via sbatch, queries squeue/sacct.
- Idempotent: Skips completed tasks (via .done marker) unless force_rerun=True;
  detects already running tasks without spawning duplicates.
- Process tracking: Validates PID via /proc/<pid>/stat start-time to guard against PID recycling.
- Status, logs & cancellation: Interactive inspection via tm.status(), tm.tail(), tm.cancel().
"""

import os
import sys
import time
import json
import signal
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import pandas as pd


class TaskManager:
    """Manages background tasks with persistent state on disk."""

    def __init__(self, work_dir: str = "./work_dir", backend: str = "local_background", slurm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TaskManager.

        Args:
            work_dir: Working directory under which tasks/ directory is kept.
            backend: Execution backend - 'local_background' or 'slurm'.
            slurm_config: Dict containing default SLURM settings (e.g. partition, time, cpus, mem, gpus).
        """
        self.work_dir = Path(work_dir).resolve()
        self.backend = backend
        self.slurm_config = slurm_config or {
            "partition": "gpu",
            "time": "24:00:00",
            "cpus_per_task": 8,
            "mem": "64G",
            "gres": "gpu:1",
            "job_name": "nucleotide_job"
        }
        self.tasks_dir = self.work_dir / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _get_task_dir(self, task_name: str) -> Path:
        task_dir = self.tasks_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _get_process_start_time(self, pid: int) -> Optional[int]:
        """Read process start time (jiffies since boot) from /proc/<pid>/stat to prevent PID recycling issues."""
        try:
            stat_path = Path(f"/proc/{pid}/stat")
            if stat_path.exists():
                with open(stat_path, "r") as f:
                    content = f.read()
                    # Splitting after rfind(')') ensures process names containing spaces don't shift fields
                    idx = content.rfind(")")
                    if idx != -1:
                        rest = content[idx + 1:].strip().split()
                        # Field 22 is starttime; in rest (which starts at field 3: state), index 19 is starttime (22 - 3 = 19)
                        if len(rest) > 19:
                            return int(rest[19])
                    else:
                        fields = content.split()
                        if len(fields) > 21:
                            return int(fields[21])
        except Exception:
            pass
        return None

    @staticmethod
    def _is_pid_alive(pid: int, recorded_start_time: Optional[int] = None) -> bool:
        """Check if PID is alive, non-zombie, and matches the recorded start time."""
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False

        try:
            stat_path = Path(f"/proc/{pid}/stat")
            if not stat_path.exists():
                return False
            content = stat_path.read_text()
            idx = content.rfind(")")
            if idx != -1:
                rest = content[idx + 1:].strip().split()
                # rest[0] is state (field 3: 'Z' for zombie, 'X' for dead)
                if len(rest) > 0 and rest[0] in ("Z", "X"):
                    return False
                if recorded_start_time is not None and len(rest) > 19:
                    current_start_time = int(rest[19])
                    if current_start_time != recorded_start_time:
                        return False
            else:
                fields = content.split()
                if len(fields) > 2 and fields[2] in ("Z", "X"):
                    return False
                if recorded_start_time is not None and len(fields) > 21:
                    if int(fields[21]) != recorded_start_time:
                        return False
        except Exception:
            return False
        return True

    def _is_slurm_job_running(self, job_id: str) -> bool:
        """Check if SLURM job is RUNNING or PENDING."""
        try:
            res = subprocess.run(["squeue", "-j", str(job_id), "-h", "-o", "%T"], capture_output=True, text=True, check=False)
            state = res.stdout.strip()
            return state in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]
        except Exception:
            return False

    def update_progress(self, task_name: str, progress_str: str) -> None:
        """Update progress string for task in progress.txt."""
        task_dir = self._get_task_dir(task_name)
        prog_file = task_dir / "progress.txt"
        try:
            prog_file.write_text(str(progress_str).strip())
        except Exception:
            pass

    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Return task status dict, updating state dynamically."""
        task_dir = self._get_task_dir(task_name)
        status_file = task_dir / "status.json"
        done_marker = task_dir / ".done"

        if not status_file.exists():
            return {"name": task_name, "status": "NOT_STARTED", "backend": self.backend}

        try:
            with open(status_file, "r") as f:
                info = json.load(f)
        except Exception:
            return {"name": task_name, "status": "UNKNOWN", "backend": self.backend}

        # Check completion marker
        if done_marker.exists():
            info["status"] = "COMPLETED"
        else:
            rc_file = task_dir / "exit_code.txt"
            if rc_file.exists():
                exit_code = info.get("exit_code")
                try:
                    exit_code = int(rc_file.read_text().strip())
                    info["exit_code"] = exit_code
                except Exception:
                    pass
                if exit_code == 0:
                    done_marker.touch()
                    info["status"] = "COMPLETED"
                elif exit_code == 143 or info.get("status") == "CANCELLED":
                    info["status"] = "CANCELLED"
                else:
                    info["status"] = "FAILED"
            else:
                # Check if currently running
                backend = info.get("backend", "local_background")
                if backend == "local_background":
                    pid = info.get("pid")
                    start_time = info.get("start_time_proc")
                    if pid and self._is_pid_alive(pid, start_time):
                        info["status"] = "RUNNING"
                        info["elapsed_sec"] = time.time() - info.get("submit_time", time.time())
                    else:
                        info["status"] = "FAILED"
                elif backend == "slurm":
                    job_id = info.get("job_id")
                    if job_id and self._is_slurm_job_running(job_id):
                        info["status"] = "RUNNING"
                        info["elapsed_sec"] = time.time() - info.get("submit_time", time.time())
                    else:
                        # Query sacct for exit code
                        try:
                            res = subprocess.run(["sacct", "-j", str(job_id), "--format=State,ExitCode", "-n", "-P"],
                                                 capture_output=True, text=True, check=False)
                            first_line = res.stdout.strip().split("\n")[0] if res.stdout.strip() else ""
                            if "COMPLETED" in first_line:
                                done_marker.touch()
                                info["status"] = "COMPLETED"
                            else:
                                info["status"] = "FAILED"
                        except Exception:
                            info["status"] = "UNKNOWN"

        # Check progress file or field
        prog_file = task_dir / "progress.txt"
        if prog_file.exists():
            try:
                info["progress"] = prog_file.read_text().strip()
            except Exception:
                pass
        if "progress" not in info or not info["progress"]:
            if info.get("status") == "COMPLETED":
                info["progress"] = "100% (Done)"
            else:
                info["progress"] = "-"
        elif info.get("status") == "COMPLETED" and "100%" not in info.get("progress", ""):
            info["progress"] = "100% (Done)"

        return info

    def submit(
        self,
        task_name: str,
        cmd: List[str] | str,
        args_dict: Optional[Dict[str, Any]] = None,
        force_rerun: bool = False,
        env_vars: Optional[Dict[str, str]] = None,
        slurm_overrides: Optional[Dict[str, Any]] = None,
        depends_on: Optional[Union[str, List[str]]] = None,
        requires_gpu: bool = False
    ) -> Dict[str, Any]:
        """
        Submit a task for background execution in an idempotent manner.

        Args:
            task_name: Unique task identifier.
            cmd: Command string or list of tokens.
            args_dict: Dictionary of parameters (used to compute hash for reproducibility).
            force_rerun: If True, rerun even if .done marker exists.
            env_vars: Extra environment variables.
            slurm_overrides: SLURM parameter overrides if backend == 'slurm'.
            depends_on: Task name or list of task names this task depends on.
            requires_gpu: If True, serializes execution via exclusive flock on tasks/.gpu.lock.
        """
        task_dir = self._get_task_dir(task_name)
        status_file = task_dir / "status.json"
        done_marker = task_dir / ".done"
        log_file = task_dir / f"{task_name}.log"

        current_info = self.get_task_info(task_name)

        # 1. Check if already running (never spawn duplicate process)
        if current_info.get("status") == "RUNNING":
            elapsed = current_info.get("elapsed_sec", 0)
            print(f"[RUNNING] Task '{task_name}' is already running (PID/Job: {current_info.get('pid') or current_info.get('job_id')}, elapsed: {elapsed:.1f}s). Skipping submission.")
            return current_info

        # 2. Check if already completed
        if task_name != "download_resources" and done_marker.exists() and not force_rerun:
            print(f"[ALREADY COMPLETED] Task '{task_name}' previously completed (.done marker exists). Set force_rerun=True to rerun.")
            return current_info

        # 3. Clean up previous artifacts for rerun
        if done_marker.exists():
            done_marker.unlink()
        exit_code_file = task_dir / "exit_code.txt"
        if exit_code_file.exists():
            exit_code_file.unlink()
        if force_rerun and log_file.exists():
            with open(log_file, "a") as lf:
                lf.write(f"\n{'='*70}\n[RERUN REQUESTED] {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}\n\n")

        # Parse dependencies
        dep_list: List[str] = []
        if depends_on:
            dep_list = [depends_on] if isinstance(depends_on, str) else list(depends_on)
            for dep in dep_list:
                if dep == "download_resources":
                    dep_dir = self._get_task_dir(dep)
                    manifest_p = self.work_dir / "downloads" / "resource_manifest.tsv"
                    if manifest_p.exists() and not (dep_dir / ".done").exists():
                        try:
                            m_df = pd.read_csv(manifest_p, sep="\t")
                            if not any(m_df["Status"].astype(str).str.startswith("PENDING")):
                                (dep_dir / ".done").touch()
                                st_file = dep_dir / "status.json"
                                if not st_file.exists():
                                    with open(st_file, "w") as f:
                                        json.dump({"name": "download_resources", "status": "COMPLETED", "progress": "100% (Done)"}, f, indent=2)
                        except Exception:
                            pass

        # Compute arguments hash
        import shlex
        cmd_str = cmd if isinstance(cmd, str) else shlex.join([str(x) for x in cmd])
        args_payload = json.dumps({"cmd": cmd_str, "args": args_dict or {}, "depends_on": dep_list, "requires_gpu": requires_gpu}, sort_keys=True)
        args_hash = hashlib.sha256(args_payload.encode()).hexdigest()[:16]

        run_env = os.environ.copy()
        if env_vars:
            run_env.update(env_vars)

        if self.backend == "local_background":
            # Create a robust runner script that records exit code and creates .done marker on code 0
            bin_dir = str(Path(sys.executable).parent)
            runner_script = task_dir / "run.sh"
            gpu_lock_file = self.tasks_dir / ".gpu.lock"
            with open(runner_script, "w") as f:
                f.write("#!/usr/bin/env bash\n")
                f.write("set -o pipefail\n")
                f.write(f"export PATH=\"{bin_dir}:$PATH\"\n")
                f.write(f"cd \"{Path.cwd().resolve()}\"\n")
                f.write(f"echo \"[TASK START] $(date)\" >> \"{log_file}\"\n")

                if dep_list:
                    for dep in dep_list:
                        dep_dir = self._get_task_dir(dep)
                        f.write(f"echo \"[DEPENDENCY WAIT] Waiting for prerequisite task '{dep}'...\" >> \"{log_file}\"\n")
                        f.write(f"echo \"Waiting on {dep}\" > \"{task_dir}/progress.txt\"\n")
                        f.write(f"while true; do\n")
                        f.write(f"    if [ -f \"{dep_dir}/.done\" ]; then\n")
                        f.write(f"        echo \"[DEPENDENCY OK] Prerequisite '{dep}' satisfied.\" >> \"{log_file}\"\n")
                        f.write(f"        break\n")
                        f.write(f"    fi\n")
                        f.write(f"    if [ -f \"{dep_dir}/status.json\" ]; then\n")
                        f.write(f"        if grep -q -E '\"status\": *\"CANCELLED\"' \"{dep_dir}/status.json\" 2>/dev/null; then\n")
                        f.write(f"            echo \"[DEPENDENCY CANCELLED] Prerequisite '{dep}' was cancelled. Aborting.\" >> \"{log_file}\"\n")
                        f.write(f"            echo 143 > \"{exit_code_file}\"\n")
                        f.write(f"            echo \"Cancelled (prerequisite '{dep}' cancelled)\" > \"{task_dir}/progress.txt\"\n")
                        f.write(f"            exit 143\n")
                        f.write(f"        fi\n")
                        f.write(f"        if grep -q -E '\"status\": *\"FAILED\"' \"{dep_dir}/status.json\" 2>/dev/null; then\n")
                        f.write(f"            echo \"[DEPENDENCY FAILED] Prerequisite '{dep}' failed. Aborting.\" >> \"{log_file}\"\n")
                        f.write(f"            echo 1 > \"{exit_code_file}\"\n")
                        f.write(f"            echo \"Failed (prerequisite '{dep}' failed)\" > \"{task_dir}/progress.txt\"\n")
                        f.write(f"            exit 1\n")
                        f.write(f"        fi\n")
                        f.write(f"    fi\n")
                        f.write(f"    if [ -f \"{dep_dir}/exit_code.txt\" ]; then\n")
                        f.write(f"        DEP_RC=$(cat \"{dep_dir}/exit_code.txt\" 2>/dev/null || echo 1)\n")
                        f.write(f"        if [ \"$DEP_RC\" = \"143\" ]; then\n")
                        f.write(f"            echo \"[DEPENDENCY CANCELLED] Prerequisite '{dep}' was cancelled (RC 143). Aborting.\" >> \"{log_file}\"\n")
                        f.write(f"            echo 143 > \"{exit_code_file}\"\n")
                        f.write(f"            echo \"Cancelled (prerequisite '{dep}' cancelled)\" > \"{task_dir}/progress.txt\"\n")
                        f.write(f"            exit 143\n")
                        f.write(f"        elif [ \"$DEP_RC\" != \"0\" ]; then\n")
                        f.write(f"            echo \"[DEPENDENCY ERROR] Prerequisite '{dep}' failed with exit code $DEP_RC. Aborting.\" >> \"{log_file}\"\n")
                        f.write(f"            echo 1 > \"{exit_code_file}\"\n")
                        f.write(f"            echo \"Failed (prerequisite '{dep}' failed)\" > \"{task_dir}/progress.txt\"\n")
                        f.write(f"            exit 1\n")
                        f.write(f"        fi\n")
                        f.write(f"    fi\n")
                        f.write(f"    DEP_PID=\"\"\n")
                        f.write(f"    if [ -f \"{dep_dir}/pid.txt\" ]; then\n")
                        f.write(f"        DEP_PID=$(cat \"{dep_dir}/pid.txt\" 2>/dev/null | tr -d '[:space:]')\n")
                        f.write(f"    elif [ -f \"{dep_dir}/status.json\" ]; then\n")
                        f.write(f"        DEP_PID=$(grep -o -E '\"pid\": *[0-9]+' \"{dep_dir}/status.json\" 2>/dev/null | grep -o -E '[0-9]+')\n")
                        f.write(f"    fi\n")
                        f.write(f"    if [ -n \"$DEP_PID\" ] && ! kill -0 \"$DEP_PID\" 2>/dev/null; then\n")
                        f.write(f"        sleep 2\n")
                        f.write(f"        if [ ! -f \"{dep_dir}/.done\" ] && [ ! -f \"{dep_dir}/exit_code.txt\" ]; then\n")
                        f.write(f"            echo \"[DEPENDENCY DIED] Prerequisite '{dep}' (PID $DEP_PID) terminated unexpectedly. Aborting.\" >> \"{log_file}\"\n")
                        f.write(f"            echo 1 > \"{exit_code_file}\"\n")
                        f.write(f"            echo \"Failed (prerequisite '{dep}' died)\" > \"{task_dir}/progress.txt\"\n")
                        f.write(f"            exit 1\n")
                        f.write(f"        fi\n")
                        f.write(f"    fi\n")
                        f.write(f"    sleep 3\n")
                        f.write(f"done\n")

                if requires_gpu:
                    f.write(f"echo \"Waiting for GPU lock...\" > \"{task_dir}/progress.txt\"\n")
                    f.write(f"echo \"[GPU LOCK] Waiting for exclusive GPU lock on {gpu_lock_file}...\" >> \"{log_file}\"\n")
                    f.write(f"exec 200>\"{gpu_lock_file}\"\n")
                    f.write(f"flock 200\n")
                    f.write(f"echo \"[GPU LOCK] Acquired exclusive GPU lock at $(date)\" >> \"{log_file}\"\n")
                    f.write(f"echo \"Running (GPU acquired)\" > \"{task_dir}/progress.txt\"\n")

                f.write(f"{cmd_str} >> \"{log_file}\" 2>&1\n")
                f.write("RC=$?\n")
                f.write(f"echo $RC > \"{exit_code_file}\"\n")
                f.write(f"echo \"[TASK FINISHED] Exit code: $RC at $(date)\" >> \"{log_file}\"\n")
                f.write("if [ $RC -eq 0 ]; then\n")
                f.write(f"    touch \"{done_marker}\"\n")
                f.write("fi\n")
                f.write("exit $RC\n")
            runner_script.chmod(0o755)

            # Spawn fully detached process in its own session (pgid == pid)
            res = subprocess.run(
                ["/usr/bin/env", "bash", "-c", f"setsid bash \"{runner_script}\" </dev/null >/dev/null 2>&1 & echo $!"],
                cwd=str(Path.cwd().resolve()),
                capture_output=True,
                text=True,
                check=True,
                env=run_env
            )
            pid = int(res.stdout.strip())
            (task_dir / "pid.txt").write_text(f"{pid}\n")
            time.sleep(0.3)  # Brief pause to ensure process initialization
            proc_start = self._get_process_start_time(pid)

            status_payload = {
                "name": task_name,
                "backend": "local_background",
                "pid": pid,
                "start_time_proc": proc_start,
                "submit_time": time.time(),
                "cmd": cmd_str,
                "args_hash": args_hash,
                "log_file": str(log_file),
                "requires_gpu": requires_gpu,
                "status": "RUNNING"
            }
            with open(status_file, "w") as f:
                json.dump(status_payload, f, indent=2)

            print(f"[SUBMITTED] Task '{task_name}' launched in background (PID: {pid}). Log: {log_file}")
            return status_payload

        elif self.backend == "slurm":
            sbatch_opts = self.slurm_config.copy()
            if slurm_overrides:
                sbatch_opts.update(slurm_overrides)

            bin_dir = str(Path(sys.executable).parent)
            sbatch_file = task_dir / f"{task_name}.sbatch"
            with open(sbatch_file, "w") as f:
                f.write("#!/usr/bin/env bash\n")
                f.write(f"#SBATCH --job-name={task_name}\n")
                f.write(f"#SBATCH --output={log_file}\n")
                f.write(f"#SBATCH --error={log_file}\n")
                f.write(f"#SBATCH --time={sbatch_opts.get('time', '24:00:00')}\n")
                f.write(f"#SBATCH --cpus-per-task={sbatch_opts.get('cpus_per_task', 8)}\n")
                f.write(f"#SBATCH --mem={sbatch_opts.get('mem', '64G')}\n")
                if "partition" in sbatch_opts and sbatch_opts["partition"]:
                    f.write(f"#SBATCH --partition={sbatch_opts['partition']}\n")
                emit_gres = False
                if slurm_overrides and "gres" in slurm_overrides and slurm_overrides["gres"]:
                    emit_gres = True
                elif requires_gpu and "gres" in sbatch_opts and sbatch_opts["gres"]:
                    emit_gres = True

                if emit_gres:
                    f.write(f"#SBATCH --gres={sbatch_opts['gres']}\n")
                if dep_list:
                    dep_jobs = []
                    for dep in dep_list:
                        d_info = self.get_task_info(dep)
                        # Only depend on active running/pending jobs; skip already COMPLETED dependencies to prevent sbatch rejection
                        if d_info.get("job_id") and d_info.get("status") == "RUNNING":
                            dep_jobs.append(str(d_info["job_id"]))
                    if dep_jobs:
                        f.write(f"#SBATCH --dependency=afterok:{':'.join(dep_jobs)}\n")
                f.write("\nset -o pipefail\n")
                f.write(f"export PATH=\"{bin_dir}:$PATH\"\n")
                f.write(f"cd \"{Path.cwd().resolve()}\"\n")
                f.write(f"echo \"[SLURM TASK START] $(date)\"\n")
                f.write(f"{cmd_str}\n")
                f.write("RC=$?\n")
                f.write("if [ $RC -eq 0 ]; then\n")
                f.write(f"    touch \"{done_marker}\"\n")
                f.write("fi\n")
                f.write("exit $RC\n")

            res = subprocess.run(["sbatch", str(sbatch_file)], capture_output=True, text=True, check=True, env=run_env)
            # Typical output: "Submitted batch job 123456"
            job_id = res.stdout.strip().split()[-1]

            status_payload = {
                "name": task_name,
                "backend": "slurm",
                "job_id": job_id,
                "submit_time": time.time(),
                "cmd": cmd_str,
                "args_hash": args_hash,
                "log_file": str(log_file),
                "status": "RUNNING"
            }
            with open(status_file, "w") as f:
                json.dump(status_payload, f, indent=2)

            print(f"[SUBMITTED] Task '{task_name}' submitted to SLURM (Job ID: {job_id}). Log: {log_file}")
            return status_payload

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def cancel(self, task_name: str) -> bool:
        """Cancel a running task."""
        info = self.get_task_info(task_name)
        status = info.get("status")
        if status != "RUNNING":
            print(f"Task '{task_name}' is not running (status: {status}).")
            return False

        task_dir = self._get_task_dir(task_name)
        status_file = task_dir / "status.json"

        backend = info.get("backend", "local_background")
        if backend == "local_background":
            pid = info.get("pid")
            if pid:
                try:
                    # Kill entire process group if distinct from current process group
                    curr_pgid = os.getpgrp()
                    pgid = os.getpgid(pid)
                    if pgid != curr_pgid:
                        os.killpg(pgid, signal.SIGTERM)
                        time.sleep(0.5)
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.5)
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                except Exception:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
                print(f"[CANCELLED] Terminated local task '{task_name}' (PID: {pid}).")
        elif backend == "slurm":
            job_id = info.get("job_id")
            if job_id:
                subprocess.run(["scancel", str(job_id)], check=False)
                print(f"[CANCELLED] Scancelled SLURM task '{task_name}' (Job ID: {job_id}).")

        info["status"] = "CANCELLED"
        with open(status_file, "w") as f:
            json.dump(info, f, indent=2)
        try:
            (task_dir / "exit_code.txt").write_text("143\n")
        except Exception:
            pass
        return True

    def cancel_all(self) -> pd.DataFrame:
        """
        Iterate over all task directories in self.tasks_dir, terminate every RUNNING
        local process group (SIGTERM then SIGKILL) or SLURM job (scancel), clean up
        any stale/zombie status.json states, and print a summary table of terminated tasks.
        """
        terminated_records = []
        if self.tasks_dir.exists():
            for task_folder in sorted(self.tasks_dir.iterdir()):
                if not task_folder.is_dir():
                    continue
                t_name = task_folder.name
                info = self.get_task_info(t_name)
                status = info.get("status")
                backend = info.get("backend", "local_background")
                pid = info.get("pid")
                job_id = info.get("job_id")
                status_file = task_folder / "status.json"

                action_taken = "None"
                # If RUNNING or has an active PID, terminate it
                if status == "RUNNING" or (pid and self._is_pid_alive(pid)):
                    if backend == "local_background" and pid:
                        try:
                            curr_pgid = os.getpgrp()
                            pgid = os.getpgid(pid)
                            if pgid != curr_pgid:
                                os.killpg(pgid, signal.SIGTERM)
                                time.sleep(0.3)
                                try:
                                    os.killpg(pgid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                                action_taken = f"Terminated process group {pgid} (PID {pid})"
                            else:
                                os.kill(pid, signal.SIGTERM)
                                time.sleep(0.3)
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                                action_taken = f"Terminated PID {pid}"
                        except Exception:
                            try:
                                os.kill(pid, signal.SIGKILL)
                                action_taken = f"Killed PID {pid}"
                            except Exception as e:
                                action_taken = f"Kill error: {e}"
                    elif backend == "slurm" and job_id:
                        subprocess.run(["scancel", str(job_id)], check=False)
                        action_taken = f"Scancelled SLURM job {job_id}"

                    info["status"] = "CANCELLED"
                    with open(status_file, "w") as sf:
                        json.dump(info, sf, indent=2)
                    try:
                        (task_folder / "exit_code.txt").write_text("143\n")
                    except Exception:
                        pass

                    terminated_records.append({
                        "Task": t_name,
                        "Backend": backend,
                        "PID / JobID": str(pid or job_id or "-"),
                        "Previous_Status": status,
                        "Action": action_taken,
                        "New_Status": "CANCELLED"
                    })
                else:
                    # Clean up stale/zombie status if recorded as RUNNING in status.json
                    try:
                        if status_file.exists():
                            with open(status_file, "r") as sf:
                                raw_info = json.load(sf)
                            if raw_info.get("status") == "RUNNING":
                                raw_info["status"] = status
                                with open(status_file, "w") as sf:
                                    json.dump(raw_info, sf, indent=2)
                    except Exception:
                        pass

        df_term = pd.DataFrame(terminated_records) if terminated_records else pd.DataFrame(
            columns=["Task", "Backend", "PID / JobID", "Previous_Status", "Action", "New_Status"]
        )
        print("=" * 80)
        print(f"[*] TaskManager.cancel_all(): Terminated {len(terminated_records)} running task(s).")
        if not df_term.empty:
            print(df_term.to_string(index=False))
        else:
            print("    No actively running tasks found.")
        print("=" * 80)
        return df_term

    def tail(self, task_name: str, n: int = 25) -> str:
        """Return the last n lines from the task's log file."""
        task_dir = self._get_task_dir(task_name)
        log_file = task_dir / f"{task_name}.log"
        if not log_file.exists():
            return f"Log file for '{task_name}' does not exist yet: {log_file}"

        try:
            with open(log_file, "r", errors="replace") as f:
                lines = f.readlines()
                tail_lines = lines[-n:] if len(lines) >= n else lines
                content = "".join(tail_lines)
                print(f"=== Last {len(tail_lines)} lines of {log_file} ===\n{content}")
                return content
        except Exception as e:
            err = f"Failed to read log {log_file}: {e}"
            print(err)
            return err

    def status(self) -> pd.DataFrame:
        """Return a formatted DataFrame of all registered tasks."""
        records = []
        if self.tasks_dir.exists():
            for task_folder in sorted(self.tasks_dir.iterdir()):
                if task_folder.is_dir():
                    info = self.get_task_info(task_folder.name)
                    submit_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.get("submit_time", 0))) if "submit_time" in info else "-"
                    elapsed = info.get("elapsed_sec")
                    elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "-"
                    records.append({
                        "Task": task_folder.name,
                        "Status": info.get("status", "UNKNOWN"),
                        "Progress": info.get("progress", "-"),
                        "Backend": info.get("backend", "-"),
                        "PID / JobID": str(info.get("pid") or info.get("job_id") or "-"),
                        "Submitted": submit_str,
                        "Elapsed": elapsed_str,
                        "Log": str(info.get("log_file", "-"))
                    })

        df = pd.DataFrame(records) if records else pd.DataFrame(columns=["Task", "Status", "Progress", "Backend", "PID / JobID", "Submitted", "Elapsed", "Log"])
        return df

    def wait(self, task_name: str, poll_interval: int = 5, timeout: Optional[int] = None) -> str:
        """Poll task until completion or timeout, printing progress."""
        start_wait = time.time()
        print(f"[*] Waiting for task '{task_name}' to complete...")
        while True:
            info = self.get_task_info(task_name)
            status = info.get("status")
            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                print(f"[+] Task '{task_name}' finished with status: {status}")
                return status

            if timeout and (time.time() - start_wait > timeout):
                print(f"[-] Timed out waiting for task '{task_name}'.")
                return "TIMEOUT"

            time.sleep(poll_interval)

    def wait_all(self, task_names: List[str], poll_interval: int = 5, timeout: Optional[int] = None) -> Dict[str, str]:
        """Poll multiple tasks until all reach terminal state or timeout."""
        start_wait = time.time()
        print(f"[*] Waiting for {len(task_names)} tasks to complete: {', '.join(task_names)}")
        remaining = set(task_names)
        results = {}
        while remaining:
            for t_name in list(remaining):
                info = self.get_task_info(t_name)
                st = info.get("status")
                if st in ["COMPLETED", "FAILED", "CANCELLED"]:
                    results[t_name] = st
                    remaining.remove(t_name)
                    print(f"[+] Task '{t_name}' finished with status: {st}")
            if not remaining:
                break
            if timeout and (time.time() - start_wait > timeout):
                print(f"[-] Timed out waiting for tasks: {list(remaining)}")
                for t_name in remaining:
                    results[t_name] = "TIMEOUT"
                break
            time.sleep(poll_interval)
        return results

    def chain(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Submit a sequential chain of tasks where each subsequent task depends on the previous one.
        Each element in tasks is a dict matching submit() parameters: {'task_name': ..., 'cmd': ..., ...}
        """
        submitted = []
        prev_name = None
        for t in tasks:
            t_copy = t.copy()
            name = t_copy.pop("task_name", t_copy.pop("name", None))
            if not name:
                raise ValueError("Each task in chain must have 'task_name' or 'name'")
            cmd = t_copy.pop("cmd")
            deps = t_copy.pop("depends_on", [])
            if isinstance(deps, str):
                deps = [deps]
            else:
                deps = list(deps)
            if prev_name and prev_name not in deps:
                deps.append(prev_name)
            res = self.submit(task_name=name, cmd=cmd, depends_on=deps, **t_copy)
            submitted.append(res)
            prev_name = name
        return submitted


def set_task_progress(work_dir: str | Path, task_name: str, progress_str: str) -> None:
    """
    Standalone helper to update progress string in work_dir/tasks/{task_name}/progress.txt
    without requiring a full TaskManager instance.
    """
    try:
        prog_file = Path(work_dir) / "tasks" / task_name / "progress.txt"
        prog_file.parent.mkdir(parents=True, exist_ok=True)
        prog_file.write_text(str(progress_str).strip())
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Persistent Background Task Manager CLI")
    parser.add_argument("action", choices=["status", "tail", "cancel", "cancel_all", "wait"], help="Action to perform")
    parser.add_argument("task_name", nargs="?", default=None, help="Name of task")
    parser.add_argument("--work-dir", default="./work_dir", help="Work directory containing tasks/")
    parser.add_argument("-n", type=int, default=25, help="Number of log lines for tail")
    args = parser.parse_args()

    tm = TaskManager(work_dir=args.work_dir)

    if args.action == "status":
        df = tm.status()
        if df.empty:
            print("No tasks registered.")
        else:
            print(df.to_string(index=False))
    elif args.action == "tail":
        if not args.task_name:
            print("Error: task_name required for tail", file=sys.stderr)
            sys.exit(1)
        tm.tail(args.task_name, n=args.n)
    elif args.action == "cancel":
        if not args.task_name:
            print("Error: task_name required for cancel", file=sys.stderr)
            sys.exit(1)
        tm.cancel(args.task_name)
    elif args.action == "cancel_all":
        tm.cancel_all()
    elif args.action == "wait":
        if not args.task_name:
            print("Error: task_name required for wait", file=sys.stderr)
            sys.exit(1)
        tm.wait(args.task_name)
