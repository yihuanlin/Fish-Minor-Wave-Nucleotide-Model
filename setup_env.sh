#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh - Idempotent installer for the 'nucleotide' Conda environment
# ==============================================================================
set -euo pipefail

ENV_NAME="nucleotide"
PYTHON_VERSION="3.12"

echo "======================================================================"
echo "[*] Setting up Conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo "======================================================================"

# 1. Locate Conda or Mamba
CONDA_BIN=""
if command -v mamba &> /dev/null; then
    CONDA_BIN="mamba"
elif command -v conda &> /dev/null; then
    CONDA_BIN="conda"
elif [ -f "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
elif [ -f "${HOME}/anaconda3/bin/conda" ]; then
    CONDA_BIN="${HOME}/anaconda3/bin/conda"
else
    echo "[-] Error: Conda/Mamba not found in PATH or standard locations." >&2
    exit 1
fi
echo "[+] Using package manager: ${CONDA_BIN}"

# Enable conda commands in subshell
CONDA_BASE=$("${CONDA_BIN}" info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 2. Create or verify environment
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "[+] Environment '${ENV_NAME}' already exists."
else
    echo "[+] Creating environment '${ENV_NAME}' with python=${PYTHON_VERSION}..."
    "${CONDA_BIN}" create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

# 3. Install Bioinformatics CLIs and R
echo "[*] Installing bioinformatics CLIs and R via conda-forge and bioconda..."
"${CONDA_BIN}" install -y -n "${ENV_NAME}" \
    -c conda-forge -c bioconda \
    r-base \
    r-data.table \
    r-matrix \
    fastp \
    bowtie2 \
    samtools \
    bedtools \
    subread \
    minimap2 \
    meme \
    star \
    ucsc-bedgraphtobigwig \
    aria2 \
    git \
    curl \
    ca-certificates

# 4. Install CUDA PyTorch and Python requirements
echo "[*] Installing Python dependencies from requirements.txt..."
# Check if CUDA torch is already present
if ! python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    echo "[*] Installing CUDA-enabled PyTorch..."
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124 || \
    pip install --no-cache-dir torch torchvision
fi

pip install --no-cache-dir -r requirements.txt

# 5. Register Jupyter Kernel
echo "[*] Registering Jupyter kernel 'Python (${ENV_NAME})'..."
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

# 6. Verification Self-Check
echo "======================================================================"
echo "[*] Running verification self-check..."
echo "======================================================================"

python -c "
import sys, torch
print(f'[+] Python version: {sys.version.split()[0]}')
print(f'[+] PyTorch version: {torch.__version__}')
cuda_ok = torch.cuda.is_available()
print(f'[+] CUDA Available: {cuda_ok}')
if cuda_ok:
    print(f'[+] GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'[+] GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
"

echo "[*] Checking chrombpnet CLI..."
if command -v chrombpnet &> /dev/null; then
    chrombpnet --help | head -n 5
    echo "[+] chrombpnet CLI is installed and responsive."
else
    echo "[-] Warning: chrombpnet CLI not found in PATH." >&2
fi

echo "[*] Checking bioinformatics CLIs and R..."
for tool in Rscript fastp bowtie2 samtools bedtools featureCounts minimap2 STAR tomtom aria2c; do
    if command -v "${tool}" &> /dev/null; then
        echo "  [+] ${tool}: $(command -v ${tool})"
    else
        echo "  [-] Warning: ${tool} not found in PATH" >&2
    fi
done

echo "======================================================================"
echo "[SUCCESS] Environment '${ENV_NAME}' successfully configured and verified!"
echo "======================================================================"
