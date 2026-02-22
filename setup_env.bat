@echo off
REM =============================================================================
REM SAM3 Track Seg - One-click Environment Setup (Windows)
REM =============================================================================
REM Prerequisites: Miniconda/Anaconda installed, NVIDIA GPU with CUDA support
REM
REM Usage:
REM   1. Open Anaconda Prompt
REM   2. cd to project root
REM   3. Run: setup_env.bat
REM =============================================================================

echo ===== SAM3 Track Seg Environment Setup =====
echo.

REM --- Step 1: Create conda env ---
echo [1/5] Creating conda environment (Python 3.12)...
conda create -n sam3 python=3.12 -y
if errorlevel 1 (
    echo WARNING: conda create failed, env may already exist
)
call conda activate sam3

REM --- Step 2: Install PyTorch with CUDA 12.6 ---
echo.
echo [2/5] Installing PyTorch with CUDA 12.6...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM --- Step 3: Install project dependencies ---
echo.
echo [3/5] Installing project dependencies...
pip install -r requirements.txt

REM --- Step 4: Install SAM3 (local editable) ---
echo.
echo [4/5] Installing SAM3 model package (local)...
pip install -e ./sam3/

REM --- Step 5: Verify ---
echo.
echo [5/5] Verifying installation...
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
python -c "import rasterio; print(f'Rasterio {rasterio.__version__}')"
python -c "import PIL; print(f'Pillow {PIL.__version__}')"
python -c "import scipy; print(f'SciPy {scipy.__version__}')"
python -c "import google.genai; print('google-genai OK')"
python -c "import sam3; print(f'SAM3 OK')"

echo.
echo ===== Setup Complete =====
echo.
echo NOTE: Make sure model/sam3.pt exists (copy from existing machine).
echo       Blender 5.0 must be installed separately.
pause
