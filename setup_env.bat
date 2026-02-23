@echo off
REM =============================================================================
REM SAM3 Track Seg - One-click Environment Setup (Windows)
REM =============================================================================
REM Prerequisites: Miniconda/Anaconda installed, NVIDIA GPU with CUDA support
REM
REM Usage:
REM   1. Open any terminal (cmd, PowerShell, or Anaconda Prompt)
REM   2. cd to project root
REM   3. Run: setup_env.bat
REM =============================================================================

echo ===== SAM3 Track Seg Environment Setup =====
echo.

REM --- Step 0: Initialize conda for this shell session ---
REM   "call conda activate" only works if conda's shell hook is loaded.
REM   In a plain cmd.exe it is NOT loaded, so we run conda's own init script
REM   first.  We try several common install locations.
set "_CONDA_HOOK="
if exist "%USERPROFILE%\miniconda3\condabin\conda_hook.bat" (
    set "_CONDA_HOOK=%USERPROFILE%\miniconda3\condabin\conda_hook.bat"
) else if exist "%USERPROFILE%\anaconda3\condabin\conda_hook.bat" (
    set "_CONDA_HOOK=%USERPROFILE%\anaconda3\condabin\conda_hook.bat"
) else if exist "%PROGRAMDATA%\miniconda3\condabin\conda_hook.bat" (
    set "_CONDA_HOOK=%PROGRAMDATA%\miniconda3\condabin\conda_hook.bat"
)
if defined _CONDA_HOOK (
    echo [0/5] Initializing conda shell hook...
    call "%_CONDA_HOOK%"
) else (
    echo [0/5] conda_hook.bat not found — assuming Anaconda Prompt
)

REM --- Step 1: Create conda env ---
echo [1/5] Creating conda environment (Python 3.12)...
conda create -n sam3 python=3.12 -y
if errorlevel 1 (
    echo WARNING: conda create failed, env may already exist
)
call conda activate sam3

REM --- Verify activation succeeded ---
python --version 2>nul | findstr /C:"3.12" >nul
if errorlevel 1 (
    echo ERROR: conda activate sam3 did not switch to Python 3.12
    echo        Please run this script from Anaconda Prompt, or check your conda installation.
    pause
    exit /b 1
)

REM --- Step 2: Install PyTorch with CUDA 12.8 (required for RTX 50xx / Blackwell) ---
echo.
echo [2/5] Installing PyTorch with CUDA 12.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

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
