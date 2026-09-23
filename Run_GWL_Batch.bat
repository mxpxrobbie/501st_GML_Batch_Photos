@echo off
echo Starting 501st GWL Photo Processing...

:: Switch to the folder where this batch file lives
cd /d "%~dp0"

:: Ensure all required libraries are installed
echo Checking and installing required packages...
python -m pip install --quiet pillow rembg psd-tools opencv-python onnxruntime pytoshop numpy six packbits tqdm

echo Running batch processor...
echo ----------------------------------------------------
python process_501st_batch.py

echo.
echo Processing Complete. Press any key to exit.
pause