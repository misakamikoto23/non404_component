@echo off
chcp 65001 >nul
setlocal
if "%~1"=="" (
  echo 请把一张原理图图片拖到这个文件上，即可自动检测和生成复核候选。
  echo 默认在本机处理，不调用付费API。
  pause
  exit /b 1
)
"%~dp0.venv-yolo\Scripts\python.exe" "%~dp0scripts\run_automatic_components.py" --images "%~1" --open
pause
