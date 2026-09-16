@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

cd /d "%~dp0"

echo ============================================================
echo   Wheat Detection - Deploy len Kaggle Dataset (offline bundle)
echo ============================================================
echo Thu muc: %cd%
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay "python". Kiem tra da cai Python va them vao PATH chua.
    pause
    exit /b 1
)

set "SLUG=wheat-yolov8-offline-bundle"
set /p SLUG_INPUT="Slug dataset Kaggle (Enter de dung mac dinh %SLUG%): "
if not "%SLUG_INPUT%"=="" set "SLUG=%SLUG_INPUT%"

set "WEIGHTS=yolov8n.pt"
set /p WEIGHTS_INPUT="Duong dan file best.pt (Enter de dung mac dinh %WEIGHTS% - model test): "
if not "%WEIGHTS_INPUT%"=="" set "WEIGHTS=%WEIGHTS_INPUT%"

if not exist "%WEIGHTS%" (
    echo [LOI] Khong tim thay file: %WEIGHTS%
    pause
    exit /b 1
)

set "MSG=Update weights + code"
set /p MSG_INPUT="Ghi chu version (Enter de dung mac dinh '%MSG%'): "
if not "%MSG_INPUT%"=="" set "MSG=%MSG_INPUT%"

echo.
echo --- Se chay: ---
echo python scripts\deploy_kaggle.py --weights "%WEIGHTS%" --slug %SLUG% -m "%MSG%"
echo (Neu day la lan DAU TIEN tao dataset voi slug nay, huy o day va tu chay them --new)
echo.
set /p CONFIRM="Xac nhan chay? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Da huy, khong lam gi ca.
    pause
    exit /b 0
)

python scripts\deploy_kaggle.py --weights "%WEIGHTS%" --slug %SLUG% -m "%MSG%"
if errorlevel 1 (
    echo.
    echo [LOI] Deploy that bai. Doc log loi o tren de biet nguyen nhan.
    echo Neu la loi "con thay doi chua commit": chay push_to_github.bat truoc, roi chay lai file nay.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   HOAN TAT! Dataset da duoc cap nhat tren Kaggle.
echo ============================================================
pause
