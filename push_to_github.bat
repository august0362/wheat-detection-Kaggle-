@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

cd /d "%~dp0"

echo ============================================================
echo   Wheat Detection - Auto Push to GitHub
echo ============================================================
echo Thu muc: %cd%
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay "git". Hay cai Git for Windows roi thu lai.
    pause
    exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo [LOI] Thu muc nay khong phai la mot git repository.
    pause
    exit /b 1
)

echo --- Trang thai hien tai (git status) ---
git status --short
echo.

set /p CONFIRM="Add + commit + push toan bo thay doi o tren? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Da huy, khong lam gi ca.
    pause
    exit /b 0
)

git add -A

git diff --cached --quiet
if errorlevel 1 (
    set "MSG="
    set /p MSG="Nhap commit message (Enter de dung mac dinh 'Update'): "
    if "!MSG!"=="" set "MSG=Update"

    git commit -m "!MSG!"
    if errorlevel 1 (
        echo [LOI] Commit that bai.
        pause
        exit /b 1
    )
) else (
    echo Khong co thay doi nao can commit -^> chi pull + push.
)

echo.
echo --- Dang pull code moi nhat tu GitHub (rebase) ---
git pull --rebase
if errorlevel 1 (
    echo [LOI] Pull that bai, co the do xung dot. Mo VS Code de xu ly thu cong roi chay lai.
    pause
    exit /b 1
)

echo.
echo --- Dang push len GitHub ---
git push
if errorlevel 1 (
    echo [LOI] Push that bai. Kiem tra ket noi mang / quyen truy cap repo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   HOAN TAT! Da push code len GitHub thanh cong.
echo ============================================================
pause
