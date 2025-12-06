@echo off
REM Git Push Script for PyTorchTest2

echo ========================================
echo Git Push Setup for PyTorchTest2
echo ========================================
echo.

REM Check if Git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/download/win
    echo After installation, restart your terminal and run this script again.
    pause
    exit /b 1
)

echo Git found! Proceeding with setup...
echo.

REM Initialize git repository if not already initialized
if not exist ".git" (
    echo Initializing Git repository...
    git init
    echo.
)

REM Add remote if not already added
git remote get-url origin >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Adding remote repository...
    git remote add origin https://github.com/spongebob2409/Robust-Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection.git
    echo.
) else (
    echo Remote already exists. Updating URL...
    git remote set-url origin https://github.com/spongebob2409/Robust-Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection.git
    echo.
)

REM Add all files (respecting .gitignore)
echo Adding files to git...
git add .
echo.

REM Commit changes
echo Committing changes...
set /p commit_msg="Enter commit message (or press Enter for default): "
if "%commit_msg%"=="" set commit_msg=Add quantization evaluation project

git commit -m "%commit_msg%"
echo.

REM Push to GitHub
echo Pushing to GitHub...
echo You may be prompted for your GitHub credentials.
echo.
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Code pushed to GitHub
    echo ========================================
    echo Repository: https://github.com/spongebob2409/Robust-Defense-Against-Quantization-Attacks-in-Large-Language-Models-Through-Weight-Noise-Injection
) else (
    echo.
    echo ========================================
    echo Push failed. Trying 'master' branch...
    echo ========================================
    git push -u origin master
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo SUCCESS! Code pushed to GitHub
    ) else (
        echo.
        echo ERROR: Push failed. Please check:
        echo 1. GitHub credentials
        echo 2. Repository access permissions
        echo 3. Internet connection
        echo.
        echo You can try manually:
        echo   git push -u origin main
        echo or
        echo   git push -u origin master
    )
)

echo.
pause
