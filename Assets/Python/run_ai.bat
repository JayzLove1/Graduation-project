@echo off
echo ========================================
echo  Python AI Training Script Launcher
echo ========================================
echo.
echo Script path: %~dp0rl_maze_ai.py
echo.
python -u "%~dp0rl_maze_ai.py"
echo.
echo ========================================
echo  Python process ended with code: %errorlevel%
echo ========================================
pause
