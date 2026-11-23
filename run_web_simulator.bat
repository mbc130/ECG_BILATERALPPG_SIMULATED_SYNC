@echo off
title Cardisense Web Simulator
echo ========================================================
echo Starting Cardisense Web Dashboard Simulator...
echo Opening in default browser at http://localhost:8050
echo ========================================================
cd /d "%~dp0"
python web_dashboard.py
pause
