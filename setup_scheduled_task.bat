@echo off
echo Requesting administrative privileges...
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Administrative privileges confirmed.
) else (
    echo Please run this script as Administrator.
    echo Right-click the batch file and select "Run as administrator"
    pause
    exit /b 1
)

REM Get full path to Python executable
for /f "delims=" %%i in ('where python') do set PYTHON_PATH=%%i
set SCRIPT_DIR=%~dp0
set RUNNER_SCRIPT=%SCRIPT_DIR%stock_runner.py

echo Using Python from: %PYTHON_PATH%
echo Script location: %RUNNER_SCRIPT%
echo.

REM Create the scheduled task
echo Creating scheduled task...
schtasks /Create /TN "StockDataCollector" /TR "\"%PYTHON_PATH%\" \"%RUNNER_SCRIPT%\"" /SC ONSTART /RU SYSTEM /RL HIGHEST /F

if %errorLevel% == 0 (
    echo Task created successfully!
    echo Starting the task...
    schtasks /Run /TN "StockDataCollector"
    
    echo.
    echo Setup complete! The stock data collection will now:
    echo - Run automatically when the system starts
    echo - Run with elevated privileges
    echo - Log to the logs directory
    echo.
    echo You can manage the task in Task Scheduler:
    echo - To open Task Scheduler, run: taskschd.msc
    echo - Look for the task named "StockDataCollector"
    echo.
    echo To check the logs, look in the logs folder:
    echo - logs\stock_runner.log
    echo - logs\stock_data_job.log
) else (
    echo Failed to create the scheduled task.
    echo Please ensure you're running as Administrator.
)

pause 