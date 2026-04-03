@echo off
echo Starting Climate Analysis Dashboard...
echo.

REM Activate virtual environment
call "%~dp0..\.venv\Scripts\activate.bat"

REM Change to climate_analysis directory
cd "%~dp0climate_analysis"

REM Install/update streamlit if needed
pip install streamlit==1.28.1

REM Run the Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

pause