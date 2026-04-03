# How to View Your Climate Data on the Web

## Quick Start

1. **Generate the data first** (if you haven't already):
   ```bash
   python climate_analysis/main.py --data-only
   ```

2. **Run the web dashboard**:
   ```bash
   # Option 1: Use the batch file (easiest)
   run_dashboard.bat

   # Option 2: Manual
   cd climate_analysis
   streamlit run app.py
   ```

3. **Open your browser** to: http://localhost:8501

## What You'll See

The dashboard includes:

- **📈 Overview**: Key climate metrics and recent data
- **🌡️ Temperature Trends**: Interactive charts with date range selection
- **🔗 Correlations**: See how climate variables relate to each other
- **📅 Seasonal Analysis**: Monthly and seasonal patterns
- **🌍 Regional Analysis**: Compare data across regions
- **🔍 Data Explorer**: Create custom charts and explore raw data

## Features

- **Interactive Charts**: Hover, zoom, and filter data
- **Date Range Selection**: Focus on specific time periods
- **Real-time Updates**: Charts update as you change settings
- **Responsive Design**: Works on desktop and mobile

## Need Help?

If the dashboard doesn't load:
1. Make sure you've run the data generation step first
2. Check that Streamlit is installed: `pip install streamlit`
3. Try a different port: `streamlit run app.py --server.port 8502`

The dashboard will automatically detect and load your climate data files!