How to View Your Climate Data on the Web
Quick Start
Generate the data first (if you haven't already):

python climate_analysis/main.py --data-only
Run the web dashboard:

# Option 1: Use the batch file (easiest)
run_dashboard.bat

# Option 2: Manual
cd climate_analysis
streamlit run app.py
Open your browser to: http://localhost:8501

What You'll See
The dashboard includes:

📈 Overview: Key climate metrics and recent data
🌡️ Temperature Trends: Interactive charts with date range selection
🔗 Correlations: See how climate variables relate to each other
📅 Seasonal Analysis: Monthly and seasonal patterns
🌍 Regional Analysis: Compare data across regions
🔍 Data Explorer: Create custom charts and explore raw data
Features
Interactive Charts: Hover, zoom, and filter data
Date Range Selection: Focus on specific time periods
Real-time Updates: Charts update as you change settings
Responsive Design: Works on desktop and mobile
Need Help?
If the dashboard doesn't load:

Make sure you've run the data generation step first
Check that Streamlit is installed: pip install streamlit
Try a different port: streamlit run app.py --server.port 8502
The dashboard will automatically detect and load your climate data fil


Results:
![WhatsApp Image 2026-04-03 at 11 23 44 AM (1)](https://github.com/user-attachments/assets/9a000cae-332f-4459-bb03-d5649936e84c)

![WhatsApp Image 2026-04-03 at 11 23 45 AM](https://github.com/user-attachments/assets/6f6da340-759f-474b-a004-c37ded556c61)
![WhatsApp Image 2026-04-03 at 11 23 45 AM (1)](https://github.com/user-attachments/assets/23268655-7e7e-47f2-be63-92566bef657f)



