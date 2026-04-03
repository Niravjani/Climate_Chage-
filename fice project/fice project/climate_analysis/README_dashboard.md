# Climate Analysis Dashboard

An interactive web dashboard to visualize and explore climate data using Streamlit.

## Features

- **📈 Overview**: Key metrics and recent data summary
- **🌡️ Temperature Trends**: Interactive temperature charts with date range selection
- **🔗 Correlations**: Feature correlation heatmap
- **📅 Seasonal Analysis**: Monthly and seasonal climate patterns
- **🌍 Regional Analysis**: Compare climate data across different regions
- **🔍 Data Explorer**: Interactive scatter plots and raw data exploration

## Prerequisites

Make sure you have run the climate analysis pipeline first to generate the data:

```bash
python climate_analysis/main.py --data-only
```

## Installation

1. Install the required packages:
```bash
pip install -r climate_analysis/requirements.txt
```

## Running the Dashboard

### Option 1: Using the batch file (Windows)
```bash
run_dashboard.bat
```

### Option 2: Manual execution
```bash
# Activate virtual environment
& ".venv\Scripts\Activate.ps1"

# Navigate to climate_analysis directory
cd climate_analysis

# Run the dashboard
streamlit run app.py
```

## Accessing the Dashboard

Once running, the dashboard will be available at:
- **Local**: http://localhost:8501
- **Network**: http://0.0.0.0:8501 (if accessible on your network)

## Dashboard Features

### Navigation
Use the sidebar to navigate between different analysis sections:

1. **Overview**: Quick summary of key metrics and recent data
2. **Temperature Trends**: View temperature changes over time with customizable date ranges
3. **Correlations**: See how different climate variables relate to each other
4. **Seasonal Analysis**: Analyze monthly and seasonal patterns
5. **Regional Analysis**: Compare climate data across different geographical regions
6. **Data Explorer**: Create custom scatter plots and explore raw data

### Interactive Charts
- Hover over data points for detailed information
- Zoom and pan on charts
- Use date range selectors for focused analysis
- Click legends to show/hide data series

## Data Requirements

The dashboard expects these CSV files in the `data/` directory:
- `climate_data.csv`: Global climate dataset
- `regional_climate_data.csv`: Regional climate data

Run the data generation step if these files don't exist.

## Troubleshooting

- **Data not loading**: Make sure you've run the data generation step first
- **Port already in use**: Change the port in the run command: `streamlit run app.py --server.port 8502`
- **Permission errors**: Make sure your virtual environment is activated

## Technologies Used

- **Streamlit**: Web framework for data applications
- **Plotly**: Interactive charting library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations