"""
Climate Analysis Dashboard
Interactive web application to visualize climate data and analysis results
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Climate Analysis Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load climate data from CSV files"""
    data_paths = {
        'global': 'data/climate_data.csv',
        'regional': 'data/regional_climate_data.csv'
    }

    data = {}
    for key, path in data_paths.items():
        try:
            if os.path.exists(path):
                data[key] = pd.read_csv(path, parse_dates=['Date'])
                data[key]['Date'] = pd.to_datetime(data[key]['Date'])
            else:
                st.warning(f"Data file not found: {path}")
                data[key] = None
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
            data[key] = None

    return data

def create_temperature_trend_chart(df, title="Temperature Trends"):
    """Create interactive temperature trend chart"""
    if df is None or 'Temperature_C' not in df.columns:
        return None

    fig = px.line(df, x='Date', y='Temperature_C',
                  title=title,
                  labels={'Temperature_C': 'Temperature (°C)', 'Date': 'Date'},
                  line_shape='linear')

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified'
    )

    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    if df is None:
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r')

    return fig

def create_seasonal_analysis(df):
    """Create seasonal analysis charts"""
    if df is None or 'Temperature_C' not in df.columns:
        return None

    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Monthly averages
    monthly_avg = df.groupby('Month')['Temperature_C'].mean().reset_index()

    fig = px.bar(monthly_avg, x='Month', y='Temperature_C',
                 title="Average Temperature by Month",
                 labels={'Temperature_C': 'Average Temperature (°C)', 'Month': 'Month'})

    fig.update_xaxes(tickvals=list(range(1, 13)),
                     ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    return fig

def create_regional_comparison(regional_df):
    """Create regional temperature comparison"""
    if regional_df is None or 'Temperature_C' not in regional_df.columns:
        return None

    # Group by region and calculate averages
    regional_avg = regional_df.groupby('Region')['Temperature_C'].mean().reset_index()

    fig = px.bar(regional_avg, x='Region', y='Temperature_C',
                 title="Average Temperature by Region",
                 labels={'Temperature_C': 'Average Temperature (°C)', 'Region': 'Region'},
                 color='Region')

    return fig

def display_metrics(df):
    """Display key climate metrics"""
    if df is None:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Average Temperature",
                 f"{df['Temperature_C'].mean():.2f}°C",
                 f"±{df['Temperature_C'].std():.2f}°C")

    with col2:
        st.metric("Temperature Range",
                 f"{df['Temperature_C'].max() - df['Temperature_C'].min():.2f}°C")

    with col3:
        st.metric("CO₂ Average",
                 f"{df['CO2_ppm'].mean():.1f} ppm" if 'CO2_ppm' in df.columns else "N/A")

    with col4:
        st.metric("Data Points",
                 f"{len(df):,}")

def main():
    """Main dashboard function"""

    # Header
    st.markdown('<div class="main-header">🌡️ Climate Analysis Dashboard</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # Load data
    with st.spinner("Loading climate data..."):
        data = load_data()

    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-header">📊 Navigation</div>',
                       unsafe_allow_html=True)

    page = st.sidebar.radio("Select Analysis",
                           ["Overview", "Temperature Trends", "Correlations",
                            "Seasonal Analysis", "Regional Analysis", "Data Explorer"])

    # Data status
    st.sidebar.markdown("### Data Status")
    if data['global'] is not None:
        st.sidebar.success("✅ Global data loaded")
    else:
        st.sidebar.error("❌ Global data not found")

    if data['regional'] is not None:
        st.sidebar.success("✅ Regional data loaded")
    else:
        st.sidebar.warning("⚠️ Regional data not found")

    # Main content based on selected page
    if page == "Overview":
        st.header("📈 Climate Data Overview")

        if data['global'] is not None:
            display_metrics(data['global'])

            st.subheader("Recent Temperature Data")
            recent_data = data['global'].tail(10)
            st.dataframe(recent_data)

            # Quick temperature chart
            st.subheader("Temperature Overview")
            fig = create_temperature_trend_chart(data['global'].tail(365), "Last Year Temperature Trend")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    elif page == "Temperature Trends":
        st.header("🌡️ Temperature Trends Analysis")

        if data['global'] is not None:
            # Time period selector
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date",
                                         value=data['global']['Date'].min().date(),
                                         min_value=data['global']['Date'].min().date(),
                                         max_value=data['global']['Date'].max().date())

            with col2:
                end_date = st.date_input("End Date",
                                       value=data['global']['Date'].max().date(),
                                       min_value=data['global']['Date'].min().date(),
                                       max_value=data['global']['Date'].max().date())

            # Filter data
            mask = (data['global']['Date'].dt.date >= start_date) & (data['global']['Date'].dt.date <= end_date)
            filtered_data = data['global'][mask]

            if not filtered_data.empty:
                fig = create_temperature_trend_chart(filtered_data, f"Temperature Trends ({start_date} to {end_date})")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Temperature statistics
                st.subheader("Temperature Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average", f"{filtered_data['Temperature_C'].mean():.2f}°C")
                with col2:
                    st.metric("Maximum", f"{filtered_data['Temperature_C'].max():.2f}°C")
                with col3:
                    st.metric("Minimum", f"{filtered_data['Temperature_C'].min():.2f}°C")
            else:
                st.warning("No data available for the selected date range.")

    elif page == "Correlations":
        st.header("🔗 Feature Correlations")

        if data['global'] is not None:
            fig = create_correlation_heatmap(data['global'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Interpretation:**
            - Values close to 1.0 indicate strong positive correlation
            - Values close to -1.0 indicate strong negative correlation
            - Values near 0 indicate weak or no correlation
            """)

    elif page == "Seasonal Analysis":
        st.header("📅 Seasonal Climate Patterns")

        if data['global'] is not None:
            fig = create_seasonal_analysis(data['global'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Additional seasonal insights
            st.subheader("Seasonal Insights")

            df = data['global'].copy()
            df['Season'] = pd.cut(df['Date'].dt.month,
                                bins=[0, 3, 6, 9, 12],
                                labels=['Winter', 'Spring', 'Summer', 'Fall'])

            seasonal_stats = df.groupby('Season')['Temperature_C'].agg(['mean', 'std']).round(2)
            st.dataframe(seasonal_stats)

    elif page == "Regional Analysis":
        st.header("🌍 Regional Climate Comparison")

        if data['regional'] is not None:
            fig = create_regional_comparison(data['regional'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Regional statistics
            st.subheader("Regional Statistics")
            regional_stats = data['regional'].groupby('Region')['Temperature_C'].agg(['mean', 'std', 'min', 'max']).round(2)
            st.dataframe(regional_stats)
        else:
            st.warning("Regional data not available. Run the data generation step first.")

    elif page == "Data Explorer":
        st.header("🔍 Interactive Data Explorer")

        if data['global'] is not None:
            # Column selector
            numeric_columns = data['global'].select_dtypes(include=[np.number]).columns.tolist()
            selected_columns = st.multiselect("Select columns to display",
                                            numeric_columns,
                                            default=['Temperature_C', 'CO2_ppm', 'Rainfall_mm'])

            if selected_columns:
                # Create interactive scatter plot
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X-axis", selected_columns, index=0)
                with col2:
                    y_axis = st.selectbox("Y-axis", selected_columns, index=min(1, len(selected_columns)-1))

                if x_axis and y_axis and x_axis != y_axis:
                    fig = px.scatter(data['global'], x=x_axis, y=y_axis,
                                   title=f"{y_axis} vs {x_axis}",
                                   trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)

            # Raw data table
            st.subheader("Raw Data")
            st.dataframe(data['global'].tail(100))

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit | Climate Analysis Dashboard")

if __name__ == "__main__":
    main()