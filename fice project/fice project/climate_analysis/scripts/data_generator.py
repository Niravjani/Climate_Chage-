"""
Climate Data Generator
Generates synthetic climate datasets for analysis and model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class ClimateDataGenerator:
    """Generate realistic synthetic climate data"""
    
    def __init__(self, start_date='2010-01-01', end_date='2023-12-31', freq='D'):
        """
        Initialize data generator
        Args:
            start_date: Start date for time series
            end_date: End date for time series
            freq: Frequency of data ('D' for daily, 'M' for monthly)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.freq = freq
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq=freq)
        np.random.seed(42)
    
    def generate_temperature(self, base_temp=15, trend=0.02, seasonal_amplitude=8):
        """
        Generate temperature data with trend and seasonality
        Args:
            base_temp: Base temperature (Celsius)
            trend: Linear trend component
            seasonal_amplitude: Seasonal variation amplitude
        """
        n = len(self.date_range)
        
        # Linear trend (climate warming)
        trend_component = np.linspace(0, trend * n, n)
        
        # Seasonal component (annual cycle)
        day_of_year = np.array([d.dayofyear for d in self.date_range])
        seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365)
        
        # Noise
        noise = np.random.normal(0, 1, n)
        
        # Combine components
        temperature = base_temp + trend_component + seasonal_component + noise
        
        return temperature
    
    def generate_rainfall(self, mean_rainfall=50, std_rainfall=30):
        """
        Generate rainfall data with seasonal patterns
        Args:
            mean_rainfall: Mean monthly rainfall (mm)
            std_rainfall: Standard deviation
        """
        n = len(self.date_range)
        
        # Seasonal pattern (monsoon effect)
        month = np.array([d.month for d in self.date_range])
        seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * month / 12)
        
        # Generate rainfall
        rainfall = np.random.gamma(shape=2, scale=mean_rainfall/2, size=n)
        rainfall = rainfall * seasonal_factor
        rainfall = np.maximum(rainfall, 0)  # No negative rainfall
        
        return rainfall
    
    def generate_co2(self, base_co2=360, rate_increase=2.5):
        """Generate CO2 concentration data"""
        n = len(self.date_range)
        years = np.array([(d - self.date_range[0]).days / 365.25 for d in self.date_range])
        
        # CO2 increases over time
        co2 = base_co2 + rate_increase * years + np.random.normal(0, 0.3, n)
        
        return co2
    
    def generate_air_quality(self, base_aqi=50, trend=0.01):
        """Generate Air Quality Index data"""
        n = len(self.date_range)
        
        # Linear trend
        trend_component = np.linspace(0, trend * n, n)
        
        # Seasonal variation (worse in winter)
        month = np.array([d.month for d in self.date_range])
        seasonal = 15 * np.cos(2 * np.pi * month / 12)
        
        # Noise
        noise = np.random.normal(0, 3, n)
        
        # Combine
        aqi = base_aqi + trend_component + seasonal + noise
        aqi = np.maximum(aqi, 0)
        aqi = np.minimum(aqi, 500)  # Cap at 500
        
        return aqi
    
    def generate_sea_level(self, base_level=0, rate_rise=3.3):
        """Generate sea level rise data (mm per year)"""
        n = len(self.date_range)
        years = np.array([(d - self.date_range[0]).days / 365.25 for d in self.date_range])
        
        # Linear rise with noise
        sea_level = base_level + rate_rise * years + np.random.normal(0, 1, n)
        
        return sea_level
    
    def generate_ice_extent(self, base_extent=100):
        """Generate Arctic ice extent data (declining)"""
        n = len(self.date_range)
        years = np.array([(d - self.date_range[0]).days / 365.25 for d in self.date_range])
        
        # Declining trend
        decline = years * 1.5
        
        # Seasonal variation
        month = np.array([d.month for d in self.date_range])
        seasonal = 10 * np.sin(2 * np.pi * month / 12)
        
        # Noise
        noise = np.random.normal(0, 2, n)
        
        # Combine
        ice_extent = base_extent - decline + seasonal + noise
        ice_extent = np.maximum(ice_extent, 0)
        
        return ice_extent
    
    def create_climate_dataset(self, output_path='data/climate_data.csv'):
        """
        Create complete climate dataset
        Args:
            output_path: Path to save the dataset
        """
        print("Generating climate dataset...")
        
        data = {
            'Date': self.date_range,
            'Temperature_C': self.generate_temperature(),
            'Rainfall_mm': self.generate_rainfall(),
            'CO2_ppm': self.generate_co2(),
            'AirQuality_Index': self.generate_air_quality(),
            'SeaLevel_mm': self.generate_sea_level(),
            'IceExtent_km2': self.generate_ice_extent()
        }
        
        df = pd.DataFrame(data)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Dataset shape: {df.shape}")
        print("\nDataset preview:")
        print(df.head(10))
        print("\nDataset statistics:")
        print(df.describe())
        
        return df
    
    def create_regional_dataset(self, regions=['North_America', 'Europe', 'Asia', 'Africa', 'Oceania'], 
                                output_path='data/regional_climate_data.csv'):
        """Create regional climate dataset"""
        print("Generating regional climate dataset...")
        
        all_data = []
        
        for region in regions:
            # Add regional variations
            temp_offset = np.random.uniform(-5, 5)
            rainfall_factor = np.random.uniform(0.5, 1.5)
            
            data = {
                'Date': self.date_range,
                'Region': region,
                'Temperature_C': self.generate_temperature(base_temp=15 + temp_offset),
                'Rainfall_mm': self.generate_rainfall(mean_rainfall=50) * rainfall_factor,
                'CO2_ppm': self.generate_co2(),
                'AirQuality_Index': self.generate_air_quality()
            }
            
            all_data.append(pd.DataFrame(data))
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Regional dataset saved to {output_path}")
        print(f"Dataset shape: {df.shape}")
        
        return df


def main():
    """Generate climate datasets"""
    generator = ClimateDataGenerator(start_date='2010-01-01', end_date='2023-12-31', freq='D')
    
    # Generate global climate dataset
    global_data = generator.create_climate_dataset('data/climate_data.csv')
    
    # Generate regional climate dataset
    regional_data = generator.create_regional_dataset(
        regions=['North_America', 'Europe', 'Asia', 'Africa', 'Oceania'],
        output_path='data/regional_climate_data.csv'
    )
    
    print("\n" + "="*60)
    print("Data generation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
