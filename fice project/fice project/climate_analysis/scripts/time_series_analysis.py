"""
Time Series Analysis for Climate Forecasting
Includes ARIMA, SARIMA, and Prophet models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Install with: pip install statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")


class ClimateTimeSeriesAnalysis:
    """Time series analysis and forecasting for climate data"""
    
    def __init__(self, data_path='data/climate_data.csv', target_variable='Temperature_C'):
        """
        Initialize time series analysis
        Args:
            data_path: Path to climate dataset
            target_variable: Variable to forecast
        """
        self.data_path = data_path
        self.target_variable = target_variable
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date')
        
        self.ts_data = self.df.set_index('Date')[target_variable]
        
        print(f"Time series loaded: {len(self.ts_data)} observations")
        print(f"Date range: {self.ts_data.index.min()} to {self.ts_data.index.max()}")
        print(f"Target variable: {target_variable}")
    
    def stationarity_test(self):
        """Test for stationarity using Augmented Dickey-Fuller"""
        print("\n" + "="*60)
        print("STATIONARITY TEST (ADF Test)")
        print("="*60)
        
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available")
            return None
        
        result = adfuller(self.ts_data)
        
        print(f"\nAugmented Dickey-Fuller Test for {self.target_variable}:")
        print(f"  ADF Statistic: {result[0]:.6f}")
        print(f"  P-value: {result[1]:.6f}")
        print(f"  Critical Values:")
        for key, value in result[4].items():
            print(f"    {key}: {value:.3f}")
        
        if result[1] <= 0.05:
            print(f"\n[OK] Series is STATIONARY (p-value < 0.05)")
        else:
            print(f"\n[WARNING] Series is NOT STATIONARY (p-value >= 0.05)")
        
        return result
    
    def seasonal_decomposition(self, period=365):
        """Decompose time series into trend, seasonal, and residual"""
        print("\n" + "="*60)
        print("SEASONAL DECOMPOSITION")
        print("="*60)
        
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available")
            return None
        
        print(f"Decomposing with period={period}...")
        decomposition = seasonal_decompose(self.ts_data, model='additive', period=period)
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(15, 10))
        
        axes[0].plot(decomposition.observed, color='steelblue')
        axes[0].set_ylabel('Observed')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(decomposition.trend, color='orange')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(decomposition.seasonal, color='green')
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(decomposition.resid, color='red')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        fig.suptitle(f'Seasonal Decomposition: {self.target_variable}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/visualizations/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
        print("Saved: data/visualizations/seasonal_decomposition.png")
        plt.close()
        
        return decomposition
    
    def acf_pacf_analysis(self):
        """Analyze ACF and PACF"""
        print("\n" + "="*60)
        print("ACF/PACF ANALYSIS")
        print("="*60)
        
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # ACF plot
        plot_acf(self.ts_data, lags=40, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold', fontsize=12)
        
        # PACF plot
        plot_pacf(self.ts_data, lags=40, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('data/visualizations/acf_pacf.png', dpi=300, bbox_inches='tight')
        print("Saved: data/visualizations/acf_pacf.png")
        plt.close()
    
    def train_arima_models(self, test_size=0.2):
        """Train ARIMA and SARIMA models"""
        print("\n" + "="*60)
        print("ARIMA/SARIMA MODELS")
        print("="*60)
        
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available")
            return None, None
        
        # Split data
        split_idx = int(len(self.ts_data) * (1 - test_size))
        train_data = self.ts_data[:split_idx]
        test_data = self.ts_data[split_idx:]
        
        print(f"\nTrain size: {len(train_data)}, Test size: {len(test_data)}")
        
        # ARIMA(1,1,1)
        print("\n" + "-"*40)
        print("Training ARIMA(1,1,1)...")
        print("-"*40)
        
        arima_model = ARIMA(train_data, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        print(arima_fit.summary())
        
        arima_pred = arima_fit.forecast(steps=len(test_data))
        arima_rmse = np.sqrt(mean_squared_error(test_data, arima_pred))
        arima_mae = mean_absolute_error(test_data, arima_pred)
        arima_r2 = r2_score(test_data, arima_pred)
        
        print(f"ARIMA(1,1,1) Results:")
        print(f"  RMSE: {arima_rmse:.4f}")
        print(f"  MAE: {arima_mae:.4f}")
        print(f"  R²: {arima_r2:.4f}")
        
        # SARIMA(1,1,1)x(1,1,1,365)
        print("\n" + "-"*40)
        print("Training SARIMA(1,1,1)x(1,1,1,365)...")
        print("-"*40)
        
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 365), 
                                   enforce_stationarity=False, enforce_invertibility=False)
            sarima_fit = sarima_model.fit(disp=False)
            print(sarima_fit.summary())
            
            sarima_pred = sarima_fit.forecast(steps=len(test_data))
            sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_pred))
            sarima_mae = mean_absolute_error(test_data, sarima_pred)
            sarima_r2 = r2_score(test_data, sarima_pred)
            
            print(f"SARIMA(1,1,1)x(1,1,1,365) Results:")
            print(f"  RMSE: {sarima_rmse:.4f}")
            print(f"  MAE: {sarima_mae:.4f}")
            print(f"  R²: {sarima_r2:.4f}")
        
        except Exception as e:
            print(f"SARIMA training failed: {e}")
            sarima_pred = arima_pred
            sarima_rmse = arima_rmse
            sarima_mae = arima_mae
            sarima_r2 = arima_r2
        
        results = {
            'ARIMA': {'RMSE': arima_rmse, 'MAE': arima_mae, 'R2': arima_r2},
            'SARIMA': {'RMSE': sarima_rmse, 'MAE': sarima_mae, 'R2': sarima_r2}
        }
        
        # Visualize predictions
        fig, ax = plt.subplots(figsize=(15, 6))
        
        ax.plot(train_data.index, train_data.values, label='Train Data', linewidth=2)
        ax.plot(test_data.index, test_data.values, label='Actual Test Data', linewidth=2)
        ax.plot(test_data.index, arima_pred, label='ARIMA Forecast', linewidth=2, linestyle='--')
        ax.plot(test_data.index, sarima_pred, label='SARIMA Forecast', linewidth=2, linestyle='--')
        
        ax.set_title(f'ARIMA/SARIMA Forecasts: {self.target_variable}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel(self.target_variable)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('data/visualizations/arima_sarima.png', dpi=300, bbox_inches='tight')
        print("\nSaved: data/visualizations/arima_sarima.png")
        plt.close()
        
        return results, {'test': test_data, 'arima': arima_pred, 'sarima': sarima_pred}
    
    def train_prophet_models(self, test_size=0.2, periods=None):
        """Train Prophet model"""
        print("\n" + "="*60)
        print("FACEBOOK PROPHET MODEL")
        print("="*60)
        
        if not PROPHET_AVAILABLE:
            print("Prophet not available")
            return None
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': self.ts_data.index,
            'y': self.ts_data.values
        })
        
        # Split data
        split_idx = int(len(prophet_df) * (1 - test_size))
        train_df = prophet_df[:split_idx]
        test_df = prophet_df[split_idx:]
        
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        # Train Prophet
        print("\nTraining Prophet model...")
        
        model = Prophet(yearly_seasonality=True, daily_seasonality=False)
        model.fit(train_df)
        
        # Make forecast
        if periods is None:
            periods = len(test_df)
        
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Get predictions for test set
        test_forecast = forecast[-len(test_df):].copy()
        prophet_pred = test_forecast['yhat'].values
        
        # Calculate metrics
        prophet_rmse = np.sqrt(mean_squared_error(test_df['y'].values, prophet_pred))
        prophet_mae = mean_absolute_error(test_df['y'].values, prophet_pred)
        prophet_r2 = r2_score(test_df['y'].values, prophet_pred)
        
        print(f"\nProphet Results:")
        print(f"  RMSE: {prophet_rmse:.4f}")
        print(f"  MAE: {prophet_mae:.4f}")
        print(f"  R²: {prophet_r2:.4f}")
        
        # Plot
        fig = model.plot(forecast, figsize=(15, 6))
        fig.suptitle(f'Prophet Forecast: {self.target_variable}', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('data/visualizations/prophet_forecast.png', dpi=300, bbox_inches='tight')
        print("Saved: data/visualizations/prophet_forecast.png")
        plt.close()
        
        # Plot components
        fig = model.plot_components(forecast, figsize=(15, 10))
        fig.suptitle(f'Prophet Components: {self.target_variable}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/visualizations/prophet_components.png', dpi=300, bbox_inches='tight')
        print("Saved: data/visualizations/prophet_components.png")
        plt.close()
        
        return {
            'Prophet': {'RMSE': prophet_rmse, 'MAE': prophet_mae, 'R2': prophet_r2}
        }, {'test': test_df['y'].values, 'prophet': prophet_pred}


def main():
    """Run time series analysis"""
    ts_analysis = ClimateTimeSeriesAnalysis('data/climate_data.csv', target_variable='Temperature_C')
    
    # Stationarity test
    ts_analysis.stationarity_test()
    
    # Seasonal decomposition
    ts_analysis.seasonal_decomposition()
    
    # ACF/PACF analysis
    ts_analysis.acf_pacf_analysis()
    
    # Train ARIMA/SARIMA
    arima_results, arima_preds = ts_analysis.train_arima_models()
    
    # Train Prophet
    prophet_results, prophet_preds = ts_analysis.train_prophet_models()
    
    # Combine results
    if arima_results and prophet_results:
        all_results = {**arima_results, **prophet_results}
        results_df = pd.DataFrame(all_results).T
        results_df = results_df.sort_values('R2', ascending=False)
        
        print("\n" + "-"*60)
        print("TIME SERIES MODELS COMPARISON")
        print("-"*60)
        print(results_df)
    
    print("\n" + "="*60)
    print("TIME SERIES ANALYSIS COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()
