"""
Climate Change Analysis - Main Orchestration Script
Runs all analysis and modeling pipelines
"""

import os
import sys
import argparse
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(" "*20 + title)
    print("="*70 + "\n")

def main():
    """Main orchestration function"""
    
    parser = argparse.ArgumentParser(description='Climate Change Analysis and Forecasting')
    parser.add_argument('--data-only', action='store_true', help='Generate data only')
    parser.add_argument('--eda-only', action='store_true', help='Run EDA only')
    parser.add_argument('--ml-only', action='store_true', help='Run ML models only')
    parser.add_argument('--dl-only', action='store_true', help='Run deep learning models only')
    parser.add_argument('--ts-only', action='store_true', help='Run time series analysis only')
    parser.add_argument('--all', action='store_true', default=True, help='Run all analyses (default)')
    
    args = parser.parse_args()
    
    print_header("CLIMATE CHANGE ANALYSIS & FORECASTING PROJECT")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create necessary directories
    os.makedirs('data/visualizations', exist_ok=True)
    
    # Step 1: Generate Data
    if args.data_only or args.all:
        print_header("STEP 1: DATA GENERATION")
        try:
            from scripts.data_generator import ClimateDataGenerator
            
            generator = ClimateDataGenerator(start_date='2010-01-01', end_date='2023-12-31', freq='D')
            global_data = generator.create_climate_dataset('data/climate_data.csv')
            regional_data = generator.create_regional_dataset(
                regions=['North_America', 'Europe', 'Asia', 'Africa', 'Oceania'],
                output_path='data/regional_climate_data.csv'
            )
            print("\n[OK] Data generation completed successfully!")
        except Exception as e:
            print(f"[ERROR] Error in data generation: {e}")
            if args.data_only:
                return
    
    if args.data_only:
        print("\nData generation complete. Exiting...")
        return
    
    # Step 2: Exploratory Data Analysis
    if args.eda_only or args.all:
        print_header("STEP 2: EXPLORATORY DATA ANALYSIS")
        try:
            from scripts.eda_analysis import ClimateEDAAnalyzer
            
            analyzer = ClimateEDAAnalyzer('data/climate_data.csv')
            analyzer.basic_statistics()
            analyzer.correlation_analysis()
            analyzer.temporal_analysis()
            analyzer.seasonal_analysis()
            analyzer.outlier_detection()
            analyzer.distribution_analysis()
            analyzer.create_visualizations()
            print("\n[OK] EDA completed successfully!")
        except Exception as e:
            print(f"[ERROR] Error in EDA: {e}")
            import traceback
            traceback.print_exc()
    
    if args.eda_only:
        print("\nEDA complete. Exiting...")
        return
    
    # Step 3: Machine Learning Models
    if args.ml_only or args.all:
        print_header("STEP 3: MACHINE LEARNING MODELS")
        try:
            from scripts.ml_models import ClimateMLModels
            
            ml_models = ClimateMLModels('data/climate_data.csv', target_variable='Temperature_C')
            results_df, data = ml_models.train_models()
            ml_models.feature_importance()
            print("\n[OK] ML model training completed successfully!")
        except Exception as e:
            print(f"[ERROR] Error in ML models: {e}")
            import traceback
            traceback.print_exc()
    
    if args.ml_only:
        print("\nML training complete. Exiting...")
        return
    
    # Step 4: Deep Learning Models
    if args.dl_only or args.all:
        print_header("STEP 4: DEEP LEARNING MODELS")
        try:
            from scripts.deep_learning_models import ClimateDeepLearning
            
            print("Initializing Deep Learning models...")
            dl_models = ClimateDeepLearning('data/climate_data.csv', target_variable='Temperature_C', lookback=30)
            
            print("Starting model training...")
            results_df, predictions = dl_models.train_models(epochs=30, batch_size=32)
            
            print("\nGenerating visualizations...")
            dl_models.visualize_training_history(predictions)
            dl_models.visualize_predictions(predictions)
            dl_models.visualize_time_series(predictions)
            
            print("\n[OK] Deep learning model training completed successfully!")
        except Exception as e:
            print(f"[ERROR] Error in deep learning models: {e}")
            import traceback
            traceback.print_exc()
    
    if args.dl_only:
        print("\nDeep learning training complete. Exiting...")
        return
    
    # Step 5: Time Series Analysis
    if args.ts_only or args.all:
        print_header("STEP 5: TIME SERIES FORECASTING")
        try:
            from scripts.time_series_analysis import ClimateTimeSeriesAnalysis
            
            ts_analysis = ClimateTimeSeriesAnalysis('data/climate_data.csv', target_variable='Temperature_C')
            ts_analysis.stationarity_test()
            ts_analysis.seasonal_decomposition()
            ts_analysis.acf_pacf_analysis()
            
            arima_results, arima_preds = ts_analysis.train_arima_models()
            prophet_results, prophet_preds = ts_analysis.train_prophet_models()
            
            print("\n[OK] Time series analysis completed successfully!")
        except Exception as e:
            print(f"[ERROR] Error in time series analysis: {e}")
            import traceback
            traceback.print_exc()
    
    if args.ts_only:
        print("\nTime series analysis complete. Exiting...")
        return
    
    # Final Summary
    print_header("ANALYSIS COMPLETE")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated Outputs:")
    print("  * data/climate_data.csv - Global climate dataset")
    print("  * data/regional_climate_data.csv - Regional climate data")
    print("  * data/visualizations/ - All visualization files")
    print("\nKey Insights:")
    print("  1. Exploratory analysis reveals climate trends and patterns")
    print("  2. Machine learning models provide temperature predictions")
    print("  3. Deep learning captures complex temporal patterns")
    print("  4. Time-series forecasting enables climate projections")
    print("\nNext Steps:")
    print("  * Review generated visualizations in data/visualizations/")
    print("  * Analyze model performance metrics and predictions")
    print("  * Examine feature importance for climate drivers")
    print("  * Compare forecasts from different modeling approaches")
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()
