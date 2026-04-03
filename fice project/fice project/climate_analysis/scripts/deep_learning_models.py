"""
Deep Learning Models for Climate Forecasting
Includes LSTM, CNN-LSTM, and other neural network architectures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, ConvLSTM2D, Reshape, TimeDistributed, Conv1D


class ClimateDeepLearning:
    """Deep learning models for climate prediction by region/place"""
    
    def __init__(self, data_path='data/climate_data.csv', target_variable='Temperature_C', lookback=30, regional=False):
        """
        Initialize deep learning models
        Args:
            data_path: Path to climate dataset
            target_variable: Variable to predict
            lookback: Number of previous time steps
            regional: Whether to use regional data with place names
        """
        self.data_path = data_path
        self.target_variable = target_variable
        self.lookback = lookback
        self.regional = regional
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scalers_by_place = {}  # Store scalers for each place/region
        
        print(f"Data loaded: {self.df.shape[0]} rows")
        print(f"Target variable: {target_variable}")
        print(f"Lookback window: {lookback}")
        
        if self.regional and 'Region' in self.df.columns:
            print(f"Regional mode enabled")
            self.places = self.df['Region'].unique()
            print(f"Places/Regions found: {list(self.places)}")
    
    def prepare_sequences_by_place(self, test_size=0.2):
        """
        Prepare sequences for LSTM organized by place/region
        Returns dict with place names as keys
        """
        print(f"\nPreparing sequences by place/region...")
        
        sequences_by_place = {}
        
        for place in self.places:
            print(f"\n{'='*50}")
            print(f"REGION: {place}")
            print(f"{'='*50}")
            
            # Filter data for this place
            place_data = self.df[self.df['Region'] == place].copy().sort_values('Date')
            
            # Select numeric features
            numeric_cols = [col for col in place_data.select_dtypes(include=[np.number]).columns 
                          if col not in ['Region']]
            
            print(f"Data points: {len(place_data)}")
            print(f"Date range: {place_data['Date'].min()} to {place_data['Date'].max()}")
            
            # Scale features
            scaler_X = MinMaxScaler(feature_range=(0, 1))
            X = place_data[numeric_cols].values
            X_scaled = scaler_X.fit_transform(X)
            
            # Scale target
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            y = place_data[self.target_variable].values.reshape(-1, 1)
            y_scaled = scaler_y.fit_transform(y)
            
            # Create sequences
            X_seq = []
            y_seq = []
            
            for i in range(len(X_scaled) - self.lookback):
                X_seq.append(X_scaled[i:i + self.lookback])
                y_seq.append(y_scaled[i + self.lookback, 0])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Split into train and test
            split_idx = int(len(X_seq) * (1 - test_size))
            
            X_train = X_seq[:split_idx]
            y_train = y_seq[:split_idx]
            X_test = X_seq[split_idx:]
            y_test = y_seq[split_idx:]
            
            sequences_by_place[place] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            }
            
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Store scalers
            self.scalers_by_place[place] = {'X': scaler_X, 'y': scaler_y}
        
        return sequences_by_place
    
    def prepare_sequences(self, test_size=0.2):
        """
        Prepare sequences for LSTM
        Args:
            test_size: Test set proportion
        """
        print(f"\nPreparing sequences...")
        
        # Select features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Scale features
        X = self.df[numeric_cols].values
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Scale target
        target_idx = numeric_cols.index(self.target_variable)
        y = self.df[self.target_variable].values.reshape(-1, 1)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(len(X_scaled) - self.lookback):
            X_seq.append(X_scaled[i:i + self.lookback])
            y_seq.append(y_scaled[i + self.lookback, 0])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split into train and test
        split_idx = int(len(X_seq) * (1 - test_size))
        
        X_train = X_seq[:split_idx]
        y_train = y_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_test = y_seq[split_idx:]
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def build_lstm_model(self, units=50, dropout_rate=0.2, n_features=6):
        """Build LSTM model"""
        model = models.Sequential([
            LSTM(units=units, activation='relu', return_sequences=True, input_shape=(self.lookback, n_features)),
            Dropout(dropout_rate),
            LSTM(units=units, activation='relu', return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_cnn_lstm_model(self, cnn_filters=32, lstm_units=50, dropout_rate=0.2, n_features=6):
        """Build CNN-LSTM hybrid model"""
        model = models.Sequential([
            Conv1D(filters=cnn_filters, kernel_size=3, activation='relu', input_shape=(self.lookback, n_features)),
            Dropout(dropout_rate),
            LSTM(units=lstm_units, activation='relu', return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=lstm_units, activation='relu'),
            Dropout(dropout_rate),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_bidirectional_lstm_model(self, units=50, dropout_rate=0.2, n_features=6):
        """Build Bidirectional LSTM model"""
        from tensorflow.keras.layers import Bidirectional
        
        model = models.Sequential([
            Bidirectional(LSTM(units=units, activation='relu', return_sequences=True), 
                         input_shape=(self.lookback, n_features)),
            Dropout(dropout_rate),
            Bidirectional(LSTM(units=units, activation='relu')),
            Dropout(dropout_rate),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_stacked_lstm_model(self, units=50, dropout_rate=0.2, n_features=6):
        """Build Stacked LSTM model"""
        model = models.Sequential([
            LSTM(units=units, activation='relu', return_sequences=True, input_shape=(self.lookback, n_features)),
            Dropout(dropout_rate),
            LSTM(units=units, activation='relu', return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=units, activation='relu', return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=units, activation='relu'),
            Dropout(dropout_rate),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_models_by_place(self, epochs=30, batch_size=32):
        """Train deep learning models for each place/region"""
        print("\n" + "="*70)
        print("TRAINING DEEP LEARNING MODELS BY PLACE/REGION")
        print("="*70)
        
        if not self.regional:
            print("ERROR: Regional mode not enabled!")
            return None, None
        
        # Prepare sequences by place
        sequences_by_place = self.prepare_sequences_by_place()
        
        # Callback for early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        all_results = {}
        all_predictions = {}
        
        for place in self.places:
            print(f"\n" + "="*70)
            print(f"ANALYZING: {place}")
            print("="*70)
            
            data = sequences_by_place[place]
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            scaler_y = data['scaler_y']
            
            # Get number of features for this place
            n_features = X_train.shape[2]
            
            # Build and train LSTM model for this place
            lstm_model = self.build_lstm_model(units=50, dropout_rate=0.2, n_features=n_features)
            
            print(f"\nTraining LSTM model for {place}...")
            history = lstm_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Make predictions
            y_pred_train_scaled = lstm_model.predict(X_train, verbose=0)
            y_pred_test_scaled = lstm_model.predict(X_test, verbose=0)
            
            # Inverse transform predictions
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)
            y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            mse = mean_squared_error(y_test_actual, y_pred_test)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_actual, y_pred_test)
            r2 = r2_score(y_test_actual, y_pred_test)
            
            all_results[place] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'model': lstm_model,
                'history': history
            }
            
            all_predictions[place] = {
                'actual': y_test_actual,
                'predicted': y_pred_test,
                'history': history
            }
            
            print(f"\n{place} Performance:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
        
        # Print summary
        print(f"\n" + "="*70)
        print("SUMMARY BY PLACE")
        print("="*70)
        
        summary_df = pd.DataFrame({
            place: all_results[place] 
            for place in self.places if place in all_results
        }).T
        
        if 'RMSE' in summary_df.columns:
            summary_df = summary_df[['RMSE', 'MAE', 'R2']].sort_values('R2', ascending=False)
            print(summary_df)
        
        return all_results, all_predictions
    
    def train_models(self, epochs=50, batch_size=32):
        """Train deep learning models"""
        print("\n" + "="*60)
        print("TRAINING DEEP LEARNING MODELS")
        print("="*60)
        
        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_sequences()
        
        # Get number of features
        n_features = X_train.shape[2]
        
        # Callback for early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        models_to_train = {
            'LSTM': self.build_lstm_model(units=50, dropout_rate=0.2, n_features=n_features),
            'CNN-LSTM': self.build_cnn_lstm_model(cnn_filters=32, lstm_units=50, dropout_rate=0.2, n_features=n_features),
            'Bidirectional LSTM': self.build_bidirectional_lstm_model(units=50, dropout_rate=0.2, n_features=n_features),
            'Stacked LSTM': self.build_stacked_lstm_model(units=50, dropout_rate=0.2, n_features=n_features)
        }
        
        results = {}
        predictions = {}
        
        for model_name, model in models_to_train.items():
            print(f"\n{'='*40}")
            print(f"Training {model_name}...")
            print(f"{'='*40}")
            
            print(f"\nModel Summary:")
            model.summary()
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=1
            )
            
            # Make predictions
            y_pred_train_scaled = model.predict(X_train, verbose=0)
            y_pred_test_scaled = model.predict(X_test, verbose=0)
            
            # Inverse transform predictions
            y_pred_test = self.scaler_y.inverse_transform(y_pred_test_scaled)
            y_test_actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            mse = mean_squared_error(y_test_actual, y_pred_test)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_actual, y_pred_test)
            r2 = r2_score(y_test_actual, y_pred_test)
            
            results[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Final_Train_Loss': history.history['loss'][-1],
                'Final_Val_Loss': history.history['val_loss'][-1]
            }
            
            predictions[model_name] = (y_test_actual, y_pred_test, history)
            
            print(f"\n{model_name} Results:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
        
        # Summary
        print("\n" + "-"*60)
        print("MODEL COMPARISON")
        print("-"*60)
        
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('R2', ascending=False)
        print(results_df)
        
        return results_df, predictions
    
    def visualize_training_history(self, predictions):
        """Visualize training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Deep Learning Models - Training History', fontsize=16, fontweight='bold')
        
        for idx, (ax, (model_name, (_, _, history))) in enumerate(zip(axes.flat, list(predictions.items()))):
            ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
            ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/visualizations/training_history.png', dpi=300, bbox_inches='tight')
        print("Saved: data/visualizations/training_history.png")
        plt.close()
    
    def visualize_predictions(self, predictions):
        """Visualize DL model predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Deep Learning Predictions: {self.target_variable}', fontsize=16, fontweight='bold')
        
        for idx, (ax, (model_name, (y_test, y_pred, _))) in enumerate(zip(axes.flat, predictions.items())):
            ax.scatter(y_test, y_pred, alpha=0.5, s=20, color='darkblue')
            
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}', fontweight='bold')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/visualizations/dl_predictions.png', dpi=300, bbox_inches='tight')
        print("Saved: data/visualizations/dl_predictions.png")
        plt.close()
    
    def visualize_time_series(self, predictions):
        """Visualize predictions as time series"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Time Series Predictions: {self.target_variable}', fontsize=16, fontweight='bold')
        
        for idx, (ax, (model_name, (y_test, y_pred, _))) in enumerate(zip(axes.flat, predictions.items())):
            time_steps = range(len(y_test))
            ax.plot(time_steps, y_test, label='Actual', linewidth=2, alpha=0.7)
            ax.plot(time_steps, y_pred, label='Predicted', linewidth=2, alpha=0.7)
            
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(self.target_variable)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/visualizations/dl_timeseries.png', dpi=300, bbox_inches='tight')
        print("Saved: data/visualizations/dl_timeseries.png")
        plt.close()


def main():
    """Train deep learning models"""
    
    # Option 1: Global analysis
    print("\n" + "="*70)
    print("GLOBAL CLIMATE ANALYSIS")
    print("="*70)
    
    dl_models = ClimateDeepLearning('data/climate_data.csv', target_variable='Temperature_C', lookback=30)
    
    # Train models
    results_df, predictions = dl_models.train_models(epochs=30, batch_size=32)
    
    # Visualizations
    dl_models.visualize_training_history(predictions)
    dl_models.visualize_predictions(predictions)
    dl_models.visualize_time_series(predictions)
    
    # Option 2: Regional/Place-based analysis
    print("\n\n" + "="*70)
    print("REGIONAL/PLACE-BASED CLIMATE ANALYSIS")
    print("="*70)
    
    try:
        dl_models_regional = ClimateDeepLearning(
            'data/regional_climate_data.csv', 
            target_variable='Temperature_C', 
            lookback=30,
            regional=True
        )
        
        # Train models by place
        results_by_place, predictions_by_place = dl_models_regional.train_models_by_place(epochs=20, batch_size=16)
        
        # Display results by place
        if results_by_place and predictions_by_place:
            print("\n" + "="*70)
            print("PLACE-BASED PREDICTIONS")
            print("="*70)
            
            for place in sorted(predictions_by_place.keys()):
                preds = predictions_by_place[place]
                y_test = preds['actual']
                y_pred = preds['predicted']
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                print(f"\n{place.upper()}")
                print(f"  Predictions: {len(y_pred)} samples")
                print(f"  Avg Temperature: {y_test.mean():.2f}°C")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  R²: {r2:.4f}")
    
    except FileNotFoundError:
        print("Regional dataset not found. Skipping regional analysis.")
    
    print("\n" + "="*70)
    print("DEEP LEARNING MODEL TRAINING COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    main()
