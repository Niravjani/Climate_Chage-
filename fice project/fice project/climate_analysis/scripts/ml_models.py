"""
Machine Learning Models for Climate Prediction
Includes Random Forest, XGBoost, SVR, and linear regression models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


class ClimateMLModels:
    """Train and evaluate ML models for climate prediction"""
    
    def __init__(self, data_path='data/climate_data.csv', target_variable='Temperature_C'):
        """
        Initialize ML models
        Args:
            data_path: Path to climate dataset
            target_variable: Variable to predict
        """
        self.data_path = data_path
        self.target_variable = target_variable
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.models = {}
        self.scalers = {}
        self.scaler_feature = StandardScaler()
        self.scaler_target = StandardScaler()
        
        print(f"Data loaded: {self.df.shape[0]} rows")
        print(f"Target variable: {target_variable}")
    
    def create_features(self, lookback=30):
        """
        Create time-series features
        Args:
            lookback: Number of previous time steps to use
        """
        print(f"\nCreating features with lookback={lookback}...")
        
        # Select numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create lagged features
        X = []
        y = []
        
        for col in numeric_cols:
            for lag in range(1, lookback + 1):
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        
        # Create rolling statistics
        for col in numeric_cols:
            self.df[f'{col}_rolling_mean_7'] = self.df[col].rolling(window=7).mean()
            self.df[f'{col}_rolling_std_7'] = self.df[col].rolling(window=7).std()
        
        # Add temporal features
        self.df['day_of_year'] = self.df['Date'].dt.dayofyear
        self.df['month'] = self.df['Date'].dt.month
        self.df['quarter'] = self.df['Date'].dt.quarter
        
        # Drop NaN rows created by lagging
        self.df = self.df.dropna()
        
        # Prepare X and y
        feature_cols = [col for col in self.df.columns if col not in ['Date', self.target_variable]]
        X = self.df[feature_cols].values
        y = self.df[self.target_variable].values
        
        print(f"Features shape: {X.shape}")
        print(f"Features: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train multiple ML models"""
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        # Create features
        X, y, feature_cols = self.create_features(lookback=30)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Scale features
        X_train_scaled = self.scaler_feature.fit_transform(X_train)
        X_test_scaled = self.scaler_feature.transform(X_test)
        
        # Scale target
        y_train_scaled = self.scaler_target.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'SVR (RBF)': SVR(kernel='rbf', C=100, gamma='auto'),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1),
        }
        
        if XGBOOST_AVAILABLE:
            models_to_train['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                                          random_state=random_state, n_jobs=-1)
        
        results = {}
        
        for model_name, model in models_to_train.items():
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train_scaled)
            self.models[model_name] = model
            
            # Make predictions
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = self.scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, 
                                       scoring='r2', n_jobs=-1)
            
            results[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_Mean': cv_scores.mean(),
                'CV_Std': cv_scores.std()
            }
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R² (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # Summary
        print("\n" + "-"*60)
        print("MODEL COMPARISON")
        print("-"*60)
        
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('R2', ascending=False)
        print(results_df)
        
        return results_df, (X_train_scaled, X_test_scaled, y_train, y_test, y_test)
    
    def feature_importance(self):
        """Show feature importance for tree-based models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        
        X, y, feature_cols = self.create_features(lookback=30)
        X_scaled = self.scaler_feature.fit_transform(X)
        y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_scaled, y_scaled)
        
        # Get feature importance
        importance = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 20 Important Features:")
        print(feature_importance_df.head(20))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        ax.barh(range(len(top_features)), top_features['Importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Feature Importances (Random Forest)', fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig('data/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nSaved: data/visualizations/feature_importance.png")
        plt.close()
        
        return feature_importance_df
    
    def visualize_predictions(self, predictions_dict):
        """Visualize model predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Predictions: {self.target_variable}', fontsize=16, fontweight='bold')
        
        model_names = list(predictions_dict.keys())[:4]
        
        for idx, (ax, model_name) in enumerate(zip(axes.flat, model_names)):
            y_test, y_pred = predictions_dict[model_name]
            
            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue')
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            ax.set_title(f'{model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}', fontweight='bold')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/visualizations/model_predictions.png', dpi=300, bbox_inches='tight')
        print("Saved: data/visualizations/model_predictions.png")
        plt.close()


def main():
    """Train ML models"""
    ml_models = ClimateMLModels('data/climate_data.csv', target_variable='Temperature_C')
    
    # Train models
    results_df, data = ml_models.train_models()
    
    # Feature importance
    ml_models.feature_importance()
    
    print("\n" + "="*60)
    print("ML MODEL TRAINING COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()
