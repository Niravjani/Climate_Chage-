"""
Exploratory Data Analysis (EDA) for Climate Data
Performs comprehensive analysis of climate indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class ClimateEDAAnalyzer:
    """Perform exploratory data analysis on climate data"""
    
    def __init__(self, data_path='data/climate_data.csv'):
        """Load and initialize data"""
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
    
    def basic_statistics(self):
        """Compute basic statistics"""
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        
        stats_df = self.df.describe()
        print(stats_df)
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        return stats_df
    
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(correlation_matrix)
        
        # Find strong correlations
        print("\nStrong Correlations (|r| > 0.5):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.5:
                    print(f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}: "
                          f"{correlation_matrix.iloc[i, j]:.3f}")
        
        return correlation_matrix
    
    def temporal_analysis(self):
        """Analyze temporal trends"""
        print("\n" + "="*60)
        print("TEMPORAL TREND ANALYSIS")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate trend using linear regression
            x = np.arange(len(self.df))
            y = self.df[col].values
            
            # Remove NaNs for regression
            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) > 0:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
                
                print(f"\n{col}:")
                print(f"  Trend (slope): {slope:.6f} per day")
                print(f"  R-squared: {r_value**2:.4f}")
                print(f"  P-value: {p_value:.2e}")
                print(f"  Mean: {y_clean.mean():.2f}, Std: {y_clean.std():.2f}")
    
    def seasonal_analysis(self):
        """Analyze seasonal patterns"""
        print("\n" + "="*60)
        print("SEASONAL PATTERN ANALYSIS")
        print("="*60)
        
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Quarter'] = self.df['Date'].dt.quarter
        self.df['Year'] = self.df['Date'].dt.year
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['Month', 'Quarter', 'Year']]
        
        print("\nAverage values by Month:")
        monthly_avg = self.df.groupby('Month')[numeric_cols].mean()
        print(monthly_avg)
        
        print("\nAverage values by Quarter:")
        quarterly_avg = self.df.groupby('Quarter')[numeric_cols].mean()
        print(quarterly_avg)
    
    def outlier_detection(self, threshold=3):
        """Detect outliers using Z-score"""
        print("\n" + "="*60)
        print("OUTLIER DETECTION (Z-score > {})".format(threshold))
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers = np.where(z_scores > threshold)[0]
            
            if len(outliers) > 0:
                print(f"\n{col}: {len(outliers)} outliers detected")
                print(f"  Outlier indices: {outliers[:10]}")  # Show first 10
    
    def distribution_analysis(self):
        """Analyze distributions of variables"""
        print("\n" + "="*60)
        print("DISTRIBUTION ANALYSIS")
        print("="*60)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            # Normality test
            stat, p_value = stats.shapiro(data.sample(min(5000, len(data))))
            
            # Skewness and kurtosis
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            print(f"\n{col}:")
            print(f"  Shapiro-Wilk p-value: {p_value:.4f} (normal if > 0.05)")
            print(f"  Skewness: {skewness:.4f}")
            print(f"  Kurtosis: {kurtosis:.4f}")
    
    def create_visualizations(self, output_dir='data/visualizations'):
        """Create comprehensive visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations in {output_dir}...")
        
        # Time series plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Climate Indicators Over Time', fontsize=16, fontweight='bold')
        
        # Select only climate indicators (exclude Date if present)
        numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns if col != 'Date']
        numeric_cols = numeric_cols[:6]  # Limit to 6 columns for the subplot grid
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx // 2, idx % 2]
            ax.plot(self.df['Date'], self.df[col], linewidth=1, color='steelblue')
            ax.set_title(col, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Remove extra subplots if fewer than 6 columns
        if len(numeric_cols) < 6:
            for i in range(len(numeric_cols), 6):
                fig.delaxes(axes.flatten()[i])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/timeseries.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/timeseries.png")
        plt.close()
        
        # Correlation heatmap
        numeric_cols_all = [col for col in self.df.select_dtypes(include=[np.number]).columns if col != 'Date']
        correlation_matrix = self.df[numeric_cols_all].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        plt.title('Climate Indicators Correlation Matrix', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/correlation_heatmap.png")
        plt.close()
        
        # Distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Distribution of Climate Indicators', fontsize=16, fontweight='bold')
        
        plot_cols = numeric_cols_all[:6]
        for idx, col in enumerate(plot_cols):
            ax = axes[idx // 3, idx % 3]
            ax.hist(self.df[col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(col, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove extra subplots
        for i in range(len(plot_cols), 6):
            fig.delaxes(axes.flatten()[i])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distributions.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/distributions.png")
        plt.close()
        
        # Monthly seasonality
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Seasonal Patterns (Monthly Averages)', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(plot_cols):
            ax = axes[idx // 3, idx % 3]
            monthly_data = self.df.groupby('Month')[col].mean()
            ax.plot(monthly_data.index, monthly_data.values, marker='o', linewidth=2, markersize=8)
            ax.set_title(col, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Value')
            ax.set_xticks(range(1, 13))
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplots
        for i in range(len(plot_cols), 6):
            fig.delaxes(axes.flatten()[i])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/seasonal_patterns.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/seasonal_patterns.png")
        plt.close()


def main():
    """Run EDA analysis"""
    analyzer = ClimateEDAAnalyzer('data/climate_data.csv')
    
    # Run all analyses
    analyzer.basic_statistics()
    analyzer.correlation_analysis()
    analyzer.temporal_analysis()
    analyzer.seasonal_analysis()
    analyzer.outlier_detection()
    analyzer.distribution_analysis()
    analyzer.create_visualizations()
    
    print("\n" + "="*60)
    print("EDA Analysis Completed!")
    print("="*60)


if __name__ == '__main__':
    main()
