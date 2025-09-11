"""
Model evaluation and performance reporting module.
Provides comprehensive analysis of forecasting model performance.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """Comprehensive model evaluation and reporting"""
    
    def __init__(self, results_path: str = '../results'):
        self.results_path = Path(results_path)
        self.reports_path = self.results_path / 'reports'
        self.reports_path.mkdir(exist_ok=True)
        
    def load_training_results(self) -> Dict[str, Any]:
        """Load training results and performance metrics"""
        training_file = self.results_path / 'training_summary.json'
        
        if not training_file.exists():
            raise FileNotFoundError(f"Training results not found: {training_file}")
        
        with open(training_file, 'r') as f:
            return json.load(f)
    
    def load_forecast_results(self, forecast_file: str) -> Dict[str, Any]:
        """Load forecast results"""
        forecast_path = Path(forecast_file)
        
        if not forecast_path.exists():
            raise FileNotFoundError(f"Forecast results not found: {forecast_path}")
        
        with open(forecast_path, 'r') as f:
            return json.load(f)
    
    def calculate_accuracy_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics"""
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {metric: np.nan for metric in ['mae', 'mse', 'rmse', 'mape', 'smape', 'r2']}
        
        mae = mean_absolute_error(actual_clean, predicted_clean)
        mse = mean_squared_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mse)
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((actual_clean - predicted_clean) / np.where(actual_clean != 0, actual_clean, 1))) * 100
        
        # SMAPE
        smape = np.mean(2 * np.abs(predicted_clean - actual_clean) / 
                       (np.abs(actual_clean) + np.abs(predicted_clean))) * 100
        
        # R-squared
        r2 = r2_score(actual_clean, predicted_clean)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'r2': r2
        }
    
    def evaluate_training_performance(self) -> Dict[str, Any]:
        """Evaluate and summarize training performance across all SKUs"""
        training_results = self.load_training_results()
        
        # Extract performance metrics
        performance_data = []
        model_types = []
        
        for sku, results in training_results.items():
            perf_data = {
                'sku': sku,
                'model_type': results.get('model_type', 'unknown'),
                'mape': results.get('mape', np.nan),
                'smape': results.get('smape', np.nan),
                'rmse': results.get('rmse', np.nan),
                'aic': results.get('aic', np.nan)
            }
            performance_data.append(perf_data)
            model_types.append(results.get('model_type', 'unknown'))
        
        df_performance = pd.DataFrame(performance_data)
        
        # Calculate summary statistics
        summary_stats = {
            'total_skus': len(df_performance),
            'model_distribution': pd.Series(model_types).value_counts().to_dict(),
            'average_mape': df_performance['mape'].mean(),
            'median_mape': df_performance['mape'].median(),
            'average_smape': df_performance['smape'].mean(),
            'median_smape': df_performance['smape'].median(),
            'average_rmse': df_performance['rmse'].mean(),
            'models_meeting_target': {
                'mape_under_5': (df_performance['mape'] < 5).sum(),
                'smape_under_6': (df_performance['smape'] < 6).sum(),
                'percentage_mape_under_5': (df_performance['mape'] < 5).mean() * 100,
                'percentage_smape_under_6': (df_performance['smape'] < 6).mean() * 100
            }
        }
        
        return {
            'summary_statistics': summary_stats,
            'detailed_performance': df_performance,
            'evaluation_date': datetime.now().isoformat()
        }
    
    def analyze_residuals(self, actual: np.ndarray, predicted: np.ndarray, 
                         sku: str) -> Dict[str, Any]:
        """Analyze model residuals for diagnostic purposes"""
        residuals = actual - predicted
        
        # Basic residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'skewness': float(pd.Series(residuals).skew()),
            'kurtosis': float(pd.Series(residuals).kurtosis())
        }
        
        # Normality test (Shapiro-Wilk)
        from scipy.stats import shapiro, jarque_bera
        if len(residuals) > 3:
            shapiro_stat, shapiro_p = shapiro(residuals[:min(len(residuals), 5000)])  # Limit for computational efficiency
            jb_stat, jb_p = jarque_bera(residuals)
        else:
            shapiro_stat = shapiro_p = jb_stat = jb_p = np.nan
        
        return {
            'sku': sku,
            'residual_statistics': residual_stats,
            'normality_tests': {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p}
            },
            'residuals': residuals.tolist()
        }
    
    def compare_models_by_sku(self, training_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Compare different model types across SKUs"""
        if training_results is None:
            training_results = self.load_training_results()
        
        # Group by model type
        model_performance = {}
        
        for sku, results in training_results.items():
            model_type = results.get('model_type', 'unknown')
            
            if model_type not in model_performance:
                model_performance[model_type] = []
            
            model_performance[model_type].append({
                'sku': sku,
                'mape': results.get('mape', np.nan),
                'smape': results.get('smape', np.nan),
                'rmse': results.get('rmse', np.nan)
            })
        
        # Calculate statistics by model type
        model_comparison = {}
        
        for model_type, performances in model_performance.items():
            df_model = pd.DataFrame(performances)
            
            model_comparison[model_type] = {
                'count': len(df_model),
                'average_mape': df_model['mape'].mean(),
                'median_mape': df_model['mape'].median(),
                'std_mape': df_model['mape'].std(),
                'average_smape': df_model['smape'].mean(),
                'median_smape': df_model['smape'].median(),
                'std_smape': df_model['smape'].std(),
                'average_rmse': df_model['rmse'].mean(),
                'best_sku_mape': df_model.loc[df_model['mape'].idxmin(), 'sku'] if not df_model['mape'].isna().all() else None,
                'worst_sku_mape': df_model.loc[df_model['mape'].idxmax(), 'sku'] if not df_model['mape'].isna().all() else None
            }
        
        return model_comparison
    
    def generate_performance_visualizations(self, output_path: Optional[str] = None):
        """Generate comprehensive performance visualization plots"""
        if output_path is None:
            output_path = self.reports_path
        else:
            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True)
        
        training_results = self.load_training_results()
        eval_results = self.evaluate_training_performance()
        
        # 1. Model performance distribution
        df_performance = eval_results['detailed_performance']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Distribution', fontsize=16)
        
        # MAPE distribution
        axes[0, 0].hist(df_performance['mape'].dropna(), bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(5, color='red', linestyle='--', label='Target: 5%')
        axes[0, 0].set_xlabel('MAPE (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('MAPE Distribution')
        axes[0, 0].legend()
        
        # SMAPE distribution
        axes[0, 1].hist(df_performance['smape'].dropna(), bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].axvline(6, color='red', linestyle='--', label='Target: 6%')
        axes[0, 1].set_xlabel('SMAPE (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('SMAPE Distribution')
        axes[0, 1].legend()
        
        # Model type distribution
        model_counts = df_performance['model_type'].value_counts()
        axes[1, 0].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Model Type Distribution')
        
        # Performance by model type
        df_performance.boxplot(column='mape', by='model_type', ax=axes[1, 1])
        axes[1, 1].set_title('MAPE by Model Type')
        axes[1, 1].set_xlabel('Model Type')
        axes[1, 1].set_ylabel('MAPE (%)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Interactive performance comparison
        fig = go.Figure()
        
        for model_type in df_performance['model_type'].unique():
            model_data = df_performance[df_performance['model_type'] == model_type]
            
            fig.add_trace(go.Scatter(
                x=model_data['mape'],
                y=model_data['smape'],
                mode='markers',
                name=model_type,
                text=model_data['sku'],
                hovertemplate='<b>%{text}</b><br>MAPE: %{x:.2f}%<br>SMAPE: %{y:.2f}%<extra></extra>'
            ))
        
        fig.add_shape(type='line', x0=0, x1=5, y0=5, y1=5, 
                     line=dict(color='red', dash='dash'), name='MAPE Target')
        fig.add_shape(type='line', x0=5, x1=5, y0=0, y1=6, 
                     line=dict(color='red', dash='dash'), name='SMAPE Target')
        
        fig.update_layout(
            title='Model Performance: MAPE vs SMAPE',
            xaxis_title='MAPE (%)',
            yaxis_title='SMAPE (%)',
            width=800,
            height=600
        )
        
        fig.write_html(output_path / 'performance_comparison.html')
        
        print(f"Performance visualizations saved to {output_path}")
    
    def generate_comprehensive_report(self, forecast_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report"""
        print("Generating comprehensive evaluation report...")
        
        # Load training results
        training_eval = self.evaluate_training_performance()
        model_comparison = self.compare_models_by_sku()
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive_model_evaluation'
            },
            'training_performance': training_eval,
            'model_comparison': model_comparison,
            'success_criteria_assessment': self._assess_success_criteria(training_eval),
            'recommendations': self._generate_recommendations(training_eval, model_comparison)
        }
        
        # Add forecast evaluation if available
        if forecast_file:
            try:
                forecast_results = self.load_forecast_results(forecast_file)
                report['forecast_analysis'] = self._analyze_forecasts(forecast_results)
            except Exception as e:
                print(f"Could not load forecast results: {e}")
        
        # Save report
        report_file = self.reports_path / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self.generate_performance_visualizations()
        
        print(f"Comprehensive report saved to {report_file}")
        
        return report
    
    def _assess_success_criteria(self, training_eval: Dict) -> Dict[str, Any]:
        """Assess performance against success criteria from PRD"""
        summary = training_eval['summary_statistics']
        
        assessment = {
            'mape_target_5_percent': {
                'target': 5.0,
                'achieved_average': summary['average_mape'],
                'models_meeting_target': summary['models_meeting_target']['mape_under_5'],
                'percentage_meeting': summary['models_meeting_target']['percentage_mape_under_5'],
                'status': 'PASS' if summary['average_mape'] <= 5.0 else 'FAIL'
            },
            'smape_target_6_percent': {
                'target': 6.0,
                'achieved_average': summary['average_smape'],
                'models_meeting_target': summary['models_meeting_target']['smape_under_6'],
                'percentage_meeting': summary['models_meeting_target']['percentage_smape_under_6'],
                'status': 'PASS' if summary['average_smape'] <= 6.0 else 'FAIL'
            },
            'overall_success': summary['average_mape'] <= 5.0 and summary['average_smape'] <= 6.0
        }
        
        return assessment
    
    def _generate_recommendations(self, training_eval: Dict, model_comparison: Dict) -> List[str]:
        """Generate improvement recommendations based on results"""
        recommendations = []
        summary = training_eval['summary_statistics']
        
        if summary['average_mape'] > 5.0:
            recommendations.append("Consider feature engineering or external regressors to improve MAPE")
        
        if summary['average_smape'] > 6.0:
            recommendations.append("Investigate SKUs with high SMAPE for model-specific tuning")
        
        # Model-specific recommendations
        best_model = min(model_comparison.items(), key=lambda x: x[1]['average_mape'])
        if best_model[1]['count'] < len(training_eval['detailed_performance']) * 0.5:
            recommendations.append(f"Consider using {best_model[0]} for more SKUs as it shows best performance")
        
        if summary['models_meeting_target']['percentage_mape_under_5'] < 70:
            recommendations.append("Significant model improvement needed - consider ensemble methods or deep learning approaches")
        
        return recommendations
    
    def _analyze_forecasts(self, forecast_results: Dict) -> Dict[str, Any]:
        """Analyze forecast characteristics"""
        forecasts = forecast_results.get('forecasts', {})
        
        forecast_stats = {}
        total_demand = 0
        
        for sku, forecast_data in forecasts.items():
            values = np.array(forecast_data['forecast_values'])
            total_demand += values.sum()
            
            forecast_stats[sku] = {
                'total_forecasted': float(values.sum()),
                'avg_weekly': float(values.mean()),
                'volatility': float(values.std() / values.mean() if values.mean() > 0 else 0),
                'model_used': forecast_data.get('model_type', 'unknown')
            }
        
        return {
            'total_forecasted_demand': total_demand,
            'forecast_horizon_weeks': len(next(iter(forecasts.values()))['forecast_values']),
            'sku_forecast_statistics': forecast_stats,
            'analysis_date': datetime.now().isoformat()
        }


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate food demand forecasting models')
    parser.add_argument('--results', type=str, default='../results',
                       help='Path to results directory')
    parser.add_argument('--forecast-file', type=str, default=None,
                       help='Path to forecast results file for evaluation')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report without visualizations')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.results)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report(args.forecast_file)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    training_perf = report['training_performance']['summary_statistics']
    print(f"Total SKUs evaluated: {training_perf['total_skus']}")
    print(f"Average MAPE: {training_perf['average_mape']:.2f}%")
    print(f"Average SMAPE: {training_perf['average_smape']:.2f}%")
    
    success_criteria = report['success_criteria_assessment']
    print(f"\nSuccess Criteria Assessment:")
    print(f"MAPE ≤ 5%: {success_criteria['mape_target_5_percent']['status']}")
    print(f"SMAPE ≤ 6%: {success_criteria['smape_target_6_percent']['status']}")
    print(f"Overall Success: {'PASS' if success_criteria['overall_success'] else 'FAIL'}")
    
    if report.get('recommendations'):
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")


if __name__ == "__main__":
    main()