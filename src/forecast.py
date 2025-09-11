"""
Forecasting module for food demand prediction.
Supports batch forecasting for multiple SKUs and time horizons.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from train_model import ForecastingModel, ARIMAModel, HoltWintersModel, ProphetModel


class ForecastEngine:
    """Main forecasting engine for batch predictions"""
    
    def __init__(self, models_path: str, data_path: str, results_path: str = '../results'):
        self.models_path = Path(models_path)
        self.data_path = data_path
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Load historical data
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Load trained models
        self.models = self._load_models()
        
    def _load_models(self) -> Dict[str, ForecastingModel]:
        """Load all trained models"""
        models = {}
        
        if not self.models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_path}")
        
        model_files = list(self.models_path.glob("*_model.pkl"))
        
        for model_file in model_files:
            sku = model_file.stem.replace('_model', '')
            try:
                # Load the model based on its type
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                model_type = model_data['model_type']
                
                if model_type in ['ARIMA', 'SARIMA']:
                    model = ARIMAModel(seasonal=(model_type == 'SARIMA'))
                elif model_type == 'HoltWinters':
                    model = HoltWintersModel()
                elif model_type == 'Prophet':
                    model = ProphetModel()
                else:
                    print(f"Unknown model type {model_type} for {sku}")
                    continue
                
                model.fitted_model = model_data['fitted_model']
                model.params = model_data['params']
                models[sku] = model
                
            except Exception as e:
                print(f"Error loading model for {sku}: {str(e)}")
                continue
        
        print(f"Loaded {len(models)} trained models")
        return models
    
    def get_last_date(self) -> datetime:
        """Get the last date in the historical data"""
        return self.df['date'].max()
    
    def forecast_single_sku(self, sku: str, steps: int, 
                           include_history: bool = False) -> Dict[str, Any]:
        """Generate forecast for a single SKU"""
        if sku not in self.models:
            raise ValueError(f"No trained model found for SKU: {sku}")
        
        model = self.models[sku]
        
        try:
            # Generate forecast
            forecast_values, conf_int = model.forecast(steps)
            
            # Create forecast dates
            last_date = self.get_last_date()
            forecast_dates = [last_date + timedelta(weeks=i+1) for i in range(steps)]
            
            # Prepare results
            result = {
                'sku': sku,
                'model_type': model.model_type,
                'forecast_dates': forecast_dates,
                'forecast_values': forecast_values.tolist(),
                'confidence_intervals': {
                    'lower': conf_int[:, 0].tolist(),
                    'upper': conf_int[:, 1].tolist()
                },
                'forecast_horizon': steps,
                'generated_at': datetime.now().isoformat()
            }
            
            # Include historical data if requested
            if include_history:
                historical = self.df[self.df['sku'] == sku].copy()
                result['historical_data'] = {
                    'dates': historical['date'].dt.strftime('%Y-%m-%d').tolist(),
                    'values': historical['demand'].tolist()
                }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error generating forecast for {sku}: {str(e)}")
    
    def forecast_batch(self, skus: Optional[List[str]] = None, steps: int = 12,
                      include_history: bool = False, save_results: bool = True) -> Dict[str, Any]:
        """Generate forecasts for multiple SKUs"""
        if skus is None:
            skus = list(self.models.keys())
        
        results = {}
        failed_forecasts = []
        
        print(f"Generating forecasts for {len(skus)} SKUs, {steps} weeks ahead...")
        
        for i, sku in enumerate(skus, 1):
            print(f"Forecasting {i}/{len(skus)}: {sku}")
            
            try:
                forecast_result = self.forecast_single_sku(sku, steps, include_history)
                results[sku] = forecast_result
                
            except Exception as e:
                print(f"Failed to forecast {sku}: {str(e)}")
                failed_forecasts.append(sku)
                continue
        
        # Create summary
        summary = {
            'forecast_summary': {
                'total_skus_requested': len(skus),
                'successful_forecasts': len(results),
                'failed_forecasts': len(failed_forecasts),
                'failed_sku_list': failed_forecasts,
                'forecast_horizon_weeks': steps,
                'generated_at': datetime.now().isoformat()
            },
            'forecasts': results
        }
        
        if save_results:
            self._save_forecast_results(summary)
        
        print(f"Batch forecasting completed: {len(results)}/{len(skus)} successful")
        
        return summary
    
    def forecast_aggregated(self, skus: Optional[List[str]] = None, steps: int = 12,
                           aggregation_level: str = 'total') -> Dict[str, Any]:
        """Generate aggregated forecasts across SKUs"""
        # Get individual forecasts
        batch_results = self.forecast_batch(skus, steps, save_results=False)
        individual_forecasts = batch_results['forecasts']
        
        if not individual_forecasts:
            raise ValueError("No successful individual forecasts to aggregate")
        
        # Aggregate forecasts
        forecast_dates = next(iter(individual_forecasts.values()))['forecast_dates']
        
        aggregated_forecasts = np.zeros(steps)
        aggregated_lower = np.zeros(steps)
        aggregated_upper = np.zeros(steps)
        
        contributing_skus = []
        
        for sku, forecast_data in individual_forecasts.items():
            forecast_values = np.array(forecast_data['forecast_values'])
            lower_bounds = np.array(forecast_data['confidence_intervals']['lower'])
            upper_bounds = np.array(forecast_data['confidence_intervals']['upper'])
            
            aggregated_forecasts += forecast_values
            # For confidence intervals, we'll use simple addition (could be improved with correlation)
            aggregated_lower += lower_bounds
            aggregated_upper += upper_bounds
            
            contributing_skus.append(sku)
        
        result = {
            'aggregation_level': aggregation_level,
            'contributing_skus': contributing_skus,
            'forecast_dates': forecast_dates,
            'aggregated_forecast': aggregated_forecasts.tolist(),
            'confidence_intervals': {
                'lower': aggregated_lower.tolist(),
                'upper': aggregated_upper.tolist()
            },
            'forecast_horizon': steps,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save aggregated results
        agg_file = self.results_path / f'aggregated_forecast_{aggregation_level}.json'
        with open(agg_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return result
    
    def _save_forecast_results(self, results: Dict[str, Any]):
        """Save forecast results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save complete results as JSON
        json_file = self.results_path / f'forecasts_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create CSV summary for easy viewing
        forecast_data = []
        for sku, forecast in results['forecasts'].items():
            for i, (date, value, lower, upper) in enumerate(zip(
                forecast['forecast_dates'],
                forecast['forecast_values'],
                forecast['confidence_intervals']['lower'],
                forecast['confidence_intervals']['upper']
            )):
                forecast_data.append({
                    'sku': sku,
                    'forecast_date': date,
                    'forecast_week': i + 1,
                    'forecasted_demand': value,
                    'confidence_lower': lower,
                    'confidence_upper': upper,
                    'model_type': forecast['model_type']
                })
        
        csv_file = self.results_path / f'forecasts_{timestamp}.csv'
        pd.DataFrame(forecast_data).to_csv(csv_file, index=False)
        
        print(f"Forecast results saved:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
    
    def create_forecast_comparison(self, forecast_file: str, 
                                 actual_data_file: Optional[str] = None) -> Dict[str, Any]:
        """Compare forecasts with actual values if available"""
        # Load forecast results
        with open(forecast_file, 'r') as f:
            forecast_data = json.load(f)
        
        if actual_data_file is None:
            print("No actual data provided - returning forecast summary only")
            return self._create_forecast_summary(forecast_data)
        
        # Load actual data
        actual_df = pd.read_csv(actual_data_file)
        actual_df['date'] = pd.to_datetime(actual_df['date'])
        
        comparison_results = {}
        
        for sku, forecast_info in forecast_data['forecasts'].items():
            # Get actual values for forecast period
            forecast_dates = pd.to_datetime(forecast_info['forecast_dates'])
            sku_actual = actual_df[
                (actual_df['sku'] == sku) & 
                (actual_df['date'].isin(forecast_dates))
            ]
            
            if len(sku_actual) > 0:
                # Calculate accuracy metrics
                actual_values = sku_actual.set_index('date')['demand'].reindex(forecast_dates).values
                forecast_values = np.array(forecast_info['forecast_values'])
                
                # Remove NaN values for comparison
                mask = ~np.isnan(actual_values)
                if mask.sum() > 0:
                    actual_clean = actual_values[mask]
                    forecast_clean = forecast_values[mask]
                    
                    mape = np.mean(np.abs((actual_clean - forecast_clean) / actual_clean)) * 100
                    rmse = np.sqrt(np.mean((actual_clean - forecast_clean) ** 2))
                    
                    comparison_results[sku] = {
                        'mape': mape,
                        'rmse': rmse,
                        'actual_values': actual_clean.tolist(),
                        'forecast_values': forecast_clean.tolist(),
                        'forecast_dates_compared': forecast_dates[mask].strftime('%Y-%m-%d').tolist()
                    }
        
        return {
            'comparison_summary': {
                'total_skus_compared': len(comparison_results),
                'average_mape': np.mean([r['mape'] for r in comparison_results.values()]),
                'average_rmse': np.mean([r['rmse'] for r in comparison_results.values()]),
                'generated_at': datetime.now().isoformat()
            },
            'sku_comparisons': comparison_results
        }
    
    def _create_forecast_summary(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for forecasts"""
        forecasts = forecast_data['forecasts']
        
        summary_stats = {}
        
        for sku, forecast_info in forecasts.items():
            values = np.array(forecast_info['forecast_values'])
            
            summary_stats[sku] = {
                'total_forecasted_demand': float(values.sum()),
                'average_weekly_demand': float(values.mean()),
                'min_weekly_demand': float(values.min()),
                'max_weekly_demand': float(values.max()),
                'coefficient_of_variation': float(values.std() / values.mean() if values.mean() > 0 else 0),
                'model_type': forecast_info['model_type']
            }
        
        return {
            'forecast_summary': forecast_data['forecast_summary'],
            'sku_statistics': summary_stats,
            'overall_statistics': {
                'total_demand_all_skus': sum(s['total_forecasted_demand'] for s in summary_stats.values()),
                'average_demand_all_skus': np.mean([s['average_weekly_demand'] for s in summary_stats.values()]),
                'generated_at': datetime.now().isoformat()
            }
        }


def main():
    """Main forecasting function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate food demand forecasts')
    parser.add_argument('--models', type=str, default='../results/models',
                       help='Path to trained models directory')
    parser.add_argument('--data', type=str, default='../data/food_demand_data.csv',
                       help='Path to historical data')
    parser.add_argument('--output', type=str, default='../results',
                       help='Output directory for forecasts')
    parser.add_argument('--steps', type=int, default=12,
                       help='Number of weeks to forecast')
    parser.add_argument('--sku', type=str, default=None,
                       help='Forecast for specific SKU only')
    parser.add_argument('--aggregate', action='store_true',
                       help='Generate aggregated forecast')
    parser.add_argument('--include-history', action='store_true',
                       help='Include historical data in results')
    
    args = parser.parse_args()
    
    # Initialize forecast engine
    engine = ForecastEngine(args.models, args.data, args.output)
    
    if args.sku:
        # Single SKU forecast
        result = engine.forecast_single_sku(args.sku, args.steps, args.include_history)
        print(f"Forecast generated for {args.sku}")
        print(f"Next {args.steps} weeks total demand: {sum(result['forecast_values']):.0f}")
    else:
        # Batch forecast
        results = engine.forecast_batch(steps=args.steps, include_history=args.include_history)
        
        if args.aggregate:
            # Generate aggregated forecast
            agg_result = engine.forecast_aggregated(steps=args.steps)
            print(f"Aggregated forecast generated")
            print(f"Total demand next {args.steps} weeks: {sum(agg_result['aggregated_forecast']):.0f}")


if __name__ == "__main__":
    main()