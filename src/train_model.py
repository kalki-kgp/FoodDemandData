"""
Training module for food demand forecasting models.
Supports ARIMA, SARIMA, Holt-Winters, and Prophet models with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import itertools

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Skipping Prophet models.")


class ForecastingModel:
    """Base class for forecasting models"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.fitted_model = None
        self.params = {}
        
    def fit(self, data: pd.Series, **kwargs):
        raise NotImplementedError
        
    def forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
        
    def save_model(self, filepath: str):
        """Save trained model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'fitted_model': self.fitted_model,
                'params': self.params
            }, f)
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls(data['model_type'])
        instance.fitted_model = data['fitted_model']
        instance.params = data['params']
        return instance


class ARIMAModel(ForecastingModel):
    """ARIMA/SARIMA model wrapper"""
    
    def __init__(self, seasonal: bool = False):
        super().__init__('SARIMA' if seasonal else 'ARIMA')
        self.seasonal = seasonal
        
    def fit(self, data: pd.Series, order: Tuple[int, int, int] = (1, 1, 1), 
            seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0), **kwargs):
        """Fit ARIMA/SARIMA model"""
        self.params = {'order': order, 'seasonal_order': seasonal_order}
        
        if self.seasonal:
            self.fitted_model = SARIMAX(data, order=order, seasonal_order=seasonal_order).fit()
        else:
            self.fitted_model = ARIMA(data, order=order).fit()
            
        return self.fitted_model
    
    def forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals"""
        forecast_result = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
        return forecast_result.values, conf_int.values


class HoltWintersModel(ForecastingModel):
    """Holt-Winters Exponential Smoothing wrapper"""
    
    def __init__(self):
        super().__init__('HoltWinters')
        
    def fit(self, data: pd.Series, trend: str = 'add', seasonal: str = 'add', 
            seasonal_periods: int = 52, **kwargs):
        """Fit Holt-Winters model"""
        self.params = {
            'trend': trend, 
            'seasonal': seasonal, 
            'seasonal_periods': seasonal_periods
        }
        
        self.fitted_model = ExponentialSmoothing(
            data, 
            trend=trend, 
            seasonal=seasonal, 
            seasonal_periods=seasonal_periods
        ).fit()
        
        return self.fitted_model
    
    def forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts"""
        forecast_result = self.fitted_model.forecast(steps=steps)
        # Holt-Winters doesn't provide confidence intervals directly
        # Using simple approximation
        forecast_std = np.std(self.fitted_model.resid) * np.sqrt(np.arange(1, steps + 1))
        conf_int_lower = forecast_result - 1.96 * forecast_std
        conf_int_upper = forecast_result + 1.96 * forecast_std
        conf_int = np.column_stack([conf_int_lower, conf_int_upper])
        
        return forecast_result.values, conf_int


class ProphetModel(ForecastingModel):
    """Prophet model wrapper"""
    
    def __init__(self):
        super().__init__('Prophet')
        
    def fit(self, data: pd.Series, yearly_seasonality: bool = True, 
            weekly_seasonality: bool = True, **kwargs):
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available")
            
        # Convert to Prophet format
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        self.params = {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality
        }
        
        self.fitted_model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality
        )
        self.fitted_model.fit(df)
        
        return self.fitted_model
    
    def forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts with confidence intervals"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        # Create future dataframe
        future = self.fitted_model.make_future_dataframe(periods=steps, freq='W')
        forecast = self.fitted_model.predict(future)
        
        # Extract last 'steps' predictions
        forecast_values = forecast['yhat'].tail(steps).values
        conf_int = forecast[['yhat_lower', 'yhat_upper']].tail(steps).values
        
        return forecast_values, conf_int


class ModelTrainer:
    """Main class for training and tuning forecasting models"""
    
    def __init__(self, data_path: str, results_path: str = '../results'):
        self.data_path = data_path
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        self.models_path = self.results_path / 'models'
        self.models_path.mkdir(exist_ok=True)
        
        # Load data
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
    def prepare_data(self, sku: str) -> pd.Series:
        """Prepare time series data for a specific SKU"""
        sku_data = self.df[self.df['sku'] == sku].copy()
        sku_data = sku_data.set_index('date').sort_index()
        return sku_data['demand']
    
    def grid_search_arima(self, data: pd.Series, seasonal: bool = False) -> Dict[str, Any]:
        """Grid search for ARIMA/SARIMA hyperparameters"""
        # Parameter grids
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        
        if seasonal:
            P_values = range(0, 2)
            D_values = range(0, 2)
            Q_values = range(0, 2)
            s_values = [52]  # Weekly seasonality
        
        best_aic = np.inf
        best_params = None
        best_model = None
        results = []
        
        # Non-seasonal ARIMA
        if not seasonal:
            for p, d, q in itertools.product(p_values, d_values, q_values):
                try:
                    model = ARIMAModel(seasonal=False)
                    fitted_model = model.fit(data, order=(p, d, q))
                    aic = fitted_model.aic
                    
                    results.append({
                        'order': (p, d, q),
                        'aic': aic,
                        'bic': fitted_model.bic
                    })
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = {'order': (p, d, q)}
                        best_model = model
                        
                except Exception as e:
                    continue
        
        # Seasonal SARIMA
        else:
            for p, d, q in itertools.product(p_values, d_values, q_values):
                for P, D, Q, s in itertools.product(P_values, D_values, Q_values, s_values):
                    try:
                        model = ARIMAModel(seasonal=True)
                        fitted_model = model.fit(data, order=(p, d, q), 
                                               seasonal_order=(P, D, Q, s))
                        aic = fitted_model.aic
                        
                        results.append({
                            'order': (p, d, q),
                            'seasonal_order': (P, D, Q, s),
                            'aic': aic,
                            'bic': fitted_model.bic
                        })
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_params = {
                                'order': (p, d, q),
                                'seasonal_order': (P, D, Q, s)
                            }
                            best_model = model
                            
                    except Exception as e:
                        continue
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_aic': best_aic,
            'all_results': results
        }
    
    def train_single_model(self, sku: str, model_type: str = 'auto') -> Dict[str, Any]:
        """Train a single model for a specific SKU"""
        data = self.prepare_data(sku)
        
        # Split data for validation
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        models_results = {}
        
        # Auto model selection
        if model_type == 'auto' or model_type == 'all':
            model_types = ['arima', 'sarima', 'holtwinters']
            if PROPHET_AVAILABLE:
                model_types.append('prophet')
        else:
            model_types = [model_type]
        
        for mtype in model_types:
            try:
                if mtype == 'arima':
                    result = self.grid_search_arima(train_data, seasonal=False)
                    model = result['best_model']
                    
                elif mtype == 'sarima':
                    result = self.grid_search_arima(train_data, seasonal=True)
                    model = result['best_model']
                    
                elif mtype == 'holtwinters':
                    model = HoltWintersModel()
                    model.fit(train_data)
                    result = {'best_model': model, 'best_aic': model.fitted_model.aic}
                    
                elif mtype == 'prophet':
                    model = ProphetModel()
                    model.fit(train_data)
                    result = {'best_model': model, 'best_aic': None}
                
                # Evaluate on test set
                forecast_values, conf_int = model.forecast(len(test_data))
                
                mape = mean_absolute_percentage_error(test_data, forecast_values) * 100
                rmse = np.sqrt(mean_squared_error(test_data, forecast_values))
                smape = self.calculate_smape(test_data.values, forecast_values) * 100
                
                models_results[mtype] = {
                    'model': model,
                    'params': result.get('best_params', model.params),
                    'aic': result.get('best_aic'),
                    'mape': mape,
                    'rmse': rmse,
                    'smape': smape,
                    'forecast': forecast_values,
                    'confidence_intervals': conf_int
                }
                
            except Exception as e:
                print(f"Error training {mtype} for {sku}: {str(e)}")
                continue
        
        # Select best model based on MAPE
        if models_results:
            best_model_type = min(models_results.keys(), 
                                key=lambda k: models_results[k]['mape'])
            best_result = models_results[best_model_type]
            best_result['model_type'] = best_model_type
            
            return {
                'sku': sku,
                'best_model': best_result,
                'all_models': models_results
            }
        else:
            raise ValueError(f"No models could be trained for {sku}")
    
    def train_all_skus(self, model_type: str = 'auto') -> Dict[str, Any]:
        """Train models for all SKUs"""
        skus = self.df['sku'].unique()
        results = {}
        
        print(f"Training models for {len(skus)} SKUs...")
        
        for i, sku in enumerate(skus, 1):
            print(f"Training {i}/{len(skus)}: {sku}")
            
            try:
                result = self.train_single_model(sku, model_type)
                results[sku] = result
                
                # Save individual model
                model_file = self.models_path / f"{sku}_model.pkl"
                result['best_model']['model'].save_model(str(model_file))
                
            except Exception as e:
                print(f"Failed to train model for {sku}: {str(e)}")
                continue
        
        # Save training results summary
        self.save_training_summary(results)
        
        return results
    
    def save_training_summary(self, results: Dict[str, Any]):
        """Save training results summary"""
        summary = {}
        
        for sku, result in results.items():
            best_model = result['best_model']
            summary[sku] = {
                'model_type': best_model['model_type'],
                'parameters': best_model['params'],
                'aic': best_model['aic'],
                'mape': best_model['mape'],
                'rmse': best_model['rmse'],
                'smape': best_model['smape']
            }
        
        # Save as JSON
        summary_file = self.results_path / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create performance summary
        perf_summary = pd.DataFrame(summary).T
        perf_summary.to_csv(self.results_path / 'model_performance.csv')
        
        print(f"Training completed. Results saved to {self.results_path}")
        print(f"Average MAPE: {perf_summary['mape'].mean():.2f}%")
        print(f"Average SMAPE: {perf_summary['smape'].mean():.2f}%")
    
    @staticmethod
    def calculate_smape(actual: np.ndarray, forecast: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        return np.mean(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train food demand forecasting models')
    parser.add_argument('--data', type=str, default='../data/food_demand_data.csv',
                       help='Path to the dataset')
    parser.add_argument('--model', type=str, default='auto',
                       choices=['auto', 'arima', 'sarima', 'holtwinters', 'prophet'],
                       help='Model type to train')
    parser.add_argument('--sku', type=str, default=None,
                       help='Train model for specific SKU only')
    parser.add_argument('--output', type=str, default='../results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.data, args.output)
    
    if args.sku:
        # Train single SKU
        result = trainer.train_single_model(args.sku, args.model)
        print(f"Training completed for {args.sku}")
        print(f"Best model: {result['best_model']['model_type']}")
        print(f"MAPE: {result['best_model']['mape']:.2f}%")
    else:
        # Train all SKUs
        results = trainer.train_all_skus(args.model)
        print(f"Training completed for {len(results)} SKUs")


if __name__ == "__main__":
    main()