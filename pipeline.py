"""
Automated Pipeline for Food Demand Forecasting System
Integrates data preprocessing, model training, forecasting, and evaluation.
"""

import sys
import os
from pathlib import Path
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from train_model import ModelTrainer
    from forecast import ForecastEngine
    from evaluate import ModelEvaluator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


class ForecastingPipeline:
    """Main pipeline orchestrator for the forecasting system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        
        # Paths
        self.data_path = Path(config.get('data_path', 'data/food_demand_data.csv'))
        self.results_path = Path(config.get('results_path', 'results'))
        self.models_path = self.results_path / 'models'
        
        # Ensure directories exist
        self.results_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        
        # Pipeline state
        self.pipeline_state = {
            'data_generated': False,
            'models_trained': False,
            'forecasts_generated': False,
            'evaluation_completed': False
        }
        
        self.logger.info("Pipeline initialized")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_path / 'pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def generate_dataset(self) -> bool:
        """Generate the food demand dataset"""
        self.logger.info("Starting dataset generation...")
        
        try:
            if self.data_path.exists():
                self.logger.info(f"Dataset already exists at {self.data_path}")
                self.pipeline_state['data_generated'] = True
                return True
            
            # Import and run dataset generation
            from data.generate_dataset import generate_food_demand_data
            
            df = generate_food_demand_data()
            df.to_csv(self.data_path, index=False)
            
            self.logger.info(f"Dataset generated successfully: {len(df)} rows")
            self.logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            self.logger.info(f"SKUs: {list(df['sku'].unique())}")
            
            self.pipeline_state['data_generated'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset generation failed: {str(e)}")
            return False
    
    def train_models(self) -> bool:
        """Train forecasting models for all SKUs"""
        self.logger.info("Starting model training...")
        
        if not self.pipeline_state['data_generated']:
            self.logger.error("Dataset not available. Generate dataset first.")
            return False
        
        try:
            # Initialize trainer
            trainer = ModelTrainer(
                data_path=str(self.data_path),
                results_path=str(self.results_path)
            )
            
            # Get training configuration
            model_type = self.config.get('model_type', 'auto')
            target_skus = self.config.get('target_skus', None)
            
            if target_skus:
                self.logger.info(f"Training models for specific SKUs: {target_skus}")
                results = {}
                for sku in target_skus:
                    result = trainer.train_single_model(sku, model_type)
                    results[sku] = result
            else:
                self.logger.info("Training models for all SKUs")
                results = trainer.train_all_skus(model_type)
            
            self.pipeline_state['models_trained'] = True
            self.logger.info(f"Model training completed: {len(results)} SKUs")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            return False
    
    def generate_forecasts(self) -> bool:
        """Generate forecasts using trained models"""
        self.logger.info("Starting forecast generation...")
        
        if not self.pipeline_state['models_trained']:
            self.logger.error("Models not trained. Train models first.")
            return False
        
        try:
            # Initialize forecast engine
            engine = ForecastEngine(
                models_path=str(self.models_path),
                data_path=str(self.data_path),
                results_path=str(self.results_path)
            )
            
            # Get forecast configuration
            forecast_horizon = self.config.get('forecast_horizon', 12)
            target_skus = self.config.get('target_skus', None)
            include_history = self.config.get('include_history', True)
            generate_aggregated = self.config.get('generate_aggregated', True)
            
            # Generate batch forecasts
            results = engine.forecast_batch(
                skus=target_skus,
                steps=forecast_horizon,
                include_history=include_history,
                save_results=True
            )
            
            # Generate aggregated forecast if requested
            if generate_aggregated:
                agg_result = engine.forecast_aggregated(
                    skus=target_skus,
                    steps=forecast_horizon
                )
                self.logger.info("Aggregated forecast generated")
            
            self.pipeline_state['forecasts_generated'] = True
            self.logger.info(f"Forecast generation completed: {len(results['forecasts'])} SKUs")
            
            # Store latest forecast file path for evaluation
            latest_forecast_files = list(self.results_path.glob('forecasts_*.json'))
            if latest_forecast_files:
                self.latest_forecast_file = max(latest_forecast_files, key=os.path.getctime)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Forecast generation failed: {str(e)}")
            return False
    
    def evaluate_models(self) -> bool:
        """Evaluate model performance and generate reports"""
        self.logger.info("Starting model evaluation...")
        
        if not self.pipeline_state['models_trained']:
            self.logger.error("Models not trained. Train models first.")
            return False
        
        try:
            # Initialize evaluator
            evaluator = ModelEvaluator(results_path=str(self.results_path))
            
            # Generate comprehensive report
            forecast_file = getattr(self, 'latest_forecast_file', None)
            report = evaluator.generate_comprehensive_report(
                forecast_file=str(forecast_file) if forecast_file else None
            )
            
            # Log key metrics
            training_perf = report['training_performance']['summary_statistics']
            self.logger.info(f"Average MAPE: {training_perf['average_mape']:.2f}%")
            self.logger.info(f"Average SMAPE: {training_perf['average_smape']:.2f}%")
            
            success_criteria = report['success_criteria_assessment']
            self.logger.info(f"MAPE Target (≤5%): {success_criteria['mape_target_5_percent']['status']}")
            self.logger.info(f"SMAPE Target (≤6%): {success_criteria['smape_target_6_percent']['status']}")
            
            self.pipeline_state['evaluation_completed'] = True
            self.logger.info("Model evaluation completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete forecasting pipeline"""
        self.logger.info("="*60)
        self.logger.info("STARTING FULL FORECASTING PIPELINE")
        self.logger.info("="*60)
        
        start_time = datetime.now()
        
        # Step 1: Generate dataset
        if not self.generate_dataset():
            self.logger.error("Pipeline failed at dataset generation")
            return False
        
        # Step 2: Train models
        if not self.train_models():
            self.logger.error("Pipeline failed at model training")
            return False
        
        # Step 3: Generate forecasts
        if not self.generate_forecasts():
            self.logger.error("Pipeline failed at forecast generation")
            return False
        
        # Step 4: Evaluate models
        if not self.evaluate_models():
            self.logger.error("Pipeline failed at model evaluation")
            return False
        
        # Pipeline completed successfully
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("="*60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info(f"Total execution time: {duration}")
        self.logger.info("="*60)
        
        # Save pipeline summary
        self.save_pipeline_summary(start_time, end_time, success=True)
        
        return True
    
    def run_partial_pipeline(self, steps: List[str]) -> bool:
        """Run specific pipeline steps"""
        self.logger.info(f"Running partial pipeline: {steps}")
        
        step_methods = {
            'generate_data': self.generate_dataset,
            'train': self.train_models,
            'forecast': self.generate_forecasts,
            'evaluate': self.evaluate_models
        }
        
        success = True
        for step in steps:
            if step not in step_methods:
                self.logger.error(f"Unknown pipeline step: {step}")
                success = False
                break
            
            if not step_methods[step]():
                self.logger.error(f"Pipeline failed at step: {step}")
                success = False
                break
        
        return success
    
    def save_pipeline_summary(self, start_time: datetime, end_time: datetime, success: bool):
        """Save pipeline execution summary"""
        summary = {
            'pipeline_execution': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'success': success,
                'configuration': self.config,
                'pipeline_state': self.pipeline_state
            }
        }
        
        summary_file = self.results_path / f'pipeline_summary_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline summary saved to {summary_file}")


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load pipeline configuration"""
    default_config = {
        'data_path': 'data/food_demand_data.csv',
        'results_path': 'results',
        'model_type': 'auto',
        'forecast_horizon': 12,
        'target_skus': None,
        'include_history': True,
        'generate_aggregated': True,
        'log_level': 'INFO'
    }
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    
    return default_config


def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "data_path": "data/food_demand_data.csv",
        "results_path": "results",
        "model_type": "auto",
        "forecast_horizon": 12,
        "target_skus": null,
        "include_history": true,
        "generate_aggregated": true,
        "log_level": "INFO"
    }
    
    config_file = Path('pipeline_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Sample configuration created: {config_file}")


def main():
    """Main pipeline execution function"""
    parser = argparse.ArgumentParser(description='Food Demand Forecasting Pipeline')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--steps', nargs='+', 
                       choices=['generate_data', 'train', 'forecast', 'evaluate'],
                       help='Specific pipeline steps to run')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration file')
    parser.add_argument('--model-type', type=str, default='auto',
                       choices=['auto', 'arima', 'sarima', 'holtwinters', 'prophet'],
                       help='Model type to train')
    parser.add_argument('--forecast-horizon', type=int, default=12,
                       help='Number of weeks to forecast')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_type != 'auto':
        config['model_type'] = args.model_type
    if args.forecast_horizon != 12:
        config['forecast_horizon'] = args.forecast_horizon
    
    # Initialize and run pipeline
    pipeline = ForecastingPipeline(config)
    
    if args.steps:
        success = pipeline.run_partial_pipeline(args.steps)
    else:
        success = pipeline.run_full_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {pipeline.results_path}")
        print("Check the logs and reports for detailed information.")
    else:
        print("\n" + "="*60)
        print("PIPELINE EXECUTION FAILED")
        print("="*60)
        print("Check the logs for error details.")
        sys.exit(1)


if __name__ == "__main__":
    main()