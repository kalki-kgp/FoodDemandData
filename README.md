# 🍕 Food Demand Forecasting System

A comprehensive time series forecasting system for predicting weekly food demand across multiple SKUs using advanced statistical models.

## 📋 Project Overview

This project implements an automated forecasting pipeline that predicts weekly meal demand with **<5% MAPE** accuracy using 145 weeks of historical order data. The system supports multiple seasonality patterns, external factors, and provides interactive dashboards for business insights.

### 🎯 Success Criteria
- **Prediction Accuracy**: MAPE ≤ 5%, SMAPE ≤ 6%
- **Reproducibility**: Works on unseen datasets with minimal configuration
- **Scalability**: Handles multiple SKUs and seasonalities
- **Usability**: Interactive dashboard for business decisions

## 🏗️ Project Structure

```
FoodDemandData/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── pipeline.py                # Main automated pipeline
├── .gitignore                 # Git ignore rules
├── data/                      # Data directory
│   ├── generate_dataset.py    # Dataset generation script
│   └── food_demand_data.csv   # Generated dataset (after running)
├── src/                       # Source code modules
│   ├── train_model.py         # Model training with hyperparameter tuning
│   ├── forecast.py           # Batch forecasting module
│   └── evaluate.py           # Model performance evaluation
├── notebooks/                 # Jupyter notebooks
│   └── eda_food_demand.ipynb  # Exploratory data analysis
├── dashboard/                 # Interactive dashboard
│   └── app.py                # Streamlit dashboard application
├── results/                   # Output directory (created during execution)
│   ├── models/               # Trained model files
│   ├── reports/              # Evaluation reports
│   └── *.csv, *.json        # Results and forecasts
└── models/                    # Saved model artifacts (created during execution)
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd FoodDemandData

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
# Generate the 145-week food demand dataset
python data/generate_dataset.py
```

### 3. Run Full Pipeline

```bash
# Run complete forecasting pipeline
python pipeline.py

# Or run specific steps
python pipeline.py --steps generate_data train forecast evaluate
```

### 4. Launch Dashboard

```bash
# Start interactive dashboard
cd dashboard
streamlit run app.py
```

## 📊 Usage Guide

### Individual Modules

#### Data Generation
```bash
# Generate synthetic food demand data
python data/generate_dataset.py
```

#### Model Training
```bash
# Train models for all SKUs with auto-selection
python src/train_model.py --data data/food_demand_data.csv --model auto

# Train specific model type
python src/train_model.py --model sarima

# Train for specific SKU only
python src/train_model.py --sku pizza_pepperoni --model auto
```

#### Forecasting
```bash
# Generate forecasts for all SKUs (12 weeks ahead)
python src/forecast.py --models results/models --data data/food_demand_data.csv --steps 12

# Generate forecasts for specific SKU
python src/forecast.py --sku pizza_margherita --steps 8

# Generate aggregated forecast
python src/forecast.py --aggregate --steps 12
```

#### Model Evaluation
```bash
# Evaluate model performance and generate reports
python src/evaluate.py --results results/

# Include forecast evaluation
python src/evaluate.py --forecast-file results/forecasts_20241201_120000.json
```

### Pipeline Options

#### Full Pipeline
```bash
# Run complete pipeline with default settings
python pipeline.py

# Run with custom forecast horizon
python pipeline.py --forecast-horizon 8

# Run with specific model type
python pipeline.py --model-type sarima
```

#### Partial Pipeline
```bash
# Generate data only
python pipeline.py --steps generate_data

# Train models only
python pipeline.py --steps train

# Generate forecasts only (requires trained models)
python pipeline.py --steps forecast

# Evaluate models only (requires training results)
python pipeline.py --steps evaluate
```

#### Configuration File
```bash
# Create sample configuration
python pipeline.py --create-config

# Run with configuration file
python pipeline.py --config pipeline_config.json
```

### Dashboard Usage

1. **Launch Dashboard**: `streamlit run dashboard/app.py`
2. **Navigate Pages**:
   - **Overview**: Key metrics and success criteria
   - **Historical Analysis**: Seasonal patterns and trends
   - **Model Performance**: Training results and comparisons
   - **Forecasts**: Future demand predictions with confidence intervals
   - **SKU Deep Dive**: Detailed analysis for individual SKUs

3. **Interactive Features**:
   - Filter by SKUs and date ranges
   - Compare model performance across SKUs
   - View confidence intervals for forecasts
   - Analyze seasonal and promotional effects

## 🤖 Supported Models

### Statistical Models
- **ARIMA**: Auto-Regressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA with seasonal components
- **Holt-Winters**: Exponential smoothing with trend and seasonality
- **Prophet**: Facebook's time series forecasting (optional)

### Model Selection
- **Automatic**: System selects best model per SKU based on AIC/MAPE
- **Manual**: Specify model type for all SKUs
- **Hybrid**: Different models for different SKUs

### Hyperparameter Tuning
- Grid search with AIC/BIC optimization
- Cross-validation for robust performance estimates
- Automated parameter selection for each SKU

## 📈 Features

### Data Analysis
- **Statistical EDA**: Mean trends, variance analysis, seasonal decomposition
- **ACF/PACF Analysis**: Autocorrelation and partial autocorrelation plots
- **Anomaly Detection**: IQR-based outlier identification
- **External Factors**: Holiday and promotion effect analysis

### Forecasting Capabilities
- **Multi-SKU Forecasting**: Batch processing for all products
- **Confidence Intervals**: Prediction uncertainty quantification
- **Seasonal Modeling**: Weekly and yearly seasonality
- **External Regressors**: Holiday and promotion variables

### Evaluation Metrics
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **RMSE**: Root Mean Square Error
- **R-squared**: Coefficient of determination

### Visualization
- **Interactive Plots**: Plotly-based charts with zoom/pan
- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **Forecast Plots**: Historical data + predictions + confidence bands
- **Performance Dashboards**: Model comparison and success criteria

## 🔧 Configuration Options

### Pipeline Configuration (`pipeline_config.json`)
```json
{
  "data_path": "data/food_demand_data.csv",
  "results_path": "results",
  "model_type": "auto",
  "forecast_horizon": 12,
  "target_skus": null,
  "include_history": true,
  "generate_aggregated": true,
  "log_level": "INFO"
}
```

### Available Model Types
- `auto`: Automatic model selection (recommended)
- `arima`: ARIMA models only
- `sarima`: Seasonal ARIMA models only  
- `holtwinters`: Holt-Winters exponential smoothing only
- `prophet`: Facebook Prophet models only (requires prophet package)

## 📊 Output Files

### Training Results
- `results/training_summary.json`: Model performance by SKU
- `results/model_performance.csv`: Performance metrics table
- `results/models/*.pkl`: Trained model artifacts

### Forecast Results  
- `results/forecasts_YYYYMMDD_HHMMSS.json`: Detailed forecast results
- `results/forecasts_YYYYMMDD_HHMMSS.csv`: Forecast data in tabular format
- `results/aggregated_forecast_total.json`: Aggregated forecasts

### Evaluation Reports
- `results/reports/evaluation_report_YYYYMMDD_HHMMSS.json`: Comprehensive evaluation
- `results/reports/model_performance_distribution.png`: Performance visualizations
- `results/reports/performance_comparison.html`: Interactive comparison plots

## 🧪 Sample Dataset

The generated dataset includes:
- **145 weeks** of historical data (Jan 2021 - Nov 2023)
- **12 SKUs** across different food categories
- **Seasonal patterns**: Summer salads, winter soups, etc.
- **External factors**: Holidays, promotions
- **Realistic demand**: Based on food delivery patterns

### SKUs Included
- Pizza: Margherita, Pepperoni
- Pasta: Carbonara, Bolognese  
- Burgers: Classic, Veggie
- Salads: Caesar, Greek
- Sandwiches: Club, BLT
- Soups: Tomato, Chicken

## 🔍 Performance Monitoring

### Success Metrics
- Models achieving **MAPE ≤ 5%**: Target performance
- Models achieving **SMAPE ≤ 6%**: Alternative accuracy measure
- **Model distribution**: Optimal model selection across SKUs
- **Forecast accuracy**: Out-of-sample validation performance

### Quality Checks
- Residual analysis for model diagnostics
- Ljung-Box test for autocorrelation
- Normality tests for residuals
- Confidence interval coverage

## 🚨 Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, ensure you're running from project root
cd /path/to/FoodDemandData
python pipeline.py
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# For Prophet (optional)
pip install prophet
```

#### Memory Issues
```bash
# Reduce SKU count for testing
python src/train_model.py --sku pizza_pepperoni

# Or run pipeline with specific SKUs
python pipeline.py --config pipeline_config.json
# (Edit config to specify target_skus)
```

#### Dashboard Not Loading
```bash
# Ensure results exist before launching dashboard
python pipeline.py  # Generate results first
cd dashboard
streamlit run app.py
```

### Performance Issues
- **High MAPE**: Consider adding external regressors or ensemble methods
- **Training Slow**: Reduce hyperparameter grid search space
- **Memory Usage**: Process SKUs individually or use chunking

## 📚 Technical Details

### Model Implementation
- **Statsmodels**: ARIMA/SARIMA implementation with MLE estimation
- **Scikit-learn**: Pipeline orchestration and cross-validation
- **Prophet**: Additive model with automatic seasonality detection
- **Plotly**: Interactive visualizations and dashboards

### Performance Optimization
- **Parallel Processing**: Multi-threading for independent SKU training
- **Caching**: Model artifacts cached for reuse
- **Efficient Storage**: JSON/CSV formats for interoperability
- **Memory Management**: Streaming processing for large datasets

### Validation Strategy
- **Time Series Split**: Respects temporal ordering
- **Walk-Forward Validation**: Multiple forecast origins
- **Out-of-Sample Testing**: 20% holdout for final evaluation

## 🔮 Future Enhancements

### Planned Features
- **Deep Learning Models**: LSTM, Temporal Fusion Transformer
- **External Data Integration**: Weather, economic indicators
- **Real-time Forecasting**: Streaming predictions
- **Model Ensembling**: Combination forecasts
- **A/B Testing Framework**: Model comparison in production

### Extensibility
- **Custom Models**: Easy integration of new algorithms
- **Plugin Architecture**: Modular evaluation metrics
- **API Interface**: REST API for forecast serving
- **Cloud Deployment**: Scalable cloud infrastructure

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the code documentation in individual modules

---

**Built with ❤️ using Python, Statsmodels, and Streamlit**