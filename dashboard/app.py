"""
Interactive Streamlit Dashboard for Food Demand Forecasting System
Provides visualization and analysis of forecasting results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.forecast import ForecastEngine
    from src.evaluate import ModelEvaluator
except ImportError:
    st.error("Could not import required modules. Please ensure you're running from the correct directory.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Food Demand Forecasting Dashboard",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .error-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class ForecastDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.results_path = Path("../results")
        self.data_path = Path("../data/food_demand_data.csv")
        
        # Initialize session state
        if 'loaded_data' not in st.session_state:
            st.session_state.loaded_data = False
    
    def load_data(self):
        """Load all required data files"""
        try:
            # Load historical data
            if self.data_path.exists():
                self.historical_data = pd.read_csv(self.data_path)
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            else:
                st.error(f"Historical data not found at {self.data_path}")
                return False
            
            # Load training results
            training_file = self.results_path / 'training_summary.json'
            if training_file.exists():
                with open(training_file, 'r') as f:
                    self.training_results = json.load(f)
            else:
                self.training_results = None
            
            # Load latest forecast results
            forecast_files = list(self.results_path.glob('forecasts_*.json'))
            if forecast_files:
                latest_forecast_file = max(forecast_files, key=lambda x: x.stat().st_mtime)
                with open(latest_forecast_file, 'r') as f:
                    self.forecast_results = json.load(f)
            else:
                self.forecast_results = None
            
            # Load evaluation results
            eval_files = list((self.results_path / 'reports').glob('evaluation_report_*.json'))
            if eval_files:
                latest_eval_file = max(eval_files, key=lambda x: x.stat().st_mtime)
                with open(latest_eval_file, 'r') as f:
                    self.evaluation_results = json.load(f)
            else:
                self.evaluation_results = None
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def render_sidebar(self):
        """Render sidebar with navigation and filters"""
        st.sidebar.title("🍕 Food Demand Forecasting")
        st.sidebar.markdown("---")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Historical Analysis", "Model Performance", "Forecasts", "SKU Deep Dive"]
        )
        
        st.sidebar.markdown("---")
        
        # SKU filter
        if hasattr(self, 'historical_data'):
            available_skus = sorted(self.historical_data['sku'].unique())
            selected_skus = st.sidebar.multiselect(
                "Filter SKUs",
                available_skus,
                default=available_skus[:3]
            )
        else:
            selected_skus = []
        
        # Date range filter
        if hasattr(self, 'historical_data'):
            min_date = self.historical_data['date'].min().date()
            max_date = self.historical_data['date'].max().date()
            
            date_range = st.sidebar.date_input(
                "Date Range",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_range = None
        
        return page, selected_skus, date_range
    
    def render_overview_page(self):
        """Render overview/summary page"""
        st.title("📊 Food Demand Forecasting Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if hasattr(self, 'historical_data'):
                total_skus = self.historical_data['sku'].nunique()
                st.metric("Total SKUs", total_skus)
            else:
                st.metric("Total SKUs", "N/A")
        
        with col2:
            if hasattr(self, 'historical_data'):
                total_weeks = self.historical_data['week'].nunique()
                st.metric("Historical Weeks", total_weeks)
            else:
                st.metric("Historical Weeks", "N/A")
        
        with col3:
            if self.training_results:
                avg_mape = np.mean([result['mape'] for result in self.training_results.values()])
                color = "success" if avg_mape <= 5.0 else "error"
                st.metric("Average MAPE", f"{avg_mape:.2f}%")
            else:
                st.metric("Average MAPE", "N/A")
        
        with col4:
            if self.forecast_results:
                forecast_horizon = len(next(iter(self.forecast_results['forecasts'].values()))['forecast_values'])
                st.metric("Forecast Horizon", f"{forecast_horizon} weeks")
            else:
                st.metric("Forecast Horizon", "N/A")
        
        # Success criteria assessment
        if self.evaluation_results:
            st.subheader("🎯 Success Criteria Assessment")
            
            criteria = self.evaluation_results['success_criteria_assessment']
            
            col1, col2 = st.columns(2)
            
            with col1:
                mape_status = criteria['mape_target_5_percent']['status']
                mape_color = "success" if mape_status == "PASS" else "error"
                st.markdown(f"**MAPE ≤ 5%**: <span class='{mape_color}-metric'>{mape_status}</span>", 
                          unsafe_allow_html=True)
                
                actual_mape = criteria['mape_target_5_percent']['achieved_average']
                st.write(f"Achieved: {actual_mape:.2f}%")
            
            with col2:
                smape_status = criteria['smape_target_6_percent']['status']
                smape_color = "success" if smape_status == "PASS" else "error"
                st.markdown(f"**SMAPE ≤ 6%**: <span class='{smape_color}-metric'>{smape_status}</span>", 
                          unsafe_allow_html=True)
                
                actual_smape = criteria['smape_target_6_percent']['achieved_average']
                st.write(f"Achieved: {actual_smape:.2f}%")
        
        # Model distribution
        if self.training_results:
            st.subheader("🤖 Model Distribution")
            
            model_counts = {}
            for result in self.training_results.values():
                model_type = result.get('model_type', 'Unknown')
                model_counts[model_type] = model_counts.get(model_type, 0) + 1
            
            fig = px.pie(values=list(model_counts.values()), 
                        names=list(model_counts.keys()),
                        title="Models Used by SKU")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_historical_analysis_page(self, selected_skus, date_range):
        """Render historical data analysis page"""
        st.title("📈 Historical Demand Analysis")
        
        if not hasattr(self, 'historical_data'):
            st.error("Historical data not available")
            return
        
        # Filter data
        filtered_data = self.historical_data.copy()
        
        if selected_skus:
            filtered_data = filtered_data[filtered_data['sku'].isin(selected_skus)]
        
        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = filtered_data[
                (filtered_data['date'].dt.date >= start_date) & 
                (filtered_data['date'].dt.date <= end_date)
            ]
        
        # Time series plot
        st.subheader("🔄 Demand Over Time")
        
        fig = px.line(filtered_data, x='date', y='demand', color='sku',
                     title='Weekly Demand by SKU')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal analysis
        st.subheader("🌸 Seasonal Patterns")
        
        seasonal_data = filtered_data.groupby(['sku', 'season'])['demand'].mean().reset_index()
        seasonal_pivot = seasonal_data.pivot(index='sku', columns='season', values='demand')
        
        fig = px.imshow(seasonal_pivot.values, 
                       x=seasonal_pivot.columns,
                       y=seasonal_pivot.index,
                       title="Average Demand by Season",
                       color_continuous_scale="Viridis")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Holiday and promotion effects
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎉 Holiday Effects")
            holiday_effect = filtered_data.groupby(['sku', 'is_holiday'])['demand'].mean().reset_index()
            holiday_pivot = holiday_effect.pivot(index='sku', columns='is_holiday', values='demand')
            
            if True in holiday_pivot.columns and False in holiday_pivot.columns:
                holiday_pivot['lift_%'] = (holiday_pivot[True] / holiday_pivot[False] - 1) * 100
                
                fig = px.bar(x=holiday_pivot.index, y=holiday_pivot['lift_%'],
                           title="Holiday Demand Lift (%)")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎊 Promotion Effects")
            promo_effect = filtered_data.groupby(['sku', 'is_promotion'])['demand'].mean().reset_index()
            promo_pivot = promo_effect.pivot(index='sku', columns='is_promotion', values='demand')
            
            if True in promo_pivot.columns and False in promo_pivot.columns:
                promo_pivot['lift_%'] = (promo_pivot[True] / promo_pivot[False] - 1) * 100
                
                fig = px.bar(x=promo_pivot.index, y=promo_pivot['lift_%'],
                           title="Promotion Demand Lift (%)")
                st.plotly_chart(fig, use_container_width=True)
    
    def render_model_performance_page(self):
        """Render model performance analysis page"""
        st.title("🎯 Model Performance Analysis")
        
        if not self.training_results:
            st.error("Training results not available")
            return
        
        # Performance metrics overview
        st.subheader("📊 Performance Metrics Overview")
        
        # Create performance dataframe
        perf_data = []
        for sku, results in self.training_results.items():
            perf_data.append({
                'SKU': sku,
                'Model': results.get('model_type', 'Unknown'),
                'MAPE (%)': results.get('mape', np.nan),
                'SMAPE (%)': results.get('smape', np.nan),
                'RMSE': results.get('rmse', np.nan),
                'AIC': results.get('aic', np.nan)
            })
        
        df_perf = pd.DataFrame(perf_data)
        
        # Display metrics table
        st.dataframe(df_perf.round(2), use_container_width=True)
        
        # Performance distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_perf, x='MAPE (%)', nbins=20,
                             title="MAPE Distribution")
            fig.add_vline(x=5, line_dash="dash", line_color="red", 
                         annotation_text="Target: 5%")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df_perf, x='SMAPE (%)', nbins=20,
                             title="SMAPE Distribution")
            fig.add_vline(x=6, line_dash="dash", line_color="red",
                         annotation_text="Target: 6%")
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.subheader("🔍 Model Type Comparison")
        
        fig = px.box(df_perf, x='Model', y='MAPE (%)',
                    title="MAPE by Model Type")
        st.plotly_chart(fig, use_container_width=True)
        
        # Best and worst performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Best Performers")
            best_skus = df_perf.nsmallest(5, 'MAPE (%)')[['SKU', 'Model', 'MAPE (%)']]
            st.dataframe(best_skus, hide_index=True)
        
        with col2:
            st.subheader("⚠️ Needs Improvement")
            worst_skus = df_perf.nlargest(5, 'MAPE (%)')[['SKU', 'Model', 'MAPE (%)']]
            st.dataframe(worst_skus, hide_index=True)
    
    def render_forecasts_page(self, selected_skus):
        """Render forecasts visualization page"""
        st.title("🔮 Demand Forecasts")
        
        if not self.forecast_results:
            st.error("Forecast results not available")
            return
        
        forecasts = self.forecast_results['forecasts']
        
        # Filter forecasts by selected SKUs
        if selected_skus:
            forecasts = {sku: forecast for sku, forecast in forecasts.items() 
                        if sku in selected_skus}
        
        if not forecasts:
            st.warning("No forecasts available for selected SKUs")
            return
        
        # Forecast summary
        st.subheader("📈 Forecast Summary")
        
        total_forecasted = sum(sum(f['forecast_values']) for f in forecasts.values())
        avg_weekly = total_forecasted / (len(forecasts) * len(next(iter(forecasts.values()))['forecast_values']))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Forecasted Demand", f"{total_forecasted:,.0f}")
        with col2:
            st.metric("Average Weekly Demand", f"{avg_weekly:,.0f}")
        with col3:
            st.metric("SKUs with Forecasts", len(forecasts))
        
        # Individual SKU forecasts
        st.subheader("📊 SKU Forecasts")
        
        for sku, forecast_data in forecasts.items():
            st.write(f"**{sku}** - Model: {forecast_data.get('model_type', 'Unknown')}")
            
            # Prepare data for plotting
            forecast_dates = pd.to_datetime(forecast_data['forecast_dates'])
            forecast_values = forecast_data['forecast_values']
            conf_lower = forecast_data['confidence_intervals']['lower']
            conf_upper = forecast_data['confidence_intervals']['upper']
            
            # Get historical data for this SKU
            if hasattr(self, 'historical_data'):
                hist_data = self.historical_data[self.historical_data['sku'] == sku].copy()
                hist_data = hist_data.sort_values('date')
                
                # Take last 24 weeks for context
                hist_data = hist_data.tail(24)
            else:
                hist_data = None
            
            # Create plot
            fig = go.Figure()
            
            # Historical data
            if hist_data is not None and len(hist_data) > 0:
                fig.add_trace(go.Scatter(
                    x=hist_data['date'],
                    y=hist_data['demand'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue')
                ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=conf_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=conf_lower,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255,0,0,0.2)',
                fill='tonexty',
                name='Confidence Interval',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f'{sku} - Demand Forecast',
                xaxis_title='Date',
                yaxis_title='Demand',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Aggregated forecast
        if len(forecasts) > 1:
            st.subheader("📊 Aggregated Forecast")
            
            # Calculate aggregated forecast
            forecast_dates = pd.to_datetime(next(iter(forecasts.values()))['forecast_dates'])
            aggregated_values = np.sum([f['forecast_values'] for f in forecasts.values()], axis=0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=aggregated_values,
                mode='lines+markers',
                name='Total Forecast',
                line=dict(color='green', width=3)
            ))
            
            fig.update_layout(
                title='Total Demand Forecast (All Selected SKUs)',
                xaxis_title='Date',
                yaxis_title='Total Demand',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_sku_deep_dive_page(self, selected_skus):
        """Render detailed SKU analysis page"""
        st.title("🔍 SKU Deep Dive")
        
        if not selected_skus:
            st.warning("Please select at least one SKU from the sidebar")
            return
        
        # SKU selector
        target_sku = st.selectbox("Select SKU for detailed analysis", selected_skus)
        
        if not hasattr(self, 'historical_data'):
            st.error("Historical data not available")
            return
        
        # Get SKU data
        sku_data = self.historical_data[self.historical_data['sku'] == target_sku].copy()
        sku_data = sku_data.sort_values('date')
        
        # Basic statistics
        st.subheader("📊 Basic Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Weeks", len(sku_data))
        with col2:
            st.metric("Average Demand", f"{sku_data['demand'].mean():.0f}")
        with col3:
            st.metric("Max Demand", f"{sku_data['demand'].max():.0f}")
        with col4:
            cv = sku_data['demand'].std() / sku_data['demand'].mean()
            st.metric("Coefficient of Variation", f"{cv:.3f}")
        
        # Time series decomposition visualization
        st.subheader("📈 Demand Patterns")
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Demand Over Time', 'Seasonal Pattern', 'Monthly Averages'],
            vertical_spacing=0.08
        )
        
        # Original time series
        fig.add_trace(
            go.Scatter(x=sku_data['date'], y=sku_data['demand'],
                      mode='lines+markers', name='Demand'),
            row=1, col=1
        )
        
        # Seasonal pattern
        seasonal_avg = sku_data.groupby('season')['demand'].mean()
        fig.add_trace(
            go.Bar(x=seasonal_avg.index, y=seasonal_avg.values, name='Seasonal Average'),
            row=2, col=1
        )
        
        # Monthly pattern
        monthly_avg = sku_data.groupby('month')['demand'].mean()
        fig.add_trace(
            go.Scatter(x=monthly_avg.index, y=monthly_avg.values,
                      mode='lines+markers', name='Monthly Average'),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance for this SKU
        if self.training_results and target_sku in self.training_results:
            st.subheader("🎯 Model Performance")
            
            sku_results = self.training_results[target_sku]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mape = sku_results.get('mape', 'N/A')
                st.metric("MAPE", f"{mape:.2f}%" if isinstance(mape, (int, float)) else mape)
            
            with col2:
                smape = sku_results.get('smape', 'N/A')
                st.metric("SMAPE", f"{smape:.2f}%" if isinstance(smape, (int, float)) else smape)
            
            with col3:
                model_type = sku_results.get('model_type', 'N/A')
                st.metric("Best Model", model_type)
        
        # Forecast for this SKU
        if self.forecast_results and target_sku in self.forecast_results['forecasts']:
            st.subheader("🔮 Forecast")
            
            forecast_data = self.forecast_results['forecasts'][target_sku]
            
            forecast_dates = pd.to_datetime(forecast_data['forecast_dates'])
            forecast_values = forecast_data['forecast_values']
            
            # Show forecast table
            forecast_df = pd.DataFrame({
                'Week': range(1, len(forecast_values) + 1),
                'Date': forecast_dates.strftime('%Y-%m-%d'),
                'Forecasted Demand': forecast_values,
                'Lower Bound': forecast_data['confidence_intervals']['lower'],
                'Upper Bound': forecast_data['confidence_intervals']['upper']
            })
            
            st.dataframe(forecast_df.round(0), hide_index=True)
    
    def run(self):
        """Main dashboard runner"""
        # Load data
        if not st.session_state.loaded_data:
            with st.spinner("Loading data..."):
                if self.load_data():
                    st.session_state.loaded_data = True
                    st.success("Data loaded successfully!")
                else:
                    st.error("Failed to load data. Please ensure all required files exist.")
                    return
        
        # Render sidebar and get selections
        page, selected_skus, date_range = self.render_sidebar()
        
        # Render selected page
        if page == "Overview":
            self.render_overview_page()
        elif page == "Historical Analysis":
            self.render_historical_analysis_page(selected_skus, date_range)
        elif page == "Model Performance":
            self.render_model_performance_page()
        elif page == "Forecasts":
            self.render_forecasts_page(selected_skus)
        elif page == "SKU Deep Dive":
            self.render_sku_deep_dive_page(selected_skus)
        
        # Footer
        st.markdown("---")
        st.markdown("**Food Demand Forecasting Dashboard** | Built with Streamlit")


if __name__ == "__main__":
    dashboard = ForecastDashboard()
    dashboard.run()