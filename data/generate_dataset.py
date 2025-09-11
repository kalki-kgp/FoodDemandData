import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_food_demand_data():
    """Generate realistic food demand dataset for 145 weeks with multiple SKUs"""
    
    start_date = datetime(2021, 1, 4)  # Start on a Monday
    weeks = 145
    
    skus = [
        'pizza_margherita', 'pizza_pepperoni', 'pasta_carbonara', 'pasta_bolognese',
        'burger_classic', 'burger_veggie', 'salad_caesar', 'salad_greek',
        'sandwich_club', 'sandwich_blt', 'soup_tomato', 'soup_chicken'
    ]
    
    # Base demand patterns for different food types
    base_demands = {
        'pizza_margherita': 850, 'pizza_pepperoni': 1200, 
        'pasta_carbonara': 600, 'pasta_bolognese': 750,
        'burger_classic': 950, 'burger_veggie': 400, 
        'salad_caesar': 300, 'salad_greek': 250,
        'sandwich_club': 500, 'sandwich_blt': 450, 
        'soup_tomato': 200, 'soup_chicken': 350
    }
    
    # Seasonal multipliers (winter=1.2, spring=0.9, summer=0.8, fall=1.1)
    seasonal_patterns = {
        'pizza_margherita': [1.2, 0.9, 0.8, 1.1], 'pizza_pepperoni': [1.3, 0.9, 0.7, 1.1],
        'pasta_carbonara': [1.4, 0.8, 0.6, 1.2], 'pasta_bolognese': [1.3, 0.8, 0.7, 1.2],
        'burger_classic': [0.9, 1.0, 1.3, 1.1], 'burger_veggie': [0.8, 1.2, 1.4, 1.0],
        'salad_caesar': [0.7, 1.1, 1.5, 0.9], 'salad_greek': [0.6, 1.2, 1.6, 0.8],
        'sandwich_club': [0.9, 1.1, 1.2, 1.0], 'sandwich_blt': [0.8, 1.1, 1.3, 0.9],
        'soup_tomato': [1.8, 0.7, 0.4, 1.1], 'soup_chicken': [1.6, 0.8, 0.5, 1.2]
    }
    
    # Weekly patterns (Mon=0.8, Tue=0.9, Wed=1.0, Thu=1.1, Fri=1.4, Sat=1.3, Sun=1.2)
    weekly_pattern = [0.8, 0.9, 1.0, 1.1, 1.4, 1.3, 1.2]
    
    data = []
    
    for week in range(weeks):
        current_date = start_date + timedelta(weeks=week)
        week_of_year = current_date.isocalendar()[1]
        month = current_date.month
        
        # Determine season
        if month in [12, 1, 2]:
            season_idx = 0  # Winter
        elif month in [3, 4, 5]:
            season_idx = 1  # Spring
        elif month in [6, 7, 8]:
            season_idx = 2  # Summer
        else:
            season_idx = 3  # Fall
        
        # Special events (holidays, promotions)
        is_holiday = week_of_year in [1, 52] or month in [11, 12]  # New Year, Thanksgiving, Christmas
        is_promotion = week % 8 == 0  # Every 8 weeks promotion
        
        holiday_multiplier = 1.3 if is_holiday else 1.0
        promotion_multiplier = 1.25 if is_promotion else 1.0
        
        # Long-term trend (slight growth over time)
        trend_multiplier = 1 + (week * 0.001)
        
        for sku in skus:
            base_demand = base_demands[sku]
            seasonal_mult = seasonal_patterns[sku][season_idx]
            
            # Calculate weekly demand
            weekly_demand = (base_demand * 
                           seasonal_mult * 
                           trend_multiplier * 
                           holiday_multiplier * 
                           promotion_multiplier)
            
            # Add daily variation within the week
            daily_demands = []
            for day_idx in range(7):
                daily_demand = weekly_demand * weekly_pattern[day_idx] / 7
                
                # Add noise (±10% random variation)
                noise = np.random.normal(0, 0.1)
                daily_demand *= (1 + noise)
                
                daily_demands.append(max(0, int(daily_demand)))
            
            # Sum to get weekly total
            total_weekly_demand = sum(daily_demands)
            
            data.append({
                'week': week + 1,
                'date': current_date.strftime('%Y-%m-%d'),
                'sku': sku,
                'demand': total_weekly_demand,
                'is_holiday': is_holiday,
                'is_promotion': is_promotion,
                'season': ['winter', 'spring', 'summer', 'fall'][season_idx],
                'month': month
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_food_demand_data()
    df.to_csv('/Users/onepiece/Desktop/Data Projects/FoodDemandData/data/food_demand_data.csv', index=False)
    print(f"Generated dataset with {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"SKUs: {df['sku'].unique()}")
    print(f"Sample data:\n{df.head(10)}")