"""
Data Aggregator Module
Performs various aggregation operations on data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Aggregator:
    """Perform data aggregation operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize aggregator"""
        self.config = config or {}
        
    def aggregate(self, df: pd.DataFrame, 
                 group_by: Union[str, List[str]],
                 aggregations: Dict[str, Union[str, List[str], Dict[str, str]]]) -> pd.DataFrame:
        """Perform general aggregation"""
        logger.info(f"Aggregating data by {group_by}")
        
        if isinstance(group_by, str):
            group_by = [group_by]
        
        # Convert aggregations to pandas format
        agg_dict = {}
        for col, agg_funcs in aggregations.items():
            if isinstance(agg_funcs, str):
                agg_dict[col] = agg_funcs
            elif isinstance(agg_funcs, list):
                agg_dict[col] = agg_funcs
            elif isinstance(agg_funcs, dict):
                agg_dict[col] = list(agg_funcs.values())
        
        # Perform aggregation
        result = df.groupby(group_by).agg(agg_dict)
        
        # Flatten column names if multi-level
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip() for col in result.columns.values]
        
        result = result.reset_index()
        
        logger.info(f"Aggregation complete. Result shape: {result.shape}")
        return result
    
    def time_series_aggregation(self, df: pd.DataFrame,
                               date_column: str,
                               value_columns: List[str],
                               freq: str = 'D',
                               agg_func: str = 'sum') -> pd.DataFrame:
        """Aggregate time series data"""
        logger.info(f"Performing time series aggregation with frequency: {freq}")
        
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Set date as index
        df_ts = df.set_index(date_column)
        
        # Resample and aggregate
        if agg_func == 'sum':
            result = df_ts[value_columns].resample(freq).sum()
        elif agg_func == 'mean':
            result = df_ts[value_columns].resample(freq).mean()
        elif agg_func == 'count':
            result = df_ts[value_columns].resample(freq).count()
        elif agg_func == 'max':
            result = df_ts[value_columns].resample(freq).max()
        elif agg_func == 'min':
            result = df_ts[value_columns].resample(freq).min()
        else:
            result = df_ts[value_columns].resample(freq).agg(agg_func)
        
        result = result.reset_index()
        
        # Add derived time features
        result['year'] = result[date_column].dt.year
        result['month'] = result[date_column].dt.month
        result['quarter'] = result[date_column].dt.quarter
        result['day_of_week'] = result[date_column].dt.dayofweek
        result['week_of_year'] = result[date_column].dt.isocalendar().week
        
        return result
    
    def rolling_aggregation(self, df: pd.DataFrame,
                          date_column: str,
                          value_columns: List[str],
                          window: int = 7,
                          agg_func: str = 'mean') -> pd.DataFrame:
        """Calculate rolling aggregations"""
        logger.info(f"Calculating rolling {agg_func} with window: {window}")
        
        # Sort by date
        df = df.sort_values(date_column)
        
        for col in value_columns:
            if agg_func == 'mean':
                df[f'{col}_rolling_{window}'] = df[col].rolling(window=window).mean()
            elif agg_func == 'sum':
                df[f'{col}_rolling_{window}'] = df[col].rolling(window=window).sum()
            elif agg_func == 'std':
                df[f'{col}_rolling_{window}'] = df[col].rolling(window=window).std()
            elif agg_func == 'max':
                df[f'{col}_rolling_{window}'] = df[col].rolling(window=window).max()
            elif agg_func == 'min':
                df[f'{col}_rolling_{window}'] = df[col].rolling(window=window).min()
        
        return df
    
    def pivot_aggregation(self, df: pd.DataFrame,
                         index: Union[str, List[str]],
                         columns: str,
                         values: str,
                         agg_func: str = 'sum',
                         fill_value: Any = 0) -> pd.DataFrame:
        """Create pivot table with aggregation"""
        logger.info(f"Creating pivot table")
        
        pivot_table = pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=agg_func,
            fill_value=fill_value
        )
        
        # Reset index to convert to regular dataframe
        pivot_table = pivot_table.reset_index()
        
        return pivot_table
    
    def calculate_metrics(self, df: pd.DataFrame,
                         metrics: List[Dict[str, Any]]) -> pd.DataFrame:
        """Calculate custom metrics"""
        logger.info(f"Calculating {len(metrics)} custom metrics")
        
        result_df = df.copy()
        
        for metric in metrics:
            metric_name = metric['name']
            metric_type = metric['type']
            
            if metric_type == 'ratio':
                numerator = metric['numerator']
                denominator = metric['denominator']
                result_df[metric_name] = result_df[numerator] / result_df[denominator].replace(0, np.nan)
                
            elif metric_type == 'percentage':
                value_col = metric['value']
                total_col = metric['total']
                result_df[metric_name] = (result_df[value_col] / result_df[total_col]) * 100
                
            elif metric_type == 'difference':
                col1 = metric['column1']
                col2 = metric['column2']
                result_df[metric_name] = result_df[col1] - result_df[col2]
                
            elif metric_type == 'growth_rate':
                value_col = metric['value']
                result_df[metric_name] = result_df[value_col].pct_change() * 100
                
            elif metric_type == 'cumulative':
                value_col = metric['value']
                result_df[metric_name] = result_df[value_col].cumsum()
                
            elif metric_type == 'rank':
                value_col = metric['value']
                ascending = metric.get('ascending', False)
                result_df[metric_name] = result_df[value_col].rank(ascending=ascending)
                
            elif metric_type == 'custom':
                formula = metric['formula']
                result_df[metric_name] = eval(formula, {"df": result_df, "np": np, "pd": pd})
        
        return result_df
    
    def aggregate_by_time_window(self, df: pd.DataFrame,
                                timestamp_column: str,
                                windows: List[str],
                                value_columns: List[str],
                                agg_funcs: List[str]) -> Dict[str, pd.DataFrame]:
        """Aggregate data by multiple time windows"""
        results = {}
        
        for window in windows:
            logger.info(f"Aggregating by {window} window")
            
            # Convert window to pandas frequency
            freq_map = {
                'hourly': 'H',
                'daily': 'D',
                'weekly': 'W',
                'monthly': 'M',
                'quarterly': 'Q',
                'yearly': 'Y'
            }
            
            freq = freq_map.get(window, 'D')
            
            window_df = self.time_series_aggregation(
                df,
                timestamp_column,
                value_columns,
                freq,
                agg_funcs[0] if agg_funcs else 'sum'
            )
            
            results[window] = window_df
        
        return results
    
    def create_summary_statistics(self, df: pd.DataFrame,
                                 columns: List[str] = None) -> pd.DataFrame:
        """Create comprehensive summary statistics"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        summary = []
        
        for col in columns:
            if col in df.columns:
                stats = {
                    'column': col,
                    'count': df[col].count(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'q25': df[col].quantile(0.25),
                    'median': df[col].median(),
                    'q75': df[col].quantile(0.75),
                    'max': df[col].max(),
                    'null_count': df[col].isnull().sum(),
                    'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                    'unique_count': df[col].nunique(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                }
                summary.append(stats)
        
        return pd.DataFrame(summary)
