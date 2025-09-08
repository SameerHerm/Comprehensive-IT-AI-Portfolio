"""
File Loader Module
Loads data to various file formats
"""

import pandas as pd
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class FileLoader:
    """Load data to various file formats"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize file loader"""
        self.config = config or {}
        
    def load(self, data: pd.DataFrame, filepath: str, format: str = None, **kwargs):
        """Load data to file"""
        path = Path(filepath)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension if not provided
        if format is None:
            format = path.suffix[1:].lower()
        
        logger.info(f"Loading data to {filepath} (format: {format})")
        
        if format == 'csv':
            self.load_csv(data, filepath, **kwargs)
        elif format == 'json':
            self.load_json(data, filepath, **kwargs)
        elif format == 'parquet':
            self.load_parquet(data, filepath, **kwargs)
        elif format == 'excel' or format == 'xlsx':
            self.load_excel(data, filepath, **kwargs)
        elif format == 'feather':
            self.load_feather(data, filepath, **kwargs)
        elif format == 'pickle' or format == 'pkl':
            self.load_pickle(data, filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Successfully loaded {len(data)} rows to {filepath}")
    
    def load_csv(self, data: pd.DataFrame, filepath: str, **kwargs):
        """Load data to CSV file"""
        default_params = {
            'index': False,
            'encoding': 'utf-8'
        }
        default_params.update(kwargs)
        
        data.to_csv(filepath, **default_params)
    
    def load_json(self, data: pd.DataFrame, filepath: str, **kwargs):
        """Load data to JSON file"""
        default_params = {
            'orient': 'records',
            'date_format': 'iso',
            'indent': 2
        }
        default_params.update(kwargs)
        
        data.to_json(filepath, **default_params)
    
    def load_parquet(self, data: pd.DataFrame, filepath: str, **kwargs):
        """Load data to Parquet file"""
        default_params = {
            'engine': 'pyarrow',
            'compression': 'snappy',
            'index': False
        }
        default_params.update(kwargs)
        
        data.to_parquet(filepath, **default_params)
    
    def load_excel(self, data: pd.DataFrame, filepath: str, **kwargs):
        """Load data to Excel file"""
        sheet_name = kwargs.pop('sheet_name', 'Sheet1')
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False, **kwargs)
    
    def load_feather(self, data: pd.DataFrame, filepath: str, **kwargs):
        """Load data to Feather file"""
        data.to_feather(filepath, **kwargs)
    
    def load_pickle(self, data: pd.DataFrame, filepath: str, **kwargs):
        """Load data to Pickle file"""
        data.to_pickle(filepath, **kwargs)
    
    def load_partitioned(self, data: pd.DataFrame, base_path: str, 
                        partition_cols: list, format: str = 'parquet'):
        """Load data with partitioning"""
        logger.info(f"Loading partitioned data to {base_path}")
        
        if format == 'parquet':
            table = pa.Table.from_pandas(data)
            pq.write_to_dataset(
                table,
                root_path=base_path,
                partition_cols=partition_cols,
                compression='snappy'
            )
        else:
            # Manual partitioning for other formats
            for partition_values, group_df in data.groupby(partition_cols):
                if not isinstance(partition_values, tuple):
                    partition_values = (partition_values,)
                
                # Create partition path
                partition_path = Path(base_path)
                for col, val in zip(partition_cols, partition_values):
                    partition_path = partition_path / f"{col}={val}"
                
                partition_path.mkdir(parents=True, exist_ok=True)
                
                # Save partition
                filepath = partition_path / f"data.{format}"
                self.load(group_df, str(filepath), format)
        
        logger.info(f"Partitioned data loaded to {base_path}")
    
    def load_incremental(self, data: pd.DataFrame, filepath: str,
                        mode: str = 'append', format: str = 'csv'):
        """Load data incrementally"""
        path = Path(filepath)
        
        if mode == 'append' and path.exists():
            if format == 'csv':
                data.to_csv(filepath, mode='a', header=False, index=False)
            elif format == 'json':
                # Read existing data
                existing_data = pd.read_json(filepath)
                # Combine with new data
                combined_data = pd.concat([existing_data, data], ignore_index=True)
                # Write back
                self.load_json(combined_data, filepath)
            else:
                logger.warning(f"Append mode not supported for {format}, overwriting")
                self.load(data, filepath, format)
        else:
            self.load(data, filepath, format)
