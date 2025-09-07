"""
File Extractor Module
Handles extraction from various file formats
"""

import pandas as pd
import json
import yaml
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class FileExtractor:
    """Extract data from various file formats"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize file extractor"""
        self.config = config or {}
        self.supported_formats = ['csv', 'json', 'parquet', 'excel', 'yaml']
        
    def extract(self, filepath: str, format: Optional[str] = None) -> pd.DataFrame:
        """Extract data from file"""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Determine format from extension if not provided
        if format is None:
            format = path.suffix[1:].lower()
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Extracting data from {filepath} (format: {format})")
        
        # Call appropriate extraction method
        extract_method = getattr(self, f"extract_{format}")
        return extract_method(filepath)
    
    def extract_csv(self, filepath: str) -> pd.DataFrame:
        """Extract data from CSV file"""
        try:
            df = pd.read_csv(
                filepath,
                encoding=self.config.get('encoding', 'utf-8'),
                sep=self.config.get('separator', ','),
                parse_dates=self.config.get('parse_dates', True),
                low_memory=False
            )
            logger.info(f"Extracted {len(df)} rows from CSV")
            return df
        except Exception as e:
            logger.error(f"Error extracting CSV: {e}")
            raise
    
    def extract_json(self, filepath: str) -> pd.DataFrame:
        """Extract data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            logger.info(f"Extracted {len(df)} rows from JSON")
            return df
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            raise
    
    def extract_parquet(self, filepath: str) -> pd.DataFrame:
        """Extract data from Parquet file"""
        try:
            df = pd.read_parquet(filepath)
            logger.info(f"Extracted {len(df)} rows from Parquet")
            return df
        except Exception as e:
            logger.error(f"Error extracting Parquet: {e}")
            raise
    
    def extract_excel(self, filepath: str) -> pd.DataFrame:
        """Extract data from Excel file"""
        try:
            sheet_name = self.config.get('sheet_name', 0)
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            logger.info(f"Extracted {len(df)} rows from Excel")
            return df
        except Exception as e:
            logger.error(f"Error extracting Excel: {e}")
            raise
    
    def extract_yaml(self, filepath: str) -> pd.DataFrame:
        """Extract data from YAML file"""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            df = pd.DataFrame(data if isinstance(data, list) else [data])
            logger.info(f"Extracted {len(df)} rows from YAML")
            return df
        except Exception as e:
            logger.error(f"Error extracting YAML: {e}")
            raise
    
    def extract_multiple(self, filepaths: List[str]) -> pd.DataFrame:
        """Extract and combine data from multiple files"""
        dfs = []
        for filepath in filepaths:
            df = self.extract(filepath)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} files into {len(combined_df)} rows")
        return combined_df
