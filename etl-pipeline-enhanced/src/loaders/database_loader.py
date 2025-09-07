"""
Database Loader Module
Handles loading data to various databases
"""

import pandas as pd
import logging
from typing import Any, Dict, Optional
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extras import execute_batch

logger = logging.getLogger(__name__)

class DatabaseLoader:
    """Load data to various databases"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database loader"""
        self.config = config or {}
        self.engine = None
        
    def create_connection(self, db_type: str, connection_string: str):
        """Create database connection"""
        if db_type == 'postgresql':
            self.engine = create_engine(connection_string)
        elif db_type == 'mysql':
            self.engine = create_engine(connection_string)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        logger.info(f"Connected to {db_type} database")
        
    def load_to_postgres(self, data: pd.DataFrame, table: str, schema: str = 'public') -> int:
        """Load data to PostgreSQL"""
        try:
            # Create connection string
            conn_string = f"postgresql://{self.config.get('user', 'airflow')}:{self.config.get('password', 'airflow')}@{self.config.get('host', 'postgres')}:{self.config.get('port', 5432)}/{self.config.get('database', 'airflow')}"
            
            engine = create_engine(conn_string)
            
            # Load data
            records_loaded = data.to_sql(
                name=table,
                con=engine,
                schema=schema,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info(f"Loaded {len(data)} records to {schema}.{table}")
            return len(data)
            
        except Exception as e:
            logger.error(f"Error loading to PostgreSQL: {e}")
            raise
    
    def bulk_insert(self, data: pd.DataFrame, table: str, schema: str = 'public'):
        """Bulk insert using COPY"""
        try:
            conn = psycopg2.connect(
                host=self.config.get('host', 'postgres'),
                port=self.config.get('port', 5432),
                database=self.config.get('database', 'airflow'),
                user=self.config.get('user', 'airflow'),
                password=self.config.get('password', 'airflow')
            )
            
            cur = conn.cursor()
            
            # Create temp file
            import io
            output = io.StringIO()
            data.to_csv(output, sep='\t', header=False, index=False)
            output.seek(0)
            
            # COPY data
            cur.copy_from(output, f'{schema}.{table}', null='')
            conn.commit()
            
            cur.close()
            conn.close()
            
            logger.info(f"Bulk inserted {len(data)} records")
            
        except Exception as e:
            logger.error(f"Error in bulk insert: {e}")
            raise
