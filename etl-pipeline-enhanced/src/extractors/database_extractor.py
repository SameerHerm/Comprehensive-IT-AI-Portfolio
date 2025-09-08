"""
Database Data Extractor
Extracts data from various databases
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text
import psycopg2
import pymongo
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseExtractor:
    """Extract data from databases"""
    
    def __init__(self, db_type: str, connection_params: Dict[str, Any]):
        """Initialize database extractor"""
        self.db_type = db_type
        self.connection_params = connection_params
        self.engine = None
        
    @contextmanager
    def get_connection(self):
        """Get database connection context manager"""
        if self.db_type == 'postgresql':
            conn = psycopg2.connect(**self.connection_params)
        elif self.db_type == 'mysql':
            import pymysql
            conn = pymysql.connect(**self.connection_params)
        elif self.db_type == 'mongodb':
            client = pymongo.MongoClient(**self.connection_params)
            conn = client[self.connection_params.get('database', 'test')]
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        try:
            yield conn
        finally:
            if hasattr(conn, 'close'):
                conn.close()
    
    def create_sqlalchemy_engine(self):
        """Create SQLAlchemy engine"""
        if self.db_type == 'postgresql':
            url = f"postgresql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params.get('port', 5432)}/{self.connection_params['database']}"
        elif self.db_type == 'mysql':
            url = f"mysql+pymysql://{self.connection_params['user']}:{self.connection_params['password']}@{self.connection_params['host']}:{self.connection_params.get('port', 3306)}/{self.connection_params['database']}"
        else:
            raise ValueError(f"SQLAlchemy not supported for {self.db_type}")
        
        self.engine = create_engine(url)
        return self.engine
    
    def extract_table(self, table_name: str, schema: str = None,
                     columns: List[str] = None, limit: int = None) -> pd.DataFrame:
        """Extract data from a database table"""
        if not self.engine:
            self.create_sqlalchemy_engine()
        
        query = f"SELECT "
        
        if columns:
            query += ", ".join(columns)
        else:
            query += "*"
        
        query += f" FROM "
        
        if schema:
            query += f"{schema}.{table_name}"
        else:
            query += table_name
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"Executing query: {query}")
        
        df = pd.read_sql_query(query, self.engine)
        logger.info(f"Extracted {len(df)} rows from {table_name}")
        
        return df
    
    def extract_query(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Extract data using custom SQL query"""
        if not self.engine:
            self.create_sqlalchemy_engine()
        
        logger.info(f"Executing custom query")
        
        df = pd.read_sql_query(text(query), self.engine, params=params)
        logger.info(f"Extracted {len(df)} rows")
        
        return df
    
    def extract_incremental(self, table_name: str, timestamp_column: str,
                          last_timestamp: str, schema: str = None) -> pd.DataFrame:
        """Extract incremental data based on timestamp"""
        query = f"""
        SELECT *
        FROM {schema + '.' if schema else ''}{table_name}
        WHERE {timestamp_column} > :last_timestamp
        ORDER BY {timestamp_column}
        """
        
        return self.extract_query(query, {'last_timestamp': last_timestamp})
    
    def extract_partitioned(self, table_name: str, partition_column: str,
                          partition_value: Any, schema: str = None) -> pd.DataFrame:
        """Extract data from specific partition"""
        query = f"""
        SELECT *
        FROM {schema + '.' if schema else ''}{table_name}
        WHERE {partition_column} = :partition_value
        """
        
        return self.extract_query(query, {'partition_value': partition_value})
    
    def extract_mongodb_collection(self, collection_name: str,
                                  filter_query: Dict[str, Any] = None,
                                  projection: Dict[str, int] = None,
                                  limit: int = None) -> List[Dict[str, Any]]:
        """Extract data from MongoDB collection"""
        with self.get_connection() as db:
            collection = db[collection_name]
            
            cursor = collection.find(filter_query or {}, projection or {})
            
            if limit:
                cursor = cursor.limit(limit)
            
            data = list(cursor)
            logger.info(f"Extracted {len(data)} documents from {collection_name}")
            
            return data
    
    def extract_with_joins(self, main_table: str, joins: List[Dict[str, Any]],
                          columns: List[str] = None, where: str = None) -> pd.DataFrame:
        """Extract data with joins"""
        query = "SELECT "
        
        if columns:
            query += ", ".join(columns)
        else:
            query += f"{main_table}.*"
        
        query += f" FROM {main_table}"
        
        for join in joins:
            join_type = join.get('type', 'INNER')
            join_table = join['table']
            join_on = join['on']
            
            query += f" {join_type} JOIN {join_table} ON {join_on}"
        
        if where:
            query += f" WHERE {where}"
        
        return self.extract_query(query)
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict[str, Any]:
        """Get table metadata information"""
        if self.db_type == 'postgresql':
            query = """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_name = :table_name
            """
            
            if schema:
                query += " AND table_schema = :schema"
                params = {'table_name': table_name, 'schema': schema}
            else:
                params = {'table_name': table_name}
            
            columns_df = self.extract_query(query, params)
            
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {schema + '.' if schema else ''}{table_name}"
            count_df = self.extract_query(count_query)
            
            return {
                'columns': columns_df.to_dict('records'),
                'row_count': count_df['row_count'].iloc[0] if not count_df.empty else 0
            }
        
        return {}
