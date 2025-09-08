"""
Airflow module for ETL Pipeline Enhanced
"""

__version__ = "2.0.0"
__author__ = "ETL Pipeline Team"

# Import key components for easier access
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

__all__ = ['DAG', 'PythonOperator', 'BashOperator']
