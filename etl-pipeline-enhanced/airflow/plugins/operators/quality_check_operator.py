"""
Custom Data Quality Check Operator for Airflow
"""

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.hooks.postgres_hook import PostgresHook
import logging

logger = logging.getLogger(__name__)

class DataQualityCheckOperator(BaseOperator):
    """
    Perform data quality checks on database tables
    """
    
    template_fields = ['table', 'date_filter']
    ui_color = '#89DA59'
    
    @apply_defaults
    def __init__(
        self,
        table,
        checks,
        postgres_conn_id='postgres_default',
        date_filter=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.table = table
        self.checks = checks
        self.postgres_conn_id = postgres_conn_id
        self.date_filter = date_filter
    
    def execute(self, context):
        """Execute quality checks"""
        logger.info(f"Starting data quality checks for table: {self.table}")
        
        postgres_hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        
        failed_checks = []
        
        for check in self.checks:
            check_name = check.get('name', 'unnamed_check')
            check_query = check.get('query')
            expected_result = check.get('expected_result')
            comparison = check.get('comparison', 'equals')
            
            logger.info(f"Running check: {check_name}")
            
            # Add date filter if provided
            if self.date_filter and '{{ date_filter }}' in check_query:
                check_query = check_query.replace('{{ date_filter }}', self.date_filter)
            
            # Execute query
            result = postgres_hook.get_first(check_query)[0]
            
            # Validate result
            check_passed = False
            if comparison == 'equals':
                check_passed = result == expected_result
            elif comparison == 'greater_than':
                check_passed = result > expected_result
            elif comparison == 'less_than':
                check_passed = result < expected_result
            elif comparison == 'greater_than_or_equals':
                check_passed = result >= expected_result
            elif comparison == 'less_than_or_equals':
                check_passed = result <= expected_result
            elif comparison == 'not_equals':
                check_passed = result != expected_result
            
            if not check_passed:
                failed_checks.append({
                    'check_name': check_name,
                    'expected': expected_result,
                    'actual': result
                })
                logger.error(f"Check failed: {check_name}. Expected {expected_result}, got {result}")
            else:
                logger.info(f"Check passed: {check_name}")
        
        if failed_checks:
            raise ValueError(f"Data quality checks failed: {failed_checks}")
        
        logger.info(f"All quality checks passed for table: {self.table}")
        return True
