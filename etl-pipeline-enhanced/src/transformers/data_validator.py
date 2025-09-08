"""
Data Validator Module
Validates data quality and integrity
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate data quality and integrity"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data validator"""
        self.config = config or {}
        self.validation_results = []
        
    def validate(self, df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate dataframe against rules"""
        logger.info(f"Starting validation for dataframe with {len(df)} rows")
        
        results = []
        all_passed = True
        
        for rule in rules:
            rule_name = rule.get('name', 'unnamed_rule')
            rule_type = rule.get('type')
            
            logger.info(f"Applying rule: {rule_name}")
            
            if rule_type == 'null_check':
                result = self.check_nulls(df, rule)
            elif rule_type == 'duplicate_check':
                result = self.check_duplicates(df, rule)
            elif rule_type == 'range_check':
                result = self.check_range(df, rule)
            elif rule_type == 'pattern_check':
                result = self.check_pattern(df, rule)
            elif rule_type == 'referential_check':
                result = self.check_referential_integrity(df, rule)
            elif rule_type == 'custom_check':
                result = self.check_custom(df, rule)
            else:
                logger.warning(f"Unknown rule type: {rule_type}")
                continue
            
            results.append({
                'rule_name': rule_name,
                'rule_type': rule_type,
                'passed': result['passed'],
                'message': result.get('message', ''),
                'details': result.get('details', {})
            })
            
            if not result['passed']:
                all_passed = False
                logger.warning(f"Rule failed: {rule_name} - {result.get('message', '')}")
        
        self.validation_results = results
        return all_passed, results
    
    def check_nulls(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Check for null values"""
        columns = rule.get('columns', df.columns.tolist())
        threshold = rule.get('threshold', 0)  # Maximum allowed null percentage
        
        null_counts = {}
        failed_columns = []
        
        for col in columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                null_counts[col] = {
                    'count': int(null_count),
                    'percentage': round(null_percentage, 2)
                }
                
                if null_percentage > threshold:
                    failed_columns.append(col)
        
        passed = len(failed_columns) == 0
        
        return {
            'passed': passed,
            'message': f"Found nulls in columns: {failed_columns}" if not passed else "No null violations",
            'details': {
                'null_counts': null_counts,
                'failed_columns': failed_columns,
                'threshold': threshold
            }
        }
    
    def check_duplicates(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Check for duplicate records"""
        columns = rule.get('columns', None)  # None means check all columns
        
        if columns:
            duplicates = df.duplicated(subset=columns, keep=False)
        else:
            duplicates = df.duplicated(keep=False)
        
        duplicate_count = duplicates.sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        max_allowed = rule.get('max_duplicates', 0)
        passed = duplicate_count <= max_allowed
        
        return {
            'passed': passed,
            'message': f"Found {duplicate_count} duplicate records ({duplicate_percentage:.2f}%)",
            'details': {
                'duplicate_count': int(duplicate_count),
                'duplicate_percentage': round(duplicate_percentage, 2),
                'sample_duplicates': df[duplicates].head(10).to_dict('records') if duplicate_count > 0 else []
            }
        }
    
    def check_range(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Check if values are within specified range"""
        column = rule.get('column')
        min_value = rule.get('min_value')
        max_value = rule.get('max_value')
        
        if column not in df.columns:
            return {
                'passed': False,
                'message': f"Column {column} not found",
                'details': {}
            }
        
        violations = pd.Series([False] * len(df))
        
        if min_value is not None:
            violations |= df[column] < min_value
        
        if max_value is not None:
            violations |= df[column] > max_value
        
        violation_count = violations.sum()
        passed = violation_count == 0
        
        return {
            'passed': passed,
            'message': f"Found {violation_count} values outside range [{min_value}, {max_value}]",
            'details': {
                'violation_count': int(violation_count),
                'min_value': min_value,
                'max_value': max_value,
                'actual_min': float(df[column].min()) if not df[column].empty else None,
                'actual_max': float(df[column].max()) if not df[column].empty else None
            }
        }
    
    def check_pattern(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Check if values match specified pattern"""
        column = rule.get('column')
        pattern = rule.get('pattern')
        
        if column not in df.columns:
            return {
                'passed': False,
                'message': f"Column {column} not found",
                'details': {}
            }
        
        # Convert to string and check pattern
        string_values = df[column].astype(str)
        matches = string_values.str.match(pattern)
        
        violation_count = (~matches).sum()
        passed = violation_count == 0
        
        return {
            'passed': passed,
            'message': f"Found {violation_count} values not matching pattern",
            'details': {
                'violation_count': int(violation_count),
                'pattern': pattern,
                'sample_violations': df[~matches].head(5)[column].tolist() if violation_count > 0 else []
            }
        }
    
    def check_referential_integrity(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Check referential integrity with another dataset"""
        column = rule.get('column')
        reference_values = rule.get('reference_values', [])
        
        if column not in df.columns:
            return {
                'passed': False,
                'message': f"Column {column} not found",
                'details': {}
            }
        
        unique_values = df[column].unique()
        missing_references = set(unique_values) - set(reference_values)
        
        violation_count = len(missing_references)
        passed = violation_count == 0
        
        return {
            'passed': passed,
            'message': f"Found {violation_count} values without reference",
            'details': {
                'violation_count': violation_count,
                'missing_references': list(missing_references)[:10]  # Limit to first 10
            }
        }
    
    def check_custom(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom validation function"""
        func = rule.get('function')
        
        if not callable(func):
            return {
                'passed': False,
                'message': "Invalid custom function",
                'details': {}
            }
        
        try:
            result = func(df)
            
            if isinstance(result, bool):
                return {
                    'passed': result,
                    'message': "Custom validation " + ("passed" if result else "failed"),
                    'details': {}
                }
            elif isinstance(result, dict):
                return result
            else:
                return {
                    'passed': False,
                    'message': "Invalid custom function return type",
                    'details': {}
                }
                
        except Exception as e:
            return {
                'passed': False,
                'message': f"Custom validation error: {str(e)}",
                'details': {}
            }
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, str]) -> Dict[str, Any]:
        """Validate dataframe schema"""
        actual_schema = dict(df.dtypes.astype(str))
        
        missing_columns = set(expected_schema.keys()) - set(actual_schema.keys())
        extra_columns = set(actual_schema.keys()) - set(expected_schema.keys())
        type_mismatches = {}
        
        for col, expected_type in expected_schema.items():
            if col in actual_schema:
                actual_type = actual_schema[col]
                if not self._types_compatible(actual_type, expected_type):
                    type_mismatches[col] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
        
        passed = len(missing_columns) == 0 and len(type_mismatches) == 0
        
        return {
            'passed': passed,
            'message': "Schema validation " + ("passed" if passed else "failed"),
            'details': {
                'missing_columns': list(missing_columns),
                'extra_columns': list(extra_columns),
                'type_mismatches': type_mismatches
            }
        }
    
    def _types_compatible(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible"""
        # Simple compatibility check - can be extended
        type_groups = {
            'numeric': ['int64', 'int32', 'float64', 'float32', 'int', 'float'],
            'string': ['object', 'string', 'str'],
            'datetime': ['datetime64[ns]', 'datetime', 'timestamp']
        }
        
        for group in type_groups.values():
            if actual in group and expected in group:
                return True
        
        return actual == expected
    
    def generate_validation_report(self) -> str:
        """Generate validation report"""
        if not self.validation_results:
            return "No validation results available"
        
        report = ["Data Validation Report", "=" * 50]
        
        total_rules = len(self.validation_results)
        passed_rules = sum(1 for r in self.validation_results if r['passed'])
        
        report.append(f"Total Rules: {total_rules}")
        report.append(f"Passed: {passed_rules}")
        report.append(f"Failed: {total_rules - passed_rules}")
        report.append("")
        
        for result in self.validation_results:
            status = "✓" if result['passed'] else "✗"
            report.append(f"{status} {result['rule_name']} ({result['rule_type']})")
            report.append(f"  {result['message']}")
            
            if not result['passed'] and result.get('details'):
                for key, value in result['details'].items():
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        report.append(f"  {key}: [truncated]")
                    else:
                        report.append(f"  {key}: {value}")
            report.append("")
        
        return "\n".join(report)
