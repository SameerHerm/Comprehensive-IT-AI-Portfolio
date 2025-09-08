"""
API response utilities
"""

from flask import jsonify
from typing import Any, Dict, Optional

def create_response(data: Any = None, message: str = None, 
                   status_code: int = 200, success: bool = True) -> tuple:
    """Create standardized API response"""
    response = {
        'success': success,
        'status_code': status_code
    }
    
    if message:
        response['message'] = message
    
    if data is not None:
        response['data'] = data
    
    return jsonify(response), status_code

def success_response(data: Any = None, message: str = 'Success', 
                    status_code: int = 200) -> tuple:
    """Create success response"""
    return create_response(data, message, status_code, True)

def error_response(message: str, status_code: int = 400, 
                  error_details: Optional[Dict] = None) -> tuple:
    """Create error response"""
    response = {
        'success': False,
        'status_code': status_code,
        'error': {
            'message': message
        }
    }
    
    if error_details:
        response['error']['details'] = error_details
    
    return jsonify(response), status_code

def paginated_response(items: list, page: int, per_page: int, 
                      total: int, **kwargs) -> tuple:
    """Create paginated response"""
    total_pages = (total + per_page - 1) // per_page
    
    response = {
        'success': True,
        'data': items,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    }
    
    # Add any additional fields
    response.update(kwargs)
    
    return jsonify(response), 200
