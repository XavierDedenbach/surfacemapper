"""
Utilities for data serialization.
"""
import numpy as np
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

def make_json_serializable(data: Any) -> Any:
    """
    Recursively traverses a data structure to convert non-serializable
    types to their native Python equivalents.
    
    Handles:
    - NumPy arrays and scalars
    - SciPy objects (Delaunay, etc.)
    - Custom objects with __dict__ or __slots__
    - Sets and other collections
    """
    try:
        if data is None:
            return None
            
        # Handle dictionaries
        if isinstance(data, dict):
            return {k: make_json_serializable(v) for k, v in data.items()}
            
        # Handle lists and tuples
        if isinstance(data, (list, tuple)):
            return [make_json_serializable(i) for i in data]
            
        # Handle sets
        if isinstance(data, set):
            return list(make_json_serializable(i) for i in data)
            
        # Handle NumPy arrays
        if isinstance(data, np.ndarray):
            return data.tolist()
            
        # Handle NumPy scalars - use proper type checking
        if hasattr(data, 'dtype') and hasattr(data, 'item'):
            # This is a NumPy scalar
            return data.item()
            
        # Handle SciPy objects (Delaunay, etc.)
        if hasattr(data, '__class__') and 'scipy' in str(data.__class__):
            logger.warning(f"Converting SciPy object {type(data)} to serializable format")
            return _convert_scipy_object(data)
            
        # Handle objects with __dict__ (custom classes)
        if hasattr(data, '__dict__'):
            logger.warning(f"Converting object {type(data)} to dict")
            return make_json_serializable(data.__dict__)
            
        # Handle objects with __slots__
        if hasattr(data, '__slots__'):
            logger.warning(f"Converting slotted object {type(data)} to dict")
            return make_json_serializable({slot: getattr(data, slot, None) for slot in data.__slots__})
            
        # Handle basic Python types that are already serializable
        if isinstance(data, (str, int, float, bool)):
            return data
            
        # For any other type, try to convert to string or log warning
        logger.warning(f"Unknown type {type(data)} in serialization, converting to string")
        return str(data)
        
    except Exception as e:
        logger.error(f"Error serializing {type(data)}: {e}")
        return f"<SerializationError: {type(data).__name__}>"

def _convert_scipy_object(obj: Any) -> Union[Dict, List, None]:
    """
    Convert SciPy objects to serializable formats.
    """
    try:
        # Handle Delaunay triangulation objects
        if hasattr(obj, 'simplices') and hasattr(obj, 'points'):
            return {
                'type': 'delaunay_triangulation',
                'points': obj.points.tolist() if hasattr(obj.points, 'tolist') else obj.points,
                'simplices': obj.simplices.tolist() if hasattr(obj.simplices, 'tolist') else obj.simplices
            }
            
        # Handle other SciPy objects with common attributes
        if hasattr(obj, 'points'):
            return {
                'type': 'scipy_object',
                'points': obj.points.tolist() if hasattr(obj.points, 'tolist') else obj.points
            }
            
        # Generic SciPy object conversion
        return {
            'type': 'scipy_object',
            'class_name': obj.__class__.__name__,
            'attributes': {k: make_json_serializable(v) for k, v in obj.__dict__.items() 
                          if not k.startswith('_')}
        }
        
    except Exception as e:
        logger.error(f"Error converting SciPy object {type(obj)}: {e}")
        return None

def validate_json_serializable(data: Any) -> bool:
    """
    Validate that data can be serialized to JSON without errors.
    """
    try:
        import json
        json.dumps(data)
        return True
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization validation failed: {e}")
        return False

def safe_json_serialize(data: Any) -> str:
    """
    Safely serialize data to JSON string with comprehensive error handling.
    """
    try:
        import json
        serializable_data = make_json_serializable(data)
        
        if not validate_json_serializable(serializable_data):
            logger.error("Data still not JSON serializable after conversion")
            return json.dumps({"error": "Serialization failed"})
            
        return json.dumps(serializable_data)
        
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        return json.dumps({"error": f"Serialization error: {str(e)}"}) 

def clean_floats_for_json(data):
    """Recursively replace NaN, inf, -inf with None for JSON compliance."""
    if isinstance(data, dict):
        return {k: clean_floats_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_floats_for_json(v) for v in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
        return data
    else:
        return data 