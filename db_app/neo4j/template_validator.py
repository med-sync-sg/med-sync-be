import re
import logging
import datetime
from typing import Any, Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)

def get_field_validation_rules(field_data: Dict[str, Any], parents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract validation rules from a field's data and its inheritance chain
    
    Args:
        field_data: Dictionary of field properties
        parents: List of parent fields in inheritance chain (optional)
        
    Returns:
        Dictionary of validation rules
    """
    # Start with field's own properties
    validation_rules = {
        "data_type": field_data.get("data_type", "string"),
        "required": field_data.get("required", False)
    }
    
    # Add specific validation rules based on data type
    for key in ["min_value", "max_value", "format", "pattern", 
                "allowed_values", "min_length", "max_length"]:
        if key in field_data:
            validation_rules[key] = field_data[key]
    
    # Inherit validation rules from parents if not already set
    if parents:
        for parent in parents:
            for key, value in parent.items():
                if key not in validation_rules and key not in ["id", "name", "description", "data_type"]:
                    validation_rules[key] = value
    
    return validation_rules

def validate_field_value(value: Any, validation_rules: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a value against field validation rules
    
    Args:
        value: Value to validate
        validation_rules: Dictionary of validation rules
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if value is None:
        # Check if field is required
        if validation_rules.get("required", False):
            return False, "Value is required"
        return True, ""
    
    data_type = validation_rules.get("data_type", "string")
    
    # Type validation
    if data_type == "string" or data_type == "text":
        if not isinstance(value, str):
            return False, f"Value must be a string, got {type(value).__name__}"
            
        # Length validation if specified
        if "min_length" in validation_rules and len(value) < validation_rules["min_length"]:
            return False, f"String too short (minimum {validation_rules['min_length']} characters)"
        if "max_length" in validation_rules and len(value) > validation_rules["max_length"]:
            return False, f"String too long (maximum {validation_rules['max_length']} characters)"
            
        # Pattern validation if specified
        if "pattern" in validation_rules:
            if not re.match(validation_rules["pattern"], value):
                return False, f"Value does not match required pattern"
                
    elif data_type == "number" or data_type == "integer":
        # Convert string to number if possible
        if isinstance(value, str):
            try:
                if data_type == "integer":
                    value = int(value)
                else:
                    value = float(value)
            except ValueError:
                return False, f"Value must be a valid number"
                
        if not isinstance(value, (int, float)):
            return False, f"Value must be a number, got {type(value).__name__}"
            
        # Integer validation
        if data_type == "integer" and not isinstance(value, int) and not value.is_integer():
            return False, "Value must be an integer"
            
        # Range validation
        if "min_value" in validation_rules and value < validation_rules["min_value"]:
            return False, f"Value too small (minimum {validation_rules['min_value']})"
        if "max_value" in validation_rules and value > validation_rules["max_value"]:
            return False, f"Value too large (maximum {validation_rules['max_value']})"
            
    elif data_type == "boolean":
        # Convert string to boolean if possible
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ["true", "1", "yes"]:
                value = True
            elif lower_value in ["false", "0", "no"]:
                value = False
            else:
                return False, "Value must be a boolean (true/false)"
                
        if not isinstance(value, bool):
            return False, f"Value must be a boolean, got {type(value).__name__}"
            
    elif data_type == "temporal" or data_type == "date" or data_type == "time":
        # Date/time validation based on format
        if "format" in validation_rules:
            if isinstance(value, str):
                try:
                    if data_type == "date":
                        datetime.datetime.strptime(value, validation_rules["format"])
                    elif data_type == "time":
                        datetime.datetime.strptime(value, validation_rules["format"]).time()
                    else:
                        datetime.datetime.strptime(value, validation_rules["format"])
                except ValueError:
                    return False, f"Invalid {data_type} format (expected {validation_rules['format']})"
    
    # Check allowed values if specified
    if "allowed_values" in validation_rules and value not in validation_rules["allowed_values"]:
        return False, f"Value must be one of: {', '.join(str(v) for v in validation_rules['allowed_values'])}"
    
    return True, ""

def validate_field_values(field_values: Dict[str, Any], 
                          template_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate multiple field values against their field definitions
    
    Args:
        field_values: Dictionary mapping field names to values
        template_fields: List of field definitions from template
        
    Returns:
        List of validation errors (empty if all valid)
    """
    # Create field map for easy lookup
    field_map = {f["field_name"]: f for f in template_fields}
    
    # Track validation errors
    validation_errors = []
    
    # Process each field value
    for field_name, value in field_values.items():
        # Check if field exists in template
        if field_name not in field_map:
            validation_errors.append({
                "field": field_name,
                "error": "Field not found in template"
            })
            continue
            
        # Get field data
        field_data = field_map[field_name]
        
        # Get validation rules
        validation_rules = get_field_validation_rules({
            "data_type": field_data.get("data_type", "string"),
            "required": field_data.get("required", False)
        })
        
        # Validate field value
        is_valid, error = validate_field_value(value, validation_rules)
        if not is_valid:
            validation_errors.append({
                "field": field_name,
                "error": error
            })
    
    return validation_errors