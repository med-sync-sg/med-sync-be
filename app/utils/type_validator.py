"""
Neo4j data type mapper for the MedSync application.

This module provides utilities for mapping between Neo4j TemplateField data_type values
and their actual representation in the application, handling validation and conversion.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Type
from datetime import datetime
import re
import json
import logging

# Configure logger
logger = logging.getLogger(__name__)

class DataTypeValidator:
    """Definition of a data type with validation and conversion functions"""
    
    def __init__(
        self,
        name: str,
        description: str,
        python_type: Type,
        validators: List[Callable[[Any], bool]] = None,
        converter: Callable[[Any], Any] = None,
        default_value: Any = None,
        is_complex: bool = False,
        allowed_values: List[Any] = None,
        format_pattern: Optional[str] = None,
        additional_props: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a data type definition.
        
        Args:
            name: Name of the data type
            description: Description of the data type
            python_type: Python type corresponding to this data type
            validators: List of validation functions
            converter: Function to convert value to the correct type
            default_value: Default value for this type
            is_complex: Whether this is a complex type (e.g. nested objects)
            allowed_values: List of allowed values (for enum-like types)
            format_pattern: Regex pattern for string validation
            additional_props: Additional properties for this type
        """
        self.name = name
        self.description = description
        self.python_type = python_type
        self.validators = validators or []
        self.converter = converter
        self.default_value = default_value
        self.is_complex = is_complex
        self.allowed_values = allowed_values
        self.format_pattern = format_pattern
        self.additional_props = additional_props or {}
    
    def validate(self, value: Any) -> bool:
        """
        Validate a value against this data type.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Skip validation if value is None and we allow that
        if value is None:
            return True
            
        # Check type
        if not isinstance(value, self.python_type) and self.converter is None:
            return False
            
        # Apply type conversion if available
        if self.converter is not None:
            try:
                value = self.converter(value)
            except (ValueError, TypeError):
                return False
        
        # Check allowed values if specified
        if self.allowed_values is not None and value not in self.allowed_values:
            return False
            
        # Check format pattern if specified
        if self.format_pattern is not None and isinstance(value, str):
            if not re.match(self.format_pattern, value):
                return False
        
        # Run all validators
        for validator in self.validators:
            if not validator(value):
                return False
                
        return True
    
    def convert(self, value: Any) -> Any:
        """
        Convert a value to this data type.
        
        Args:
            value: Value to convert
            
        Returns:
            Converted value or default value if conversion fails
        """
        if value is None:
            return self.default_value
            
        if self.converter is not None:
            try:
                return self.converter(value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting value '{value}' to {self.name}: {str(e)}")
                return self.default_value
                
        # If no converter specified, try direct conversion to python_type
        try:
            return self.python_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting value '{value}' to {self.name}: {str(e)}")
            return self.default_value


class NestedTypeDefinition(DataTypeValidator):
    """Definition for a nested object type with sub-fields"""
    
    def __init__(
        self,
        name: str,
        description: str,
        field_definitions: Dict[str, DataTypeValidator],
        **kwargs
    ):
        """
        Initialize a nested type definition.
        
        Args:
            name: Name of the data type
            description: Description of the data type
            field_definitions: Dictionary of field name to DataTypeDefinition
            **kwargs: Additional arguments passed to DataTypeDefinition
        """
        super().__init__(
            name=name,
            description=description,
            python_type=dict,
            is_complex=True,
            **kwargs
        )
        self.field_definitions = field_definitions
    
    def validate(self, value: Any) -> bool:
        """
        Validate a nested object against its field definitions.
        
        Args:
            value: Dictionary object to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not super().validate(value):
            return False
            
        if not isinstance(value, dict):
            return False
            
        # Validate each field
        for field_name, field_def in self.field_definitions.items():
            if field_name in value:
                if not field_def.validate(value[field_name]):
                    return False
                    
        return True
    
    def convert(self, value: Any) -> Dict[str, Any]:
        """
        Convert a value to a properly formatted nested object.
        
        Args:
            value: Value to convert (dict or JSON string)
            
        Returns:
            Converted dictionary with properly typed fields
        """
        if value is None:
            return {}
            
        # Convert from JSON string if needed
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
                
        if not isinstance(value, dict):
            return {}
            
        # Convert each field
        result = {}
        for field_name, field_def in self.field_definitions.items():
            if field_name in value:
                result[field_name] = field_def.convert(value[field_name])
                
        return result


class DataTypeEnum(str, Enum):
    """Enum of available data types"""
    STRING = "string"
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    CODE = "code"
    ANY = "any"


class DataTypeMapper:
    """
    Manager class for data type mappings between Neo4j and the application.
    
    This class provides methods for:
    - Validating values against type definitions
    - Converting values to the appropriate types
    - Getting type information for template fields
    """
    
    def __init__(self):
        """Initialize the data type mapper with standard type definitions"""
        self.type_definitions = self._initialize_type_definitions()
    
    def _initialize_type_definitions(self) -> Dict[str, DataTypeValidator]:
        """
        Initialize standard data type definitions
        
        Returns:
            Dictionary of data_type name to DataTypeDefinition
        """
        # Helper converters
        def convert_date(value: Any) -> Optional[str]:
            """Convert to ISO format date string"""
            if isinstance(value, datetime):
                return value.strftime('%Y-%m-%d')
            if isinstance(value, str):
                try:
                    # Try to parse as ISO date
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    pass
                    
                # Try common formats
                formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
                for fmt in formats:
                    try:
                        dt = datetime.strptime(value, fmt)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            return None
            
        def convert_datetime(value: Any) -> Optional[str]:
            """Convert to ISO format datetime string"""
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, str):
                try:
                    # Try to parse as ISO datetime
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return dt.isoformat()
                except ValueError:
                    pass
                    
                # Try common formats
                formats = [
                    '%Y-%m-%dT%H:%M:%S', 
                    '%Y-%m-%d %H:%M:%S',
                    '%m/%d/%Y %H:%M:%S',
                    '%d-%m-%Y %H:%M:%S'
                ]
                for fmt in formats:
                    try:
                        dt = datetime.strptime(value, fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue
            return None
        
        # Define standard type definitions
        definitions = {
            # Basic types
            DataTypeEnum.STRING: DataTypeValidator(
                name="string",
                description="Short string value",
                python_type=str,
                default_value=""
            ),
            
            DataTypeEnum.INTEGER: DataTypeValidator(
                name="integer",
                description="Integer value",
                python_type=int,
                default_value=0,
                converter=lambda v: int(float(v)) if isinstance(v, (str, float)) else int(v)
            ),
            
            DataTypeEnum.FLOAT: DataTypeValidator(
                name="float",
                description="Floating point number",
                python_type=float,
                default_value=0.0,
                converter=lambda v: float(v) if isinstance(v, (str, int)) else float(v)
            ),
            
            DataTypeEnum.BOOLEAN: DataTypeValidator(
                name="boolean",
                description="Boolean value",
                python_type=bool,
                default_value=False,
                converter=lambda v: (
                    v if isinstance(v, bool) else 
                    v.lower() in ('true', 'yes', '1', 't', 'y') if isinstance(v, str) else 
                    bool(v)
                )
            ),
            
            DataTypeEnum.DATE: DataTypeValidator(
                name="date",
                description="Date value (ISO format)",
                python_type=str,
                default_value=None,
                converter=convert_date,
                format_pattern=r'^\d{4}-\d{2}-\d{2}$'
            ),
            
            DataTypeEnum.TIME: DataTypeValidator(
                name="time",
                description="Time value (HH:MM:SS)",
                python_type=str,
                default_value=None,
                format_pattern=r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$'
            ),
            
            DataTypeEnum.DATETIME: DataTypeValidator(
                name="datetime",
                description="Date and time value (ISO format)",
                python_type=str,
                default_value=None,
                converter=convert_datetime
            ),
            
            # Medical code types
            DataTypeEnum.CODE: DataTypeValidator(
                name="code",
                description="Generic medical code",
                python_type=str,
                default_value="",
                additional_props={
                    "code_system": None
                }
            ),
            
            DataTypeEnum.ANY: DataTypeValidator(
                name="any",
                description="Any Python object",
                python_type=Any,
                default_value=None,
                is_complex=True,
                converter=lambda v: v
            )
        }
        
        return definitions
    
    def validate(self, value: Any, data_type: str) -> bool:
        """
        Validate a value against a data type
        
        Args:
            value: Value to validate
            data_type: Data type name
            
        Returns:
            True if valid, False otherwise
        """
        type_def = self.get_type_definition(data_type)
        if not type_def:
            logger.warning(f"Unknown data type: {data_type}")
            return False
            
        return type_def.validate(value)
    
    def convert(self, value: Any, data_type: str) -> Any:
        """
        Convert a value to the specified data type
        
        Args:
            value: Value to convert
            data_type: Data type name
            
        Returns:
            Converted value
        """
        type_def = self.get_type_definition(data_type)
        if not type_def:
            logger.warning(f"Unknown data type: {data_type}")
            return value
            
        return type_def.convert(value)
    
    def get_type_definition(self, data_type: str) -> Optional[DataTypeValidator]:
        """
        Get the definition for a data type
        
        Args:
            data_type: Data type name
            
        Returns:
            DataTypeDefinition or None if not found
        """
        if data_type in self.type_definitions:
            return self.type_definitions[data_type]
            
        # Handle case-insensitive lookup
        data_type_lower = data_type.lower()
        for key, type_def in self.type_definitions.items():
            if key.lower() == data_type_lower:
                return type_def
                
        return None
    
    def get_python_type(self, data_type: str) -> Type:
        """
        Get the Python type for a data type
        
        Args:
            data_type: Data type name
            
        Returns:
            Python type class
        """
        type_def = self.get_type_definition(data_type)
        if not type_def:
            return str  # Default to string
            
        return type_def.python_type
    
    def get_default_value(self, data_type: str) -> Any:
        """
        Get the default value for a data type
        
        Args:
            data_type: Data type name
            
        Returns:
            Default value
        """
        type_def = self.get_type_definition(data_type)
        if not type_def:
            return ""  # Default to empty string
            
        return type_def.default_value
    
    def is_complex_type(self, data_type: str) -> bool:
        """
        Check if a data type is complex (e.g. nested object)
        
        Args:
            data_type: Data type name
            
        Returns:
            True if complex, False otherwise
        """
        type_def = self.get_type_definition(data_type)
        if not type_def:
            return False
            
        return type_def.is_complex
    
    def register_type(self, type_def: DataTypeValidator) -> None:
        """
        Register a new data type definition
        
        Args:
            type_def: DataTypeDefinition to register
        """
        self.type_definitions[type_def.name] = type_def
        logger.info(f"Registered new data type: {type_def.name}")
    
    def register_enum_type(self, name: str, allowed_values: List[str], description: str = None) -> None:
        """
        Register a new enum type with allowed values
        
        Args:
            name: Name of the enum type
            allowed_values: List of allowed values
            description: Optional description
        """
        type_def = DataTypeValidator(
            name=name,
            description=description or f"Enum of {name} values",
            python_type=str,
            default_value=allowed_values[0] if allowed_values else "",
            allowed_values=allowed_values,
            additional_props={
                "allowed_values": allowed_values
            }
        )
        self.register_type(type_def)
    
    def register_nested_type(self, name: str, field_definitions: Dict[str, str], 
                            description: str = None) -> None:
        """
        Register a new nested object type with field definitions
        
        Args:
            name: Name of the nested type
            field_definitions: Dictionary of field name to data_type
            description: Optional description
        """
        # Convert field type names to definitions
        field_defs = {}
        for field_name, field_type in field_definitions.items():
            type_def = self.get_type_definition(field_type)
            if type_def:
                field_defs[field_name] = type_def
            else:
                logger.warning(f"Unknown field type {field_type} for field {field_name} in nested type {name}")
                field_defs[field_name] = self.type_definitions[DataTypeEnum.STRING]
        
        nested_def = NestedTypeDefinition(
            name=name,
            description=description or f"Nested object for {name}",
            field_definitions=field_defs
        )
        self.register_type(nested_def)


# Create the global instance for application-wide use
data_type_mapper = DataTypeMapper()


def process_field_value(field_value: Any, data_type: str) -> Any:
    """
    Process a field value according to its data type
    
    Args:
        field_value: Value to process
        data_type: Data type name
        
    Returns:
        Processed value
    """
    return data_type_mapper.convert(field_value, data_type)


def validate_field_value(field_value: Any, data_type: str) -> bool:
    """
    Validate a field value against its data type
    
    Args:
        field_value: Value to validate
        data_type: Data type name
        
    Returns:
        True if valid, False otherwise
    """
    return data_type_mapper.validate(field_value, data_type)