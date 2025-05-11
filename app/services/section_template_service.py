import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from app.db.neo4j_session import neo4j_session
from app.utils.nlp.nlp_utils import cosine_similarity
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

# Data type-related functions
DATA_TYPE_GROUPS = {
    "text_group": ["string", "text"],
    "numeric_group": ["integer", "float", "number"],
    "date_group": ["date", "datetime", "time"],
    "boolean_group": ["boolean"],
    "code_group": ["code"],
    "complex_group": ["object", "any"],
}

# Create a reverse mapping for quick lookup
DATA_TYPE_TO_GROUP = {}
for group_name, types in DATA_TYPE_GROUPS.items():
    for data_type in types:
        DATA_TYPE_TO_GROUP[data_type] = group_name

def get_data_type_group(data_type: str) -> str:
    """
    Get the compatibility group for a data type
    
    Args:
        data_type: Data type string
        
    Returns:
        Group name or 'other' if not found
    """
    return DATA_TYPE_TO_GROUP.get(data_type.lower(), "other")

def are_data_types_compatible(type1: str, type2: str) -> bool:
    """
    Check if two data types are compatible
    
    Args:
        type1: First data type
        type2: Second data type
        
    Returns:
        True if compatible, False otherwise
    """
    # If either type is missing, assume compatibility
    if not type1 or not type2:
        return True
        
    # Get groups for both types
    group1 = get_data_type_group(type1)
    group2 = get_data_type_group(type2)
    
    # Types in the same group are compatible
    return group1 == group2

def infer_value_data_type(value: Any) -> str:
    """
    Infer the most likely data type for a value
    
    Args:
        value: Value to analyze
        
    Returns:
        Inferred data type
    """
    if value is None:
        return "string"  # Default to string for None values
        
    if isinstance(value, bool):
        return "boolean"
        
    if isinstance(value, int):
        return "integer"
        
    if isinstance(value, float):
        return "float"
        
    if isinstance(value, list):
        return "list"
        
    if isinstance(value, dict):
        return "object"
        
    # Handle string values
    if isinstance(value, str):
        value_lower = value.lower().strip()
        
        # Check for date/time patterns
        if any(pattern in value_lower for pattern in ["-", "/"]) and any(c.isdigit() for c in value_lower):
            if ":" in value_lower:  # Has time component
                return "datetime"
            else:
                return "date"
                
        # Check for boolean-like values
        if value_lower in ["true", "false", "yes", "no", "y", "n"]:
            return "boolean"
            
        # Check for numeric strings
        try:
            float(value_lower)
            if "." in value_lower:
                return "float"
            else:
                return "integer"
        except ValueError:
            pass
        
        # Default to string for all other cases
        return "string"
    
    # Default for any other type
    return "any"



class SectionTemplateService:
    """
    Service for managing section templates and fields
    """
    
    def __init__(self):
        """Initialize the template service"""
        self.neo4j = neo4j_session
    
    def find_templates_by_text(self, search_text: str, is_doctor: bool=True, similarity_threshold: float = 0.65, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find section templates by semantic similarity to text
        
        Args:
            search_text: Text to search for
            similarity_threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
            
        Returns:
            List of matching templates with similarity scores
        """
        # Generate embedding for the query
        embedding = self.neo4j.get_embedding(search_text)
        
        # Search for templates using vector search
        results = self.neo4j.run_vector_search(
            label="SectionTemplate",
            embedding_field="embedding_1",
            vector=embedding,
            similarity_threshold=similarity_threshold,
            is_doctor=is_doctor,
            limit=limit
        )
        # Format results
        templates = []
        for result in results:
            node = result.get("n", {})
            print(search_text, node.get("id"), result.get("similarity"))
            if node.get("id") == "base-section":
                templates.append({
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "description": node.get("description"),
                    "soap_category": node.get("soap_category", "OTHER"),
                    "system_defined": node.get("system_defined", False),
                    "version": node.get("version", "1.0"),
                    "similarity_score": result.get("similarity", 0) - 0.1 # Do not use base section unless absolutely needed
                })
            else:
                templates.append({
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "description": node.get("description"),
                    "soap_category": node.get("soap_category", "OTHER"),
                    "system_defined": node.get("system_defined", False),
                    "version": node.get("version", "1.0"),
                    "similarity_score": result.get("similarity", 0)
                })
            
        return templates
    
    def get_base_fields(self) -> List[Dict[str, Any]]:
        """
        Retrieve all base-level TemplateField nodes from Neo4j.
        
        This method fetches all template fields that extend the base field
        (with ID 'base-field'). These fields serve as the foundation for 
        all field types in the application.
        
        Returns:
            List of field dictionaries with all properties
        """
        try:
            # Query for fields that extend the base field
            query = """
            MATCH (field:TemplateField)-[:EXTENDS*]->(base:TemplateField {id: 'base-field'})
            WHERE NOT (field)<-[:EXTENDS]-(:TemplateField)
            RETURN field, 
                labels(field) as labels,
                [(field)-[:EXTENDS]->(parent:TemplateField) | parent.id] as direct_parents
            ORDER BY field.id
            """
            
            # Alternative query if you want ALL fields extending base-field at any level
            all_fields_query = """
            MATCH (field:TemplateField)-[:EXTENDS*]->(base:TemplateField {id: 'base-field'})
            RETURN field, 
                labels(field) as labels,
                [(field)-[:EXTENDS]->(parent:TemplateField) | parent.id] as direct_parents
            ORDER BY field.id
            """
            
            # Execute the query
            results = self.neo4j.run_query(query)
            
            if not results:
                logger.warning("No base template fields found extending 'base-field'")
                return []
            
            # Format the results
            fields = []
            for result in results:
                field_node = result.get("field", {})
                labels = result.get("labels", [])
                direct_parents = result.get("direct_parents", [])
                
                # Copy properties from the node
                field_data = dict(field_node)
                
                # Add labels and parent info
                field_data["labels"] = labels
                field_data["direct_parents"] = direct_parents
                
                # Check if the field has vector embedding
                field_data["has_embedding"] = "embedding_1" in field_node
                
                # Add to result list
                fields.append(field_data)
            
            logger.info(f"Retrieved {len(fields)} base template fields")
            return fields
            
        except Exception as e:
            logger.error(f"Error retrieving base template fields: {str(e)}")
            return []

    def get_template_with_fields(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a template with all its fields
        
        Args:
            template_id: ID of the template
            
        Returns:
            Template with fields or None if not found
        """
        query = """
        MATCH (template:SectionTemplate {id: $template_id})
        OPTIONAL MATCH (template)-[r:HAS_FIELD]->(field:TemplateField)
        RETURN template, collect([r, field]) as fields
        """
        
        results = self.neo4j.run_query(query, {"template_id": template_id})
        
        if not results or "template" not in results[0]:
            return None
            
        template_node = results[0].get("template", {})
        rel_fields = results[0].get("fields")

        # Build template object
        template = {
            "id": template_node.get("id"),
            "name": template_node.get("name"),
            "description": template_node.get("description"),
            "soap_category": template_node.get("soap_category", "OTHER"),
            "system_defined": template_node.get("system_defined", False),
            "created_at": template_node.get("created_at"),
            "created_by": template_node.get("created_by"),
            "version": template_node.get("version", "1.0"),
            "fields": []
        }

        if rel_fields == None:
            logger.info(f"No fields or relationships found for section template {template_id}.")
            return template
            
        if len(rel_fields) < 1:
            logger.info(f"No fields associated with the section template {template_id} has been found.")
            return template

        
        for data in rel_fields:
            if data == None:
                continue
            
            if len(data) != 2:
                logger.warning("Invalid relationship or field!")
                continue
            relationship = data[0]
            field_data = data[1]
            
            if field_data:
                field = {
                    "id": field_data.get("id"),
                    "name": field_data.get("name"),
                    "description": field_data.get("description"),
                    "data_type": field_data.get("data_type"),
                    "field_name": field_data.get("name"),
                    "required": field_data.get("required", False),
                    "order": field_data.get("order", 0)
                }
                template["fields"].append(field)
                
        # Sort fields by order
        template["fields"].sort(key=lambda f: f.get("order", 0))
        
        return template
    
    def get_all_templates(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """
        Get all available templates
        
        Args:
            include_system: Whether to include system-defined templates
            
        Returns:
            List of templates
        """
        query = """
        MATCH (t:SectionTemplate)
        WHERE $include_system OR t.system_defined = false
        RETURN t as template
        ORDER BY t.name
        """
        
        results = self.neo4j.run_query(query, {"include_system": include_system})
        
        templates = []
        for result in results:
            template_node = result.get("template", {})
            
            templates.append({
                "id": template_node.get("id"),
                "name": template_node.get("name"),
                "soap_category": template_node.get("soap_category", "OTHER"),
                "description": template_node.get("description"),
                "system_defined": template_node.get("system_defined", False),
                "version": template_node.get("version", "1.0")
            })
            
        return templates
    
    def create_template(self, template_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new template
        
        Args:
            template_data: Template data including name, description
            
        Returns:
            ID of the created template or None if failed
        """
        # Generate a unique ID if not provided
        if "id" not in template_data:
            template_data["id"] = f"template-{uuid.uuid4()}"
            
        # Generate embedding from name and description
        embedding_text = f"{template_data.get('name', '')} {template_data.get('description', '')}"
        embedding = self.neo4j.get_embedding(embedding_text)
        
        query = """
        CREATE (t:SectionTemplate {
            id: $id,
            name: $name,
            description: $description,
            soap_category: $soap_category,
            embedding_1: $embedding,
            created_at: timestamp(),
            created_by: $created_by,
            system_defined: $system_defined,
            version: $version
        })
        RETURN t.id as id
        """
        
        # Set default values
        template_params = {
            "id": template_data.get("id"),
            "name": template_data.get("name"),
            "description": template_data.get("description", ""),
            "soap_category": template_data.get("soap_category", "OTHER"),
            "embedding_1": embedding,
            "created_by": template_data.get("created_by", "user"),
            "system_defined": template_data.get("system_defined", False),
            "version": template_data.get("version", "1.0")
        }
        
        results = self.neo4j.run_query(query, template_params)
        
        if results and "id" in results[0]:
            return results[0]["id"]
        return None
    
    def add_field_to_template(self, template_id: str, field_data: Dict[str, Any]) -> bool:
        """
        Add a field to a template
        
        Args:
            template_id: ID of the template
            field_data: Field data including id, field_name, required, order
            
        Returns:
            True if successful, False otherwise
        """
        query = """
        MATCH (t:SectionTemplate {id: $template_id})
        MATCH (f:TemplateField {id: $field_id})
        MERGE (t)-[r:HAS_FIELD {name: $field_name}]->(f)
        SET r.required = $required,
            r.order = $order,
            r.created_at = timestamp()
        RETURN count(r) as count
        """
        
        params = {
            "template_id": template_id,
            "field_id": field_data.get("id"),
            "field_name": field_data.get("field_name"),
            "required": field_data.get("required", False),
            "order": field_data.get("order", 0)
        }
        
        results = self.neo4j.run_query(query, params)
        
        if results and results[0].get("count", 0) > 0:
            return True
        return False
    
    def find_or_create_field(self, field_data: Dict[str, Any]) -> str:
        """
        Find a field by type or create it if it doesn't exist
        
        Args:
            field_data: Field data
            
        Returns:
            ID of the field
        """
        # Generate a unique ID if not provided
        if "id" not in field_data:
            # Create a slug-like ID based on the field type and name
            base_id = f"{field_data.get('data_type', 'text')}-{field_data.get('name', 'field')}".lower()
            field_data["id"] = f"{base_id}-{uuid.uuid4()}"
            
        # First try to find an existing field
        query = """
        MATCH (f:TemplateField)
        WHERE f.name = $name AND f.data_type = $data_type
        RETURN f.id as id
        LIMIT 1
        """
        
        params = {
            "name": field_data.get("name"),
            "data_type": field_data.get("data_type", "text")
        }
        
        results = self.neo4j.run_query(query, params)
        
        if results and "id" in results[0]:
            return results[0]["id"]
        
        # If not found, create a new field
        # Generate embedding from name, description and data type
        embedding_text = f"{field_data.get('name', '')} {field_data.get('description', '')} {field_data.get('data_type', 'text')}"
        embedding = self.neo4j.get_embedding(embedding_text)
        
        create_query = """
        CREATE (f:TemplateField {
            id: $id,
            name: $name,
            description: $description,
            data_type: $data_type,
            required: $required,
            embedding_1: $embedding,
            system_defined: $system_defined,
            created_at: timestamp()
        })
        RETURN f.id as id
        """
        
        create_params = {
            "id": field_data.get("id"),
            "name": field_data.get("name"),
            "description": field_data.get("description", ""),
            "data_type": field_data.get("data_type", "text"),
            "required": field_data.get("required", False),
            "embedding_1": embedding,
            "system_defined": field_data.get("system_defined", False)
        }
        
        create_results = self.neo4j.run_query(create_query, create_params)
        
        if create_results and "id" in create_results[0]:
            return create_results[0]["id"]
        
        # If creation failed, return the original ID
        return field_data.get("id")
    
    def find_best_matching_field(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a TemplateField node that best matches the input context
        
        Args:
            context: Dictionary containing information to match against fields
                - should contain at least 'text' or 'name' for embedding generation
                - can contain 'data_type' to restrict search
        
        Returns:
            Dictionary with field data or None if no good match found
        """
        # Create embedding from context
        embedding_text = context.get('text', '')
        if not embedding_text:
            # Use name if text not provided
            embedding_text = context.get('name', '')
            
            # Add description if available
            if 'description' in context:
                embedding_text += f" {context['description']}"
                
            # Add data type if available
            if 'data_type' in context:
                embedding_text += f" {context['data_type']}"
        
        # Validate we have something to search with
        if not embedding_text:
            return None
        
        # Generate embedding
        embedding = self.neo4j.get_embedding(embedding_text)
        
        # Base query for vector similarity search
        query = """
        MATCH (f:TemplateField)
        WHERE f.embedding_1 IS NOT NULL
        """
        
        # Add data type filter if provided
        if 'data_type' in context:
            query += " AND f.data_type = $data_type"
        
        # Add similarity search and return fields
        query += """
        WITH f, vector.similarity.cosine(f.embedding_1, $embedding) AS similarity
        WHERE similarity > $threshold
        RETURN f.id AS id, f.name AS name, f.description AS description, 
            f.data_type AS data_type, f.required AS required,
            f.system_defined AS system_defined, similarity
        ORDER BY similarity DESC
        LIMIT 1
        """
        
        # Prepare parameters
        params = {
            "embedding": embedding,
            "threshold": context.get("threshold", 0.7)  # Default threshold
        }
        
        # Add data_type if provided
        if 'data_type' in context:
            params["data_type"] = context['data_type']
        
        # Run the query
        results = self.neo4j.run_query(query, params)
        
        # If no results with good similarity, return None
        if not results:
            return None
        
        # Return the best matching field
        return {
            "id": results[0]["id"],
            "name": results[0]["name"],
            "description": results[0]["description"],
            "data_type": results[0]["data_type"],
            "required": results[0]["required"],
            "system_defined": results[0]["system_defined"],
            "similarity_score": results[0]["similarity"]
        }
        
    def find_matching_section_template_fields(self, template_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find TemplateFields connected to a specific template that best match the input context
        
        Args:
            template_id: ID of the SectionTemplate to search within
            context: Dictionary containing information to match against fields
                - should contain at least 'text' or 'name' for embedding generation
                - can contain 'data_type' to restrict search
                - can contain 'limit' to control number of results returned
        
        Returns:
            List of dictionaries containing field data, sorted by similarity
        """
        # Create embedding from context
        embedding_text = context.get('text', '')
        if not embedding_text:
            # Use name if text not provided
            embedding_text = context.get('name', '')
            
            # Add description if available
            if 'description' in context:
                embedding_text += f" {context['description']}"
                
            # Add data type if available
            if 'data_type' in context:
                embedding_text += f" {context['data_type']}"
        
        # Validate we have something to search with
        if not embedding_text:
            return []
        
        # Generate embedding
        embedding = self.neo4j.get_embedding(embedding_text)
        
        # Query to find template fields connected to the specified template
        query = """
        MATCH (t:SectionTemplate)-[r:HAS_FIELD]->(f:TemplateField)
        WHERE t.id = $template_id AND f.embedding_1 IS NOT NULL
        """
        
        # Add data type filter if provided
        if 'data_type' in context:
            query += " AND f.data_type = $data_type"
        
        # Add similarity search and return fields with relationship properties
        query += """
        WITH f, r, vector.similarity.cosine(f.embedding_1, $embedding) AS similarity
        WHERE similarity > $threshold
        RETURN f.id AS id, 
            f.name AS name, 
            f.description AS description, 
            f.data_type AS data_type, 
            f.required AS required,
            f.system_defined AS system_defined, 
            r.name AS relationship_name,
            r.required AS relationship_required,
            r.order AS display_order,
            similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        # Prepare parameters
        params = {
            "template_id": template_id,
            "embedding": embedding,
            "threshold": context.get("threshold", 0.6),  # Default threshold
            "limit": context.get("limit", 5)  # Default limit
        }
        
        # Add data_type if provided
        if 'data_type' in context:
            params["data_type"] = context['data_type']
        
        # Run the query
        results = self.neo4j.run_query(query, params)
        
        # If no results with good similarity, return empty list
        if not results:
            return []
        
        # Process and return the matching fields
        matching_fields = []
        for result in results:
            matching_fields.append({
                "id": result["id"],
                "name": result["name"],
                "description": result["description"],
                "data_type": result["data_type"],
                "required": result["required"],
                "system_defined": result["system_defined"],
                "relationship_name": result["relationship_name"],
                "relationship_required": result["relationship_required"],
                "display_order": result["display_order"],
                "similarity_score": result["similarity"]
            })
        
        return matching_fields

    def search_content(self, search_text: str, similarity_threshold: float = 0.65) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for both templates and fields matching search text
        
        Args:
            search_text: Text to search for
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary with 'templates' and 'fields' lists
        """
        # Generate embedding for the query
        embedding = self.neo4j.get_embedding(search_text)
        
        # Combined query to search all vector-indexed content
        combined_query = """
        // Search templates
        MATCH (t:SectionTemplate)
        WHERE t.embedding IS NOT NULL
        WITH t, vector.similarity.cosine(t.embedding, $query_embedding) AS similarity
        WHERE similarity > $threshold
        
        // Collect template results
        WITH collect({
            node_type: 'template',
            id: t.id,
            name: t.name,
            description: t.description,
            similarity: similarity
        }) AS template_results
        
        // Search fields
        MATCH (f:TemplateField)
        WHERE f.embedding IS NOT NULL
        WITH template_results, f, vector.similarity.cosine(f.embedding, $query_embedding) AS similarity
        WHERE similarity > $threshold
        
        // Collect field results
        WITH template_results, collect({
            node_type: 'field',
            id: f.id,
            name: f.name,
            description: f.description,
            data_type: f.data_type,
            similarity: similarity
        }) AS field_results
        
        // Return combined results
        RETURN template_results, field_results
        """
        
        params = {
            "query_embedding": embedding,
            "threshold": similarity_threshold
        }
        
        results = self.neo4j.run_query(combined_query, params)
        
        if not results:
            return {"templates": [], "fields": []}
            
        return {
            "templates": results[0].get("template_results", []),
            "fields": results[0].get("field_results", [])
        }
        
    def match_template_fields(
        self,
        template_fields: List[Dict[str, Any]],
        keyword_dict: Dict[str, Any],
        similarity_threshold: float = 0.6,
        data_type_bonus: float = 0.2,
    ) -> List[Tuple[str, Any, List[Dict[str, Any]]]]:
        """
        Match template fields to keywords and return tuples of (keyword_key, value, list_of_matching_fields).
        Always returns a list of template fields as the third item in each tuple.
        
        Args:
            template_fields: List of template field dictionaries
            keyword_dict: Dictionary containing extracted medical keywords
            similarity_threshold: Minimum similarity score for matching
            data_type_bonus: Additional score for data type compatibility
            max_matches_per_value: Maximum number of field matches per keyword value
            
        Returns:
            List of tuples (keyword_key, keyword_value, [template_fields])
        """
        if not template_fields or not keyword_dict:
            return []
            
        neo4j_session = self.neo4j
        all_matches = []
        
        try:
            # Define keyword types we want to match
            keyword_keys = [
                "term", "temporal", "locations", "quantities", 
                "negations", "modifiers"
            ]
            
            # Data type mappings for different keyword types
            type_mappings = {
                "temporal": ["date", "datetime", "time"],
                "locations": ["location", "string"],
                "quantities": ["number", "float", "integer"],
                "modifiers": ["string", "enum", "text"],
                "term": ["string", "enum", "text"],
                "negations": ["boolean", "string"]
            }
            
            # Track matches by keyword key and value
            matches_by_key_value = {}
            
            # Process each keyword type
            for key in keyword_keys:
                if key not in keyword_dict or not keyword_dict[key]:
                    continue
                    
                values = keyword_dict[key]
                if not isinstance(values, list):
                    values = [values]
                
                max_field_num = len(values)
                for value in values:
                    if not value:  # Skip empty values
                        continue
                    
                    # Create embedding for this keyword value
                    value_text = f"{key}: {value}"
                    value_embedding = neo4j_session.get_embedding(value_text)
                    
                    # Dictionary to store scores for this value
                    value_matches = []
                    
                    for field in template_fields:
                        field_type = field.get("data_type", "string").lower()
                        field_name = field.get("name", "")
                        field_desc = field.get("description", "")
                        
                        # Check data type compatibility
                        is_compatible = field_type in type_mappings.get(key, ["string"])
                        
                        # Generate field embedding
                        field_text = f"{field_name} {field_desc}"
                        field_embedding = neo4j_session.get_embedding(field_text)
                        
                        # Calculate similarity
                        similarity = cosine_similarity(value_embedding, field_embedding)
                        
                        # Add bonus for compatible data types
                        if is_compatible:
                            similarity += data_type_bonus
                            
                        # Add bonus for name matches
                        if isinstance(value, str) and (
                            value.lower() in field_name.lower() or 
                            key.lower() in field_name.lower()
                        ):
                            similarity += 0.15
                        
                        # Store match if above threshold
                        if similarity >= similarity_threshold:
                            # Create a clean copy of the template field
                            field_copy = {
                                "id": field.get("id", ""),
                                "name": field.get("name", ""),
                                "data_type": field.get("data_type", "string"),
                                "similarity_score": similarity
                            }
                            value_matches.append(field_copy)
                    
                    # Sort matches by similarity score
                    value_matches.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
                    
                    # Take top matches
                    top_matches = value_matches[:max_field_num]
                    
                    # Store matches for this key-value pair if any exist
                    if top_matches:
                        matches_by_key_value[(key, str(value))] = top_matches
            
            # If no matches for a keyword type, add best match even if below threshold
            matched_keys = {k for k, _ in matches_by_key_value.keys()}
            unmatched_keys = set(k for k in keyword_keys if k in keyword_dict) - matched_keys
            
            for key in unmatched_keys:
                if not keyword_dict.get(key):
                    continue
                    
                values = keyword_dict[key]
                if not isinstance(values, list):
                    values = [values]
                
                if not values or not values[0]:
                    continue
                
                max_field_num = len(values)
                value = values[0]
                
                # Find best fields for this keyword type
                best_fields = []
                
                for field in template_fields:
                    field_type = field.get("data_type", "string").lower()
                    field_name = field.get("name", "")
                    
                    # Calculate a base score
                    if field_type in type_mappings.get(key, ["string"]):
                        base_score = 0.3
                    else:
                        base_score = 0.1
                        
                    # Name match bonus
                    if key.lower() in field_name.lower():
                        base_score += 0.2
                    
                    if base_score > 0:
                        # Create a clean copy of the template field
                        field_copy = {
                            "id": field.get("id", ""),
                            "name": field.get("name", ""),
                            "data_type": field.get("data_type", "string"),
                            "similarity_score": base_score
                        }
                        best_fields.append(field_copy)
                
                # Sort and take top matches
                if best_fields:
                    best_fields.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
                    matches_by_key_value[(key, str(value))] = best_fields[:max_field_num]
            
            # Convert matches dictionary to list of tuples
            for (key, value_str), fields in matches_by_key_value.items():
                # Convert value back to original type if possible
                try:
                    # Handle potential numeric values
                    if value_str.isdigit():
                        value = int(value_str)
                    else:
                        try:
                            value = float(value_str)
                        except ValueError:
                            value = value_str
                except:
                    value = value_str
                    
                all_matches.append((key, value, fields))
            
            # Sort by highest similarity score in each field group
            all_matches.sort(
                key=lambda x: max([f.get("similarity_score", 0) for f in x[2]]), 
                reverse=True
            )
            
            return all_matches
            
        except Exception as e:
            logger.error(f"Error matching template fields: {str(e)}")
            return []