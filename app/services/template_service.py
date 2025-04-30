import logging
from typing import List, Dict, Any, Optional, Union
from app.db.neo4j_session import neo4j_session
import uuid

logger = logging.getLogger(__name__)

class TemplateService:
    """
    Service for managing section templates and fields
    """
    
    def __init__(self):
        """Initialize the template service"""
        self.neo4j = neo4j_session
    
    def find_templates_by_text(self, search_text: str, similarity_threshold: float = 0.65, limit: int = 5) -> List[Dict[str, Any]]:
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
            embedding_field="embedding",
            vector=embedding,
            similarity_threshold=similarity_threshold,
            limit=limit
        )
        
        # Format results
        templates = []
        for result in results:
            node = result.get("n", {})
            
            templates.append({
                "id": node.get("id"),
                "name": node.get("name"),
                "description": node.get("description"),
                "system_defined": node.get("system_defined", False),
                "version": node.get("version", "1.0"),
                "similarity_score": result.get("similarity", 0)
            })
            
        return templates
    
    def get_template_with_fields(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a template with all its fields
        
        Args:
            template_id: ID of the template
            
        Returns:
            Template with fields or None if not found
        """
        query = """
        MATCH (t:SectionTemplate {id: $template_id})
        OPTIONAL MATCH (t)-[r:HAS_FIELD]->(f:TemplateField)
        RETURN t as template, collect({field: f, relationship: r}) as fields
        """
        
        results = self.neo4j.run_query(query, {"template_id": template_id})
        
        if not results or "template" not in results[0]:
            return None
            
        template_node = results[0].get("template", {})
        fields_data = results[0].get("fields", [])
        
        # Build template object
        template = {
            "id": template_node.get("id"),
            "name": template_node.get("name"),
            "description": template_node.get("description"),
            "system_defined": template_node.get("system_defined", False),
            "created_at": template_node.get("created_at"),
            "created_by": template_node.get("created_by"),
            "version": template_node.get("version", "1.0"),
            "fields": []
        }
        
        # Add fields with relationship properties
        for field_data in fields_data:
            field_node = field_data.get("field", {})
            rel = field_data.get("relationship", {})
            
            if field_node:
                field = {
                    "id": field_node.get("id"),
                    "name": field_node.get("name"),
                    "description": field_node.get("description"),
                    "data_type": field_node.get("data_type"),
                    "field_name": rel.get("name"),
                    "required": rel.get("required", False),
                    "order": rel.get("order", 0)
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
            embedding: $embedding,
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
            "embedding": embedding,
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
            embedding: $embedding,
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
            "embedding": embedding,
            "system_defined": field_data.get("system_defined", False)
        }
        
        create_results = self.neo4j.run_query(create_query, create_params)
        
        if create_results and "id" in create_results[0]:
            return create_results[0]["id"]
        
        # If creation failed, return the original ID
        return field_data.get("id")
    
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