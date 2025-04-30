import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class TemplateManager:
    """
    Manager for section templates and fields in Neo4j
    """
    
    def __init__(self, connection_manager):
        """
        Initialize with a connection manager
        
        Args:
            connection_manager: Neo4j connection manager
        """
        self.connection = connection_manager
        # Load the sentence transformer model for generating embeddings
        self.model = SentenceTransformer("all-minilm-l6-v2")
        
    def find_template_by_description(self, description: str, 
                                    similarity_threshold: float = 0.7, 
                                    limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find section templates by semantic similarity to description
        
        Args:
            description: Description text to match
            similarity_threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
            
        Returns:
            List of matching templates with similarity scores
        """
        # Generate embedding for the search query
        query_embedding = self.model.encode(description).tolist()
        
        # Use the vector search method from the connection manager
        results = self.connection.run_vector_search(
            label="SectionTemplate",
            embedding_field="embedding",
            query_embedding=query_embedding,
            similarity_threshold=similarity_threshold,
            limit=limit
        )
        
        # Format the results
        templates = []
        for record in results:
            template_node = record.get('n', {})
            similarity = record.get('similarity', 0)
            
            template = {
                "id": template_node.get('id'),
                "name": template_node.get('name'),
                "description": template_node.get('description'),
                "similarity_score": similarity
            }
            templates.append(template)
            
        return templates
    
    def find_field_by_description(self, description: str,
                                 similarity_threshold: float = 0.7,
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find template fields by semantic similarity to description
        
        Args:
            description: Description text to match
            similarity_threshold: Minimum similarity score (0-1)
            limit: Maximum number of results
            
        Returns:
            List of matching fields with similarity scores
        """
        # Generate embedding for the search query
        query_embedding = self.model.encode(description).tolist()
        
        # Use the vector search method from the connection manager
        results = self.connection.run_vector_search(
            label="TemplateField",
            embedding_field="embedding",
            query_embedding=query_embedding,
            similarity_threshold=similarity_threshold,
            limit=limit
        )
        
        # Format the results
        fields = []
        for record in results:
            field_node = record.get('n', {})
            similarity = record.get('similarity', 0)
            
            field = {
                "id": field_node.get('id'),
                "name": field_node.get('name'),
                "description": field_node.get('description'),
                "data_type": field_node.get('data_type'),
                "similarity_score": similarity
            }
            fields.append(field)
            
        return fields
    
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
        RETURN t, collect({field: f, relationship: r}) as fields
        """
        
        results = self.connection.run_query(query, {"template_id": template_id})
        
        if not results:
            return None
            
        template_node = results[0].get('t', {})
        fields_data = results[0].get('fields', [])
        
        # Build template object
        template = {
            "id": template_node.get('id'),
            "name": template_node.get('name'),
            "description": template_node.get('description'),
            "system_defined": template_node.get('system_defined', False),
            "created_at": template_node.get('created_at'),
            "created_by": template_node.get('created_by'),
            "version": template_node.get('version', '1.0'),
            "fields": []
        }
        
        # Add fields with relationship properties
        for field_data in fields_data:
            field_node = field_data.get('field', {})
            rel = field_data.get('relationship', {})
            
            if field_node:
                field = {
                    "id": field_node.get('id'),
                    "name": field_node.get('name'),
                    "description": field_node.get('description'),
                    "data_type": field_node.get('data_type'),
                    "field_name": rel.get('name'),
                    "required": rel.get('required', False),
                    "order": rel.get('order', 0)
                }
                template["fields"].append(field)
                
        # Sort fields by order
        template["fields"].sort(key=lambda f: f.get("order", 0))
        
        return template
    
    def create_template(self, template_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new template
        
        Args:
            template_data: Template data
            
        Returns:
            ID of the created template or None if failed
        """
        # Generate embedding if not provided
        if "embedding" not in template_data:
            embedding_text = f"{template_data.get('name', '')} {template_data.get('description', '')}"
            template_data["embedding"] = self.model.encode(embedding_text).tolist()
        
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
            "embedding": template_data.get("embedding"),
            "created_by": template_data.get("created_by", "user"),
            "system_defined": template_data.get("system_defined", False),
            "version": template_data.get("version", "1.0")
        }
        
        results = self.connection.run_query(query, template_params)
        
        if results and "id" in results[0]:
            return results[0]["id"]
        return None
    
    def add_field_to_template(self, template_id: str, field_id: str, 
                             field_name: str, required: bool = False,
                             order: int = 0) -> bool:
        """
        Add a field to a template
        
        Args:
            template_id: ID of the template
            field_id: ID of the field
            field_name: Name for this field in the template
            required: Whether the field is required
            order: Display order of the field
            
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
            "field_id": field_id,
            "field_name": field_name,
            "required": required,
            "order": order
        }
        
        results = self.connection.run_query(query, params)
        
        if results and results[0].get("count", 0) > 0:
            return True
        return False
    
    def search_templates_and_fields(self, query_text: str, 
                                   similarity_threshold: float = 0.6) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for both templates and fields matching query text
        
        Args:
            query_text: Text to search for
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary with 'templates' and 'fields' lists
        """
        # Generate embedding for the query
        query_embedding = self.model.encode(query_text).tolist()
        
        # Search for templates
        template_query = """
        MATCH (t:SectionTemplate)
        WHERE t.embedding IS NOT NULL
        WITH t, vector.similarity.cosine(t.embedding, $query_embedding) AS similarity
        WHERE similarity > $threshold
        RETURN t.id as id, t.name as name, t.description as description, similarity
        ORDER BY similarity DESC
        LIMIT 5
        """
        
        # Search for fields
        field_query = """
        MATCH (f:TemplateField)
        WHERE f.embedding IS NOT NULL
        WITH f, vector.similarity.cosine(f.embedding, $query_embedding) AS similarity
        WHERE similarity > $threshold
        RETURN f.id as id, f.name as name, f.description as description, 
               f.data_type as data_type, similarity
        ORDER BY similarity DESC
        LIMIT 5
        """
        
        params = {
            "query_embedding": query_embedding,
            "threshold": similarity_threshold
        }
        
        template_results = self.connection.run_query(template_query, params)
        field_results = self.connection.run_query(field_query, params)
        
        return {
            "templates": template_results,
            "fields": field_results
        }
    
    def create_field(self, field_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new template field
        
        Args:
            field_data: Field data
            
        Returns:
            ID of the created field or None if failed
        """
        # Generate embedding if not provided
        if "embedding" not in field_data:
            embedding_text = f"{field_data.get('name', '')} {field_data.get('description', '')} {field_data.get('data_type', '')}"
            field_data["embedding"] = self.model.encode(embedding_text).tolist()
        
        query = """
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
        
        # Set default values
        field_params = {
            "id": field_data.get("id"),
            "name": field_data.get("name"),
            "description": field_data.get("description", ""),
            "data_type": field_data.get("data_type", "string"),
            "required": field_data.get("required", False),
            "embedding": field_data.get("embedding"),
            "system_defined": field_data.get("system_defined", False)
        }
        
        results = self.connection.run_query(query, field_params)
        
        if results and "id" in results[0]:
            return results[0]["id"]
        return None
    
    def get_similar_content(self, text: str, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Find any content (templates or fields) similar to the provided text
        
        Args:
            text: Text to find similar content for
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary with templates and fields that match
        """
        # Generate embedding for the text
        query_embedding = self.model.encode(text).tolist()
        
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
            "query_embedding": query_embedding,
            "threshold": similarity_threshold
        }
        
        results = self.connection.run_query(combined_query, params)
        
        if not results:
            return {"templates": [], "fields": []}
            
        return {
            "templates": results[0].get("template_results", []),
            "fields": results[0].get("field_results", [])
        }