from neo4j import GraphDatabase
import logging
import os
from db_app.neo4j.neo4j_connection_manager import Neo4jConnectionManager
from db_app.neo4j.neo4j_manager import TemplateManager
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

logger = logging.getLogger(__name__)

class Neo4jInitializer:
    """
    Class for initializing Neo4j database with template structure
    """
    
    def __init__(self):
        """
        Initialize with a connection manager
        """
        self.neo4j_connection = Neo4jConnectionManager(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
        self.template_manager = TemplateManager(self.neo4j_connection)
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.embedding_dimension = 384
        
    def initialize(self):
        logger.info(f"Neo4j Connection Status: {self.neo4j_connection.get_connection_status()}")
        self.setup_database()

    def create_constraints_and_indexes(self):
        """Create necessary constraints and vector indexes"""
        constraints = [
            "CREATE CONSTRAINT section_template_id IF NOT EXISTS FOR (t:SectionTemplate) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT template_field_id IF NOT EXISTS FOR (f:TemplateField) REQUIRE f.id IS UNIQUE"
        ]
        
        # Vector indexes for embeddings-based similarity search
        vector_indexes = [
            f"""
            CREATE VECTOR INDEX section_template_embedding IF NOT EXISTS 
            FOR (t:SectionTemplate) 
            ON t.embedding_1
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: "cosine"
                }}
            }}
            """,
            f"""
            CREATE VECTOR INDEX template_field_embedding IF NOT EXISTS 
            FOR (f:TemplateField) 
            ON f.embedding_1
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimension},
                    `vector.similarity_function`: "cosine"
                }}
            }}
            """,
        ]
        
        # Create constraints
        for constraint in constraints:
            try:
                self.neo4j_connection.run_query(constraint)
                logger.info(f"Created constraint: {constraint}")
            except Exception as e:
                logger.error(f"Error creating constraint: {str(e)}")
                
        # Create vector indexes
        for index in vector_indexes:
            try:
                self.neo4j_connection.run_query(index)
                logger.info(f"Created vector index")
            except Exception as e:
                logger.error(f"Error creating vector index: {str(e)}")

    def setup_section_templates(self):
        """Set up basic SOAP templates with vector embeddings"""
        templates = [
            {
                "id": "base-section",
                "name": "Base Section",
                "description": "Base template for all clinical note sections",
                "created_by": "system",
                "system_defined": True
            },
            {
                "id": "subjective",
                "name": "Subjective",
                "description": "Patient-reported information and history",
                "created_by": "system",
                "system_defined": True,
                "extends": "base-section"
            },
            {
                "id": "objective",
                "name": "Objective",
                "description": "Observable findings and measurements",
                "created_by": "system",
                "system_defined": True,
                "extends": "base-section"
            },
            {
                "id": "assessment",
                "name": "Assessment",
                "description": "Clinician assessment and diagnosis",
                "created_by": "system",
                "system_defined": True,
                "extends": "base-section"
            },
            {
                "id": "plan",
                "name": "Plan",
                "description": "Treatment plan and next steps",
                "created_by": "system",
                "system_defined": True,
                "extends": "base-section"
            }
        ]
        
        # Create template nodes with vector embeddings
        create_template_query = """
        MERGE (t:SectionTemplate {id: $id})
        ON CREATE SET 
            t.name = $name,
            t.description = $description,
            t.embedding_1 = $embedding_1,
            t.created_at = timestamp(),
            t.created_by = $created_by,
            t.system_defined = $system_defined,
            t.version = '1.0'
        """
        
        # Create extension relationships
        extend_template_query = """
        MATCH (child:SectionTemplate {id: $child_id})
        MATCH (parent:SectionTemplate {id: $parent_id})
        MERGE (child)-[:EXTENDS]->(parent)
        """
        
        for template in templates:
            # Copy the template dict to avoid modifying the original
            template_params = dict(template)
            extends_id = template_params.pop("extends", None)
            
            # Generate embedding from combined name and description for better semantic matching
            embedding_text = f"{template_params['name']} {template_params['description']}"
            embedding = self.model.encode(embedding_text).tolist()
            template_params["embedding_1"] = embedding
            
            # Create the template node
            self.neo4j_connection.run_query(create_template_query, template_params)
            logger.info(f"Created template: {template_params['id']} with embedding")
            
            # Create extension relationship if needed
            if extends_id:
                self.neo4j_connection.run_query(extend_template_query, {
                    "child_id": template_params["id"],
                    "parent_id": extends_id
                })
                logger.info(f"Created EXTENDS relationship: {template_params['id']} -> {extends_id}")
    
    def setup_fields(self):
        """Set up basic field types with vector embeddings"""
        fields = [
            ### BASE FIELD
            {
                "id": "base-field",
                "name": "Base Field",
                "description": "Root field type for all template fields",
                "data_type": "any",
                "required": False,
                "system_defined": True
            },
            # STRING type
            {
                "id": "base-string-field",
                "name": "String Field",
                "description": "Short text field for single-line inputs",
                "data_type": "string",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            # NUMBER type
            {
                "id": "base-number-field",
                "name": "Number Field",
                "description": "Generic number field (integer or decimal)",
                "data_type": "number",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            
            # INTEGER type
            {
                "id": "base-integer-field",
                "name": "Integer Field",
                "description": "Whole number field without decimals",
                "data_type": "integer",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            
            # FLOAT type
            {
                "id": "base-float-field",
                "name": "Float Field",
                "description": "Decimal number field",
                "data_type": "float",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            
            # BOOLEAN type
            {
                "id": "base-boolean-field",
                "name": "Boolean Field",
                "description": "True/False toggle field",
                "data_type": "boolean",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            
            # DATE type
            {
                "id": "base-date-field",
                "name": "Date Field",
                "description": "Date selector (without time)",
                "data_type": "date",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            
            # TIME type
            {
                "id": "base-time-field",
                "name": "Time Field",
                "description": "Time selector (without date)",
                "data_type": "time",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            
            # DATETIME type
            {
                "id": "base-datetime-field",
                "name": "DateTime Field",
                "description": "Combined date and time selector",
                "data_type": "datetime",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            # CODE type
            {
                "id": "base-code-field",
                "name": "Code Field",
                "description": "Field for medical or classification codes",
                "data_type": "code",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            ### CODES
            {
                "id": "icd-10-code-field",
                "name": "ICD-10 Code Field",
                "description": "ICD-10 Code",
                "data_type": "code",
                "required": False,
                "system_defined": True,
                "extends": "code-field"
            },
            
            ### DIAGNOSIS FIELDS
            {
                "id": "diagnosis-field",
                "name": "Diagnosis Field",
                "description": "Diagnosis made by the medical provider",
                "data_type": "string",
                "required": False,
                "system_defined": True,
                "extends": "base-string-field"
            },
            {
                "id": "symptom-field",
                "name": "Symptom Field",
                "description": "Any disease-related symptoms",
                "data_type": "string",
                "required": False,
                "system_defined": True,
                "extends": "base-string-field"
            },
            {
                "id": "patient-symptom-field",
                "name": "Patient-reported Symptom Field",
                "description": "Any symptom reported by the patient",
                "data_type": "string",
                "required": False,
                "system_defined": True,
                "extends": "symptom-field"
            },
            {
                "id": "observed-symptom-field",
                "name": "Observed Symptom Field",
                "description": "Symptom observed by the medical provider",
                "data_type": "string",
                "required": False,
                "system_defined": True,
                "extends": "symptom-field"
            },
            
            ### PLAN FIELDS
            {
                "id": "base-plan-field",
                "name": "Base Plan Field",
                "description": "Any future plans for either the patient or the medical provider",
                "data_type": "any",
                "required": False,
                "system_defined": True,
                "extends": "base-field"
            },
            {
                "id": "treatment-plan-field",
                "name": "Treatment Plan Field",
                "description": "Any treatment plan for the patient",
                "data_type": "any",
                "required": False,
                "system_defined": True,
                "extends": "base-plan-field"
            }
        ]
        
        # Create field nodes with vector embeddings
        create_field_query = """
        MERGE (f:TemplateField {id: $id})
        ON CREATE SET 
            f.name = $name,
            f.description = $description,
            f.data_type = $data_type,
            f.required = $required,
            f.system_defined = $system_defined,
            f.embedding_1 = $embedding_1,
            f.created_at = timestamp()
        """
        
        # Create extension relationships
        extend_field_query = """
        MATCH (child:TemplateField {id: $child_id})
        MATCH (parent:TemplateField {id: $parent_id})
        MERGE (child)-[:EXTENDS]->(parent)
        """
        
        for field in fields:
            # Copy the field dict to avoid modifying the original
            field_params = dict(field)
            extends_id = field_params.pop("extends", None)
            
            # Generate embedding from combined name and description
            embedding_text = f"{field_params['name']} {field_params['description']} {field_params['data_type']}"
            embedding = self.model.encode(embedding_text).tolist()
            field_params["embedding_1"] = embedding
            
            # Create the field node
            self.neo4j_connection.run_query(create_field_query, field_params)
            logger.info(f"Created field: {field_params['id']} with embedding")
            
            # Create extension relationship if needed
            if extends_id:
                self.neo4j_connection.run_query(extend_field_query, {
                    "child_id": field_params["id"],
                    "parent_id": extends_id
                })
                logger.info(f"Created EXTENDS relationship: {field_params['id']} -> {extends_id}")
    
    def assign_fields_to_templates(self):
        """Assign fields to templates"""
        field_assignments = [
            # Subjective section fields
            {"template_id": "subjective", "field_id": "patient-symptom-field", "field_name": "patient_reported_symptoms", "required": True},
            
            # Objective section fields
            {"template_id": "objective", "field_id": "observed-symptom-field", "field_name": "observed_symptoms", "required": True},
            
            # Assessment section fields
            {"template_id": "assessment", "field_id": "diagnosis-field", "field_name": "diagnosis", "required": True},
            {"template_id": "assessment", "field_id": "code-field", "field_name": "diagnosis_code", "required": False},
            
            # Plan section fields
            {"template_id": "plan", "field_id": "treatment-plan-field", "field_name": "treatment", "required": True},
        ]
        
        assign_field_query = """
        MATCH (t:SectionTemplate {id: $template_id})
        MATCH (f:TemplateField {id: $field_id})
        MERGE (t)-[r:HAS_FIELD {name: $field_name}]->(f)
        SET r.required = $required,
            r.description = $description,
            r.order = $order
        """
        
        for i, assignment in enumerate(field_assignments):
            assignment_data = {
                "template_id": assignment["template_id"],
                "field_id": assignment["field_id"],
                "field_name": assignment["field_name"],
                "required": assignment["required"],
                "description": assignment.get("description", f"Field {assignment['field_name']} for {assignment['template_id']}"),
                "order": assignment.get("order", i)
            }
            
            self.neo4j_connection.run_query(assign_field_query, assignment_data)
            logger.info(f"Assigned {assignment['field_id']} as '{assignment['field_name']}' to {assignment['template_id']}")
    
    def setup_database(self):
        """Run the complete database setup process"""
        try:
            logger.info("Starting Neo4j template setup with vector search support...")
            
            # Create constraints and vector indexes
            self.create_constraints_and_indexes()
            
            # Set up templates with vector embeddings
            self.setup_section_templates()
            
            # Set up field types with vector embeddings
            self.setup_fields()
            
            # Assign fields to templates
            self.assign_fields_to_templates()
            
            # Test vector search functionality
            self.test_vector_search()
            
            logger.info("Neo4j template setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Neo4j templates: {str(e)}")
            return False
    
    def test_vector_search(self):
        """Test vector search functionality to ensure it's working properly"""
        try:
            # Test template vector search
            logger.info("Testing vector search using all-minilm-l6-v2")
            
            # Generate a test query embedding
            test_query = "information about diagnosis code and ICD standards"
            query_embedding = self.model.encode(test_query).tolist()
            
            # Run the vector search
            template_query = """
            MATCH (t:SectionTemplate)
            WHERE t.embedding_1 IS NOT NULL
            WITH t, vector.similarity.cosine(t.embedding_1, $query_embedding) AS similarity
            WHERE similarity > 0.5
            RETURN t.id, t.name, t.description, similarity
            ORDER BY similarity DESC
            LIMIT 3
            """
            
            results = self.neo4j_connection.run_query(template_query, {"query_embedding": query_embedding})
            
            if results:
                logger.info(f"Vector search found {len(results)} matching templates:")
                for record in results:
                    logger.info(f"  - {record['t.id']} ({record['t.name']}): similarity = {record['similarity']:.4f}")
            else:
                logger.warning("Vector search didn't find any templates")
                
            # Test field vector search
            logger.info("Testing vector search for template fields...")
            
            # Generate a test query embedding for fields
            field_test_query = "date and time information"
            field_query_embedding = self.model.encode(field_test_query).tolist()
            
            # Run the vector search for fields
            field_query = """
            MATCH (f:TemplateField)
            WHERE f.embedding_1 IS NOT NULL
            WITH f, vector.similarity.cosine(f.embedding_1, $query_embedding) AS similarity
            WHERE similarity > 0.5
            RETURN f.id, f.name, f.description, f.data_type, similarity
            ORDER BY similarity DESC
            LIMIT 3
            """
            
            field_results = self.neo4j_connection.run_query(field_query, {"query_embedding": field_query_embedding})
            
            if field_results:
                logger.info(f"Vector search found {len(field_results)} matching fields:")
                for record in field_results:
                    logger.info(f"  - {record['f.id']} ({record['f.name']}): similarity = {record['similarity']:.4f}")
            else:
                logger.warning("Vector search didn't find any fields")
                
        except Exception as e:
            logger.error(f"Error testing vector search: {str(e)}")