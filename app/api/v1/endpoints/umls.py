from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Any, Optional
from app.db.neo4j_session import get_neo4j_session, Neo4jSession
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

router = APIRouter()

# Pydantic models for request/response validation
class UMLSConcept(BaseModel):
    cui: str
    name: str
    semantic_type: str
    definition: Optional[str] = None
    
class UMLSRelationship(BaseModel):
    source_cui: str
    target_cui: str
    relationship_type: str
    
class UMLSSearchResult(BaseModel):
    concepts: List[UMLSConcept] = []
    
class UMLSRelatedTerms(BaseModel):
    related_terms: List[Dict[str, Any]] = []
    source_term: Optional[Dict[str, Any]] = None

# UMLS service functions
def search_umls_concepts(
    neo4j: Neo4jSession,
    search_text: str,
    semantic_types: Optional[List[str]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for UMLS concepts by text
    
    Args:
        neo4j: Neo4j session
        search_text: Text to search for
        semantic_types: List of semantic types to filter by
        limit: Maximum number of results
        
    Returns:
        List of matching concepts
    """
    # Generate embedding for semantic search
    embedding = neo4j.get_embedding(search_text)
    
    # Build query
    base_query = """
    MATCH (c:UMLS:Concept)
    WHERE c.embedding IS NOT NULL
    """
    
    # Add semantic type filter if provided
    if semantic_types:
        base_query += """
        MATCH (c)-[:UMLS_HAS_STY]->(s:UMLS:SemanticType)
        WHERE s.tui IN $semantic_types
        """
    
    # Calculate similarity and return results
    query = base_query + """
    WITH c, vector.similarity.cosine(c.embedding, $query_embedding) AS similarity
    WHERE similarity > 0.6
    RETURN c.cui as cui, c.name as name, similarity,
           collect(DISTINCT s.tui) as tuis, collect(DISTINCT s.name) as semantic_types,
           c.definition as definition
    ORDER BY similarity DESC
    LIMIT $limit
    """
    
    params = {
        "query_embedding": embedding,
        "semantic_types": semantic_types or [],
        "limit": limit
    }
    
    try:
        results = neo4j.run_query(query, params)
        return results
    except Exception as e:
        return []

def get_related_terms(
    neo4j: Neo4jSession,
    cui: str,
    relationship_types: Optional[List[str]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Get terms related to a UMLS concept
    
    Args:
        neo4j: Neo4j session
        cui: CUI of the concept
        relationship_types: Types of relationships to include
        limit: Maximum number of results
        
    Returns:
        Dictionary with source term and related terms
    """
    # Build relationship filter
    rel_filter = ""
    if relationship_types:
        rel_types = [f"'{t}'" for t in relationship_types]
        rel_filter = f"type(r) IN [{', '.join(rel_types)}]"
    
    # Build query
    query = f"""
    MATCH (source:UMLS:Concept {{cui: $cui}})
    OPTIONAL MATCH (source)-[r:UMLS_REL]->(related:UMLS:Concept)
    {f"WHERE {rel_filter}" if rel_filter else ""}
    WITH source, related, type(r) as relationship_type
    OPTIONAL MATCH (related)-[:UMLS_HAS_STY]->(sty:UMLS:SemanticType)
    RETURN source.cui as source_cui, source.name as source_name,
           collect(DISTINCT {{
               cui: related.cui,
               name: related.name,
               relationship: relationship_type,
               semantic_types: collect(DISTINCT sty.name)
           }}) as related_terms
    LIMIT $limit
    """
    
    params = {
        "cui": cui,
        "limit": limit
    }
    
    try:
        results = neo4j.run_query(query, params)
        if not results:
            return {"source_term": None, "related_terms": []}
            
        # Extract source term
        source_term = {
            "cui": results[0].get("source_cui"),
            "name": results[0].get("source_name")
        }
        
        # Extract related terms
        related_terms = results[0].get("related_terms", [])
        
        return {
            "source_term": source_term,
            "related_terms": related_terms
        }
    except Exception as e:
        return {"source_term": None, "related_terms": []}

# Endpoints
@router.get("/search", response_model=UMLSSearchResult)
async def search_concepts(
    q: str = Query(..., description="Search text"),
    semantic_types: Optional[List[str]] = Query(None, description="Semantic types to filter by"),
    limit: int = Query(10, description="Maximum number of results"),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Search for UMLS concepts"""
    results = search_umls_concepts(neo4j, q, semantic_types, limit)
    
    # Format results
    concepts = []
    for result in results:
        # Get the first semantic type if available
        semantic_type = result.get("semantic_types", ["Unknown"])[0] if result.get("semantic_types") else "Unknown"
        
        concept = UMLSConcept(
            cui=result.get("cui"),
            name=result.get("name"),
            semantic_type=semantic_type,
            definition=result.get("definition")
        )
        concepts.append(concept)
    
    return {"concepts": concepts}

@router.get("/concept/{cui}/related", response_model=UMLSRelatedTerms)
async def get_related_concepts(
    cui: str,
    relationship_types: Optional[List[str]] = Query(None, description="Relationship types to include"),
    limit: int = Query(20, description="Maximum number of results"),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Get concepts related to a specific UMLS concept"""
    results = get_related_terms(neo4j, cui, relationship_types, limit)
    return results

@router.get("/umls-semantic-types", response_model=List[Dict[str, str]])
async def get_semantic_types(
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Get all UMLS semantic types"""
    query = """
    MATCH (s:UMLS:SemanticType)
    RETURN s.tui as tui, s.name as name
    ORDER BY s.name
    """
    
    results = neo4j.run_query(query)
    return results