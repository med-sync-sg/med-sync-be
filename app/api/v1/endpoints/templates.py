from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Any, Optional
from app.services.template_service import TemplateService
from app.schemas.template import TemplateRead, TemplateCreate, TemplateWithFields, SearchResult, FieldAssignment, FieldCreate, FieldRead
from app.db.neo4j_session import get_neo4j_session, Neo4jSession

router = APIRouter()

# Initialize services
template_service = TemplateService()

# Endpoints
@router.get("/", response_model=List[TemplateRead])
async def get_all_templates(
    include_system: bool = Query(True, description="Include system-defined templates"),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Get all templates"""
    return template_service.get_all_templates(include_system=include_system)

@router.get("/search", response_model=SearchResult)
async def search_templates_and_fields(
    q: str = Query(..., description="Search query text"),
    threshold: float = Query(0.65, description="Minimum similarity threshold"),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Search for templates and fields by semantic similarity"""
    results = template_service.search_content(q, similarity_threshold=threshold)
    return results

@router.get("/{template_id}", response_model=TemplateWithFields)
async def get_template(
    template_id: str, 
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Get a specific template with its fields"""
    template = template_service.get_template_with_fields(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template with ID {template_id} not found"
        )
    return template

@router.post("/", response_model=TemplateRead, status_code=status.HTTP_201_CREATED)
async def create_template(
    template: TemplateCreate,
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Create a new template"""
    template_id = template_service.create_template(template.dict())
    if not template_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create template"
        )
    
    # Return the created template
    created_template = template_service.get_template_with_fields(template_id)
    if not created_template:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Template created but could not be retrieved"
        )
    
    return created_template

@router.post("/{template_id}/fields", status_code=status.HTTP_201_CREATED)
async def add_field_to_template(
    template_id: str,
    field: FieldAssignment,
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Add a field to a template"""
    success = template_service.add_field_to_template(template_id, field.dict())
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add field to template"
        )
    
    return {"message": f"Field {field.field_id} added to template {template_id}"}

@router.post("/fields", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED)
async def create_field(
    field: FieldCreate,
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Create a new field or find an existing one"""
    field_id = template_service.find_or_create_field(field.dict())
    return {"id": field_id}

@router.get("/similar/{text}", response_model=List[TemplateRead])
async def find_similar_templates(
    text: str,
    threshold: float = Query(0.65, description="Minimum similarity threshold"),
    limit: int = Query(5, description="Maximum number of results"),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Find templates similar to the provided text"""
    templates = template_service.find_templates_by_text(text, similarity_threshold=threshold, limit=limit)
    return templates