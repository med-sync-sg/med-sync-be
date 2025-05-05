from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Any, Optional
from app.services.section_template_service import SectionTemplateService
from app.schemas.section_template import SectionTemplateRead, SectionTemplateCreate, SectionTemplateWithTemplateFields, SearchResult, FieldValueAssignment, TemplateFieldCreate, TemplateFieldRead
from app.db.neo4j_session import get_neo4j_session, Neo4jSession
from app.models.models import User
from app.utils.auth_utils import get_current_user

router = APIRouter()

# Initialize services
section_template_service = SectionTemplateService()

# Endpoints
@router.get("/", response_model=List[SectionTemplateRead])
async def get_all_templates(
    include_system: bool = Query(True, description="Include system-defined templates"),
    current_user: User = Depends(get_current_user),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Get all templates"""
    return section_template_service.get_all_templates(include_system=include_system)

@router.get("/search", response_model=SearchResult)
async def search_templates_and_fields(
    q: str = Query(..., description="Search query text"),
    threshold: float = Query(0.65, description="Minimum similarity threshold"),
    current_user: User = Depends(get_current_user),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Search for templates and fields by semantic similarity"""
    results = section_template_service.search_content(q, similarity_threshold=threshold)
    return results

@router.get("/{template_id}", response_model=SectionTemplateWithTemplateFields)
async def get_template(
    template_id: str, 
    current_user: User = Depends(get_current_user),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Get a specific template with its fields"""
    template = section_template_service.get_template_with_fields(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template with ID {template_id} not found"
        )
    return template

@router.post("/", response_model=SectionTemplateRead, status_code=status.HTTP_201_CREATED)
async def create_template(
    template: SectionTemplateCreate,
    current_user: User = Depends(get_current_user),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Create a new template"""
    template_id = section_template_service.create_template(template.dict())
    if not template_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create template"
        )
    
    # Return the created template
    created_template = section_template_service.get_template_with_fields(template_id)
    if not created_template:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Template created but could not be retrieved"
        )
    
    return created_template

@router.post("/{template_id}/fields", status_code=status.HTTP_201_CREATED)
async def add_field_to_template(
    template_id: str,
    field: FieldValueAssignment,
    current_user: User = Depends(get_current_user),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Add a field to a template"""
    success = section_template_service.add_field_to_template(template_id, field.dict())
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add field to template"
        )
    
    return {"message": f"Field {field.field_id} added to template {template_id}"}

@router.post("/fields", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED)
async def create_field(
    field: TemplateFieldCreate,
    current_user: User = Depends(get_current_user),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Create a new field or find an existing one"""
    field_id = section_template_service.find_or_create_field(field.dict())
    return {"id": field_id}

@router.get("/similar/{text}", response_model=List[SectionTemplateRead])
async def find_similar_templates(
    text: str,
    threshold: float = Query(0.65, description="Minimum similarity threshold"),
    limit: int = Query(5, description="Maximum number of results"),
    current_user: User = Depends(get_current_user),
    neo4j: Neo4jSession = Depends(get_neo4j_session)
):
    """Find templates similar to the provided text"""
    templates = section_template_service.find_templates_by_text(text, similarity_threshold=threshold, limit=limit)
    return templates