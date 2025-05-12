from app.models.models import Base
from app.models.models import ReportTemplate
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from os import environ
from sqlalchemy.pool import QueuePool
import os

DB_USER = environ.get('DB_USER', "root")
DB_PASSWORD = environ.get('DB_PASSWORD', "medsync!")
DB_HOST = environ.get('DB_HOST', "localhost")
DB_PORT = environ.get('DB_PORT', "9000")
DB_NAME = environ.get('DB_NAME', "medsync_db")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def create_engine_with_proper_pooling():
    """Create SQLAlchemy engine with optimized connection pooling"""
    return create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,  # Adjust based on your workload
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_pre_ping=True  # Check connection validity before using
    )

class DatabaseManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.engine = create_engine_with_proper_pooling()
            cls._instance.session_factory = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=cls._instance.engine
            )
            # Initialize database if needed
            cls._instance._initialize_database()
        return cls._instance
    
    def _initialize_database(self):
        """Initialize database schema and tables"""
        # Only create tables once at startup
        Base.metadata.create_all(bind=self.engine)
        
        # self._initialize_default_templates()
        
    def _initialize_default_templates(self):
        """Add default report templates to the database with HTML content from template files."""
        session = self.session_factory()
        try:
            # Define path to template files
            template_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "services", "report_generation", "default_html"
            )
            
            # Load HTML templates from files
            doctor_template_path = os.path.join(template_dir, "default_doctor.html")
            patient_template_path = os.path.join(template_dir, "default_patient.html")
            
            # Read template files if they exist
            doctor_html_content = ""
            patient_html_content = ""
            
            if os.path.exists(doctor_template_path):
                with open(doctor_template_path, 'r', encoding='utf-8') as f:
                    doctor_html_content = f.read()
                    print(f"Loaded doctor report template from {doctor_template_path}")
            else:
                print(f"Doctor template file not found at {doctor_template_path}")
                
            if os.path.exists(patient_template_path):
                with open(patient_template_path, 'r', encoding='utf-8') as f:
                    patient_html_content = f.read()
                    print(f"Loaded patient report template from {patient_template_path}")
            else:
                print(f"Patient template file not found at {patient_template_path}")
            
            # Check if default doctor template already exists
            existing_doctor = session.query(ReportTemplate).filter_by(
                name='Comprehensive Clinical Documentation',
                is_default=True,
                template_type='doctor'
            ).first()
            
            if not existing_doctor:
                # Create the doctor report template object directly
                doctor_template = ReportTemplate(
                    user_id=1,  # System user ID (adjust as needed)
                    name='Comprehensive Clinical Documentation',
                    description='Standard medical report template with complete SOAP structure for healthcare professionals',
                    template_type='doctor',
                    is_default=True,
                    html_template=doctor_html_content,
                    template_data={
                        "sections": {
                            "SUBJECTIVE": {
                                "section_type": "SUBJECTIVE",
                                "include": True,
                                "order": 0,
                                "title_override": "History of Present Illness",
                                "format_options": {
                                    "show_evidence": True,
                                    "detailed_view": True,
                                    "include_timestamps": True,
                                    "highlight_important": True,
                                    "include_patient_demographics": True,
                                    "show_previous_history": True,
                                    "structured_fields": [
                                        "chief_complaint", 
                                        "symptom_onset", 
                                        "symptom_duration",
                                        "aggravating_factors",
                                        "alleviating_factors",
                                        "associated_symptoms"
                                    ]
                                }
                            },
                            "OBJECTIVE": {
                                "section_type": "OBJECTIVE",
                                "include": True,
                                "order": 1,
                                "title_override": "Physical Examination & Findings",
                                "format_options": {
                                    "show_evidence": True,
                                    "detailed_view": True,
                                    "highlight_abnormal": True,
                                    "include_vital_signs": True,
                                    "show_lab_results": True,
                                    "show_imaging_results": True,
                                    "structured_fields": [
                                        "vital_signs",
                                        "general_appearance",
                                        "heent",
                                        "cardiovascular",
                                        "respiratory",
                                        "gastrointestinal",
                                        "musculoskeletal",
                                        "neurological",
                                        "psychiatric",
                                        "lab_results",
                                        "imaging_results",
                                        "diagnostic_tests"
                                    ]
                                }
                            },
                            "ASSESSMENT": {
                                "section_type": "ASSESSMENT",
                                "include": True,
                                "order": 2,
                                "title_override": "Assessment & Diagnosis",
                                "format_options": {
                                    "show_differential": True,
                                    "include_codes": True,
                                    "highlight_primary": True,
                                    "show_severity": True,
                                    "include_clinical_reasoning": True,
                                    "include_rule_outs": True,
                                    "structured_fields": [
                                        "primary_diagnosis",
                                        "secondary_diagnoses",
                                        "differential_diagnoses",
                                        "problem_list",
                                        "diagnostic_codes",
                                        "clinical_impression",
                                        "risk_assessment"
                                    ]
                                }
                            },
                            "PLAN": {
                                "section_type": "PLAN",
                                "include": True,
                                "order": 3,
                                "title_override": "Treatment Plan & Recommendations",
                                "format_options": {
                                    "detailed_view": True,
                                    "categorize_items": True,
                                    "show_rationale": True,
                                    "include_order_details": True,
                                    "show_dosage_details": True,
                                    "include_cpt_codes": True,
                                    "show_followup_timeline": True,
                                    "structured_fields": [
                                        "medications",
                                        "procedures",
                                        "therapies",
                                        "consultations",
                                        "imaging_orders",
                                        "lab_orders",
                                        "patient_instructions",
                                        "follow_up_plans",
                                        "contingency_plans",
                                        "referrals"
                                    ]
                                }
                            },
                            "OTHER": {
                                "section_type": "OTHER",
                                "include": True,
                                "order": 4,
                                "title_override": "Additional Clinical Information",
                                "format_options": {
                                    "detailed_view": True,
                                    "show_care_team": True,
                                    "include_consent_info": True,
                                    "show_disclaimer": True
                                }
                            }
                        },
                        "template_settings": {
                            "show_header": True,
                            "show_footer": True,
                            "include_encounter_details": True,
                            "show_provider_details": True,
                            "show_facility_info": True,
                            "include_date_time": True,
                            "show_page_numbers": True,
                            "include_signature_line": True
                        }
                    },
                    version="1.0"
                )
                
                session.add(doctor_template)
                session.commit()
                print("Added default doctor's report template to database")
            
            # Check if default patient template already exists
            existing_patient = session.query(ReportTemplate).filter_by(
                name='Patient-Friendly Summary',
                is_default=True,
                template_type='patient'
            ).first()
            
            if not existing_patient:
                # Create the patient report template
                patient_template = ReportTemplate(
                    user_id=1,  # System user ID (adjust as needed)
                    name='Patient-Friendly Summary',
                    description='Simplified medical report template for patient understanding with plain language explanations',
                    template_type='patient',
                    is_default=True,
                    html_template=patient_html_content,
                    template_data={
                        "sections": {
                            "SUBJECTIVE": {
                                "section_type": "SUBJECTIVE",
                                "include": True,
                                "order": 0,
                                "title_override": "Your Symptoms & Concerns",
                                "format_options": {
                                    "patient_friendly": True,
                                    "simplified_language": True,
                                    "highlight_important": True,
                                    "include_explanations": True
                                }
                            },
                            "OBJECTIVE": {
                                "section_type": "OBJECTIVE",
                                "include": True,
                                "order": 1,
                                "title_override": "Examination Findings",
                                "format_options": {
                                    "patient_friendly": True,
                                    "simplified_language": True,
                                    "highlight_important": True,
                                    "include_explanations": True,
                                    "hide_technical": True
                                }
                            },
                            "ASSESSMENT": {
                                "section_type": "ASSESSMENT",
                                "include": True,
                                "order": 2,
                                "title_override": "Your Diagnosis",
                                "format_options": {
                                    "patient_friendly": True,
                                    "simplified_language": True,
                                    "explain_medical_terms": True,
                                    "hide_codes": True,
                                    "include_next_steps": True
                                }
                            },
                            "PLAN": {
                                "section_type": "PLAN",
                                "include": True,
                                "order": 3,
                                "title_override": "Your Treatment Plan",
                                "format_options": {
                                    "patient_friendly": True,
                                    "simplified_language": True,
                                    "action_oriented": True,
                                    "include_explanations": True,
                                    "highlight_important": True,
                                    "include_what_to_expect": True,
                                    "medication_instructions": True,
                                    "warning_signs": True
                                }
                            },
                            "OTHER": {
                                "section_type": "OTHER",
                                "include": True,
                                "order": 4,
                                "title_override": "Additional Information",
                                "format_options": {
                                    "patient_friendly": True,
                                    "simplified_language": True,
                                    "include_resources": True,
                                    "include_contact_info": True
                                }
                            }
                        },
                        "template_settings": {
                            "show_header": True,
                            "show_footer": True,
                            "include_date": True,
                            "show_provider_name": True,
                            "include_follow_up_details": True,
                            "show_contact_information": True
                        }
                    },
                    version="1.0"
                )
                
                session.add(patient_template)
                session.commit()
                print("Added default patient report template to database")
                
        except Exception as e:
            session.rollback()
            print(f"Error initializing default templates: {str(e)}")
        finally:
            session.close()

    # Helper function to ensure template directory exists
    def _ensure_template_directory(self):
        """Ensure the template directory exists and create HTML template files if needed."""
        template_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "services", "report_generation", "default_html"
        )
        
        # Create directory if it doesn't exist
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
            print(f"Created template directory: {template_dir}")
        
        # Define template file paths
        doctor_template_path = os.path.join(template_dir, "default_doctor_report.html")
        patient_template_path = os.path.join(template_dir, "default_patient_report.html")
        
    def get_session(self):
        """Get a database session"""
        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()