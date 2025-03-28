from os import environ
import iris
from app.schemas.section import TextCategoryEnum
from sentence_transformers import SentenceTransformer
import pandas as pd
from app.db.umls_data_loader import umls_df_dict
import numpy as np
import json
from typing import Dict, List, Any, Tuple
import copy
from Levenshtein import ratio
IRIS_USER = environ.get('IRIS_USER', "demo")
IRIS_PASSWORD = environ.get('IRIS_PASSWORD', "demo")
IRIS_HOST = environ.get('IRIS_HOST', "localhost")
IRIS_PORT = environ.get('IRIS_PORT', "1972")
IRIS_NAMESPACE = environ.get('IRIS_NAMESPACE', "USER")

CONNECTION_STRING = f"{IRIS_HOST}:{IRIS_PORT}/{IRIS_NAMESPACE}"

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2)

def table_exists(cursor, table_name: str, schema_name: str = "medsync") -> bool:
    """
    Check if a table exists in the given schema.

    Parameters:
        cursor: A database cursor object.
        table_name (str): The name of the table to check.
        schema_name (str): The schema in which to look for the table. Default is "dbo".

    Returns:
        bool: True if the table exists, False otherwise.
    """
    sql = """
    SELECT 1 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
    """
    cursor.execute(sql, (schema_name, table_name))
    result = cursor.fetchone()
    return result is not None

'''
IrisDataStore should be used when communicating with the InterSystems IRIS SQL database, which enables
vector search and basic SQL features.
'''
class IrisDataStore:
    _instance = None
    model: SentenceTransformer
    schema_name: str = "medsync"
    class Config:
        arbitrary_types_allowed = True
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IrisDataStore, cls).__new__(cls)
            
            # Load a pre-trained sentence transformer model. This model's output vectors are of size 384
            cls._instance.model = SentenceTransformer('all-MiniLM-L6-v2')
            cls._instance.conn = iris.connect(CONNECTION_STRING, username=IRIS_USER, password=IRIS_PASSWORD)
            cls._instance.cursor = cls._instance.conn.cursor()
            print("Connected to IRIS successfully!")
            
            cls._instance.set_up_categories()
            cls._instance.set_up_category_embeddings()
            cls._instance.set_up_content_dictionary_embeddings()
            
        return cls._instance
    
    def set_up_categories(self):
        table_name = "TextCategories"
        print("Setting up text categories...")
        if not table_exists(self.cursor, table_name, self.schema_name):
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} (category VARCHAR(1000), description VARCHAR(20000), description_embeddings VECTOR(DOUBLE, 384))"
            self.cursor.execute(create_table_sql)
            data = {
                "category": [],
                "description": [],
                "description_embeddings": [],
            }
            for text_category in TextCategoryEnum:
                data["category"].append(text_category.name)
                data["description"].append(text_category.value)
                data["description_embeddings"].append(self.model.encode(text_category.value, normalize_embeddings=True))
            
            result_df = pd.DataFrame(data=data)
            
            data = [
            (
                row['category'], 
                row['description'],
                str(row["description_embeddings"])
            )
                for index, row in result_df.iterrows()
            ]
            
            sql = f"""
                INSERT INTO {self.schema_name}.{table_name}
                (category, description, description_embeddings)
                VALUES (?, ?, TO_VECTOR(?))
            """
            self.cursor.executemany(sql, data)
        else:
            print("Table already exists!")


    def _get_category(tui: str) -> str:
        SYMPTOMS_AND_DISEASES_TUI = [
            'T047',  # Disease or Syndrome
            'T184',  # Sign or Symptom
        ]
        
        DRUGS_AND_MEDICINES_TUI = [
            'T195',
            'T200',
        ]

        PATIENT_INFORMATION_TUI = [
            'T98',
            'T99',
            'T100'
        ]
        if tui in SYMPTOMS_AND_DISEASES_TUI:
            return TextCategoryEnum.CHIEF_COMPLAINT.name
        if tui in PATIENT_INFORMATION_TUI:
            return TextCategoryEnum.PATIENT_INFORMATION.name
        return TextCategoryEnum.OTHERS.name

    def set_up_category_embeddings(self):
        print("Loading up category embeddings...")
        table_name = "TextCategoryEmbeddings"
        if not table_exists(self.cursor, table_name, self.schema_name):
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} (cui VARCHAR(100), term VARCHAR(1000), term_embeddings VECTOR(DOUBLE, 384), category VARCHAR(1000), semantic_type VARCHAR(1000), description VARCHAR(20000), description_embeddings VECTOR(DOUBLE, 384))"
            self.cursor.execute(create_table_sql)
            
            # Load concepts, semantic types, and definitions.        
            concepts_with_sty_def_df = umls_df_dict["concepts_with_sty_def_df"]
            # patient_information_df = umls_df_dict["patient_information_df"]
            
            # full_df = pd.concat([concepts_with_sty_def_df, patient_information_df])
            full_df = concepts_with_sty_def_df
            term_embeddings = self.model.encode(full_df["STR"].tolist(), normalize_embeddings=True).tolist()
            term_embeddings = [str(embedding) for embedding in term_embeddings]
            
            description_embeddings = self.model.encode(full_df["DEF"].tolist(), normalize_embeddings=True).tolist()
            description_embeddings = [str(embedding) for embedding in description_embeddings]
            print("Successfully created embeddings.")
            
            categories = full_df["TUI"].apply(lambda element: IrisDataStore._get_category(tui=element))
            print(categories)
            data = {
                "cui": full_df["CUI"].tolist(),
                "term": full_df["STR"].tolist(),
                "term_embeddings": term_embeddings,
                "category": categories,
                "semantic_type": full_df["STY"].tolist(),
                "description": full_df["DEF"].tolist(),
                "description_embeddings": description_embeddings
            }
            result = list(zip(
                data["cui"],
                data["term"],
                data["term_embeddings"],
                data["category"],
                data["semantic_type"],
                data["description"],
                data["description_embeddings"]
            ))
            
            
            print(f"Number of terms with embeddings: {len(result)}")
                
            sql = f"""
                INSERT INTO {self.schema_name}.{table_name}
                (cui, term, term_embeddings, category, semantic_type, description, description_embeddings)
                VALUES (?, ?, TO_VECTOR(?), ?, ?, ?, TO_VECTOR(?))
            """
                    
            current = 0
            batch_size = 1000
            while current < len(result):
                print(f"Uploading batch... index {current} with size {batch_size}")
                self.cursor.executemany(sql, result[current: current + batch_size])
                current = current + batch_size
            print("Successfully uploaded embeddings to IRIS database.")
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the input text using the preloaded model.
        
        Parameters:
            text (str): The input text.
        
        Returns:
            list: The embedding vector as a list of floats.
        """
        # Encode the text
        embeddings = self.model.encode(text, normalize_embeddings=True)
        return embeddings
    
    def embed_list(self, input_list: list) -> List[str]:
        embeddings = self.model.encode(input_list, normalize_embeddings=True)
        embeddings_str_list = []
        for embedding in embeddings:
            embeddings_str_list.append(str(embedding))
            
        return embeddings_str_list
    
    def embed_dictionary(self, data: Any) -> np.ndarray:
        embeddings = self.model.encode(json.dumps(data), normalize_embeddings=True)
        return embeddings
    


    def classify_text_category(self, text: str, threshold: float = 0.5) -> str:
        """
        Classify the input text by performing a vector search against stored category embeddings,
        and also taking into account the similarity of the term string itself.
        
        This method computes:
        1. A combined vector similarity score from both the term_embeddings and description_embeddings.
        2. A direct string similarity score (using Levenshtein ratio) between the input text and the stored term.
        
        The final score is a combination of these scores. If the final score is above the threshold,
        it returns the matched category; otherwise, it falls back to OTHERS.
        """
        # Step 1: Convert the input text into its embedding.
        # Convert the embedding vector to a Python list then to a string.
        input_embedding = str(self.embed_text(text).tolist())
        
        # Step 2: Query the vector database.
        # The query computes a combined score using both term_embeddings and description_embeddings.
        sql = f"""
            SELECT TOP 1 category, term, VECTOR_DOT_PRODUCT(term_embeddings, TO_VECTOR(?)) as score
            FROM {self.schema_name}.TextCategoryEmbeddings
            ORDER BY score DESC
        """
        # For now, if you want to search for a specific category, pass it; otherwise, you could remove the WHERE clause.
        # Here, we search for the category provided (converted to uppercase).
        self.cursor.execute(sql, [input_embedding])
        result = self.cursor.fetchone()
        
        if result:
            category, db_term, score = result
            print(f"Vector result - Category: {category}, Term: {db_term}, Score: {score}")
            
            # Step 3: Compute string similarity between input text and the stored term.
            string_sim = ratio(text.lower(), db_term.lower())
            print(f"String similarity between '{text}' and '{db_term}': {string_sim:.4f}")
            
            # Step 4: Combine the vector and string similarities.
            if string_sim > 0.85:
                return TextCategoryEnum[category].name
            
            if float(score) >= threshold:
                return TextCategoryEnum[category].name
            else:
                return TextCategoryEnum.OTHERS.name
        else:
            # If no result is found, fallback to OTHERS.
            return TextCategoryEnum.OTHERS.name
    
    def set_up_content_dictionary_embeddings(self):
        print("Setting up content dictionary embeddings...")
        table_name = "ContentDictionaryJson"

        if not table_exists(self.cursor, table_name, self.schema_name):
            chief_complaint_example = {
                "Main Symptom": {
                    "name": "The name of the primary symptom (e.g., headache, diarrhea)",
                    "duration": "The duration of the symptom (e.g., 3 days)",
                    "severity": "The severity level of the symptom (e.g., mild, moderate, severe)",
                    "additional_details": "Additional details regarding the symptom, such as frequency, triggers, or associated features"
                }
            }
            
            patient_information_example = {
                "Demographics": {
                    "name": "Patient's full name",
                    "age": "Patient's age in years",
                    "gender": "Patient's gender (e.g., Male, Female)"
                },
                "Contact Information": {
                    "address": "Patient's address (e.g., street, city, country, postal code)",
                    "phone": "Patient's contact phone number"
                },
                "Occupation": "Patient's occupation (e.g., teacher, pharmaceutical manager)",
                "Additional Details": "Other relevant personal details (e.g., marital status, living conditions, recent travel history)"
            }
            
            medical_history_example = {
                "Chronic Conditions": [
                    {
                        "name": "The name of a chronic condition (e.g., eczema)",
                        "diagnosed_date": "The date the condition was diagnosed (format: YYYY-MM-DD)",
                        "severity": "The severity of the condition (e.g., mild, moderate, severe)",
                        "notes": "Additional notes regarding the condition (e.g., treatment response, frequency of flare-ups)"
                    }
                ],
                "Past Diagnoses": [
                    {
                        "name": "The name of a past diagnosis (e.g., asthma)",
                        "diagnosed_date": "The date of diagnosis (format: YYYY-MM-DD)",
                        "notes": "Additional details regarding the diagnosis (e.g., management, history)"
                    }
                ],
                "Medications": [
                    {
                        "name": "The name of a medication (e.g., Ibuprofen)",
                        "dosage": "The dosage of the medication (e.g., 200 mg)",
                        "frequency": "The frequency at which the medication is taken (e.g., as needed)",
                        "notes": "Additional notes (e.g., response, side effects)"
                    }
                ],
                "Allergies": [
                    {
                        "substance": "The allergen or substance causing an allergic reaction (e.g., penicillin)",
                        "notes": "Details about the allergy or any additional notes (e.g., NKDA if none)"
                    }
                ]
            }
            
            others_examples = {
                "Other Observations": [
                    {
                    "observation": "A miscellaneous observation not covered by the primary categories",
                    "notes": "Additional details or context regarding the observation"
                    }
                ]
            }
    
            examples = [
                (chief_complaint_example, TextCategoryEnum.CHIEF_COMPLAINT.name, "Contains chief complaints or main symptoms of the patient."), 
                        (patient_information_example, TextCategoryEnum.PATIENT_INFORMATION.name, "Contains patient information such as demographics."), 
                        (medical_history_example, TextCategoryEnum.PATIENT_MEDICAL_HISTORY.name, "Contains the medical history information of the patient such as current medications or chronic illnesses."), 
                        (others_examples, TextCategoryEnum.OTHERS.name, "Any other information belongs to this section.")
                        ]            
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} (content_dictionary VARCHAR(32000), category VARCHAR(1000), description VARCHAR(20000), embeddings VECTOR(DOUBLE, 384), description_embeddings VECTOR(DOUBLE, 384))"
            self.cursor.execute(create_table_sql)
            
            data = [
                (json.dumps(example), category, description, str(self.embed_dictionary(example).tolist()), str(self.embed_text(description).tolist()))
                for example, category, description in examples
            ]
            
            sql = f"""
                INSERT INTO {self.schema_name}.{table_name}
                (content_dictionary, category, description, embeddings, description_embeddings)
                VALUES (?, ?, ?, TO_VECTOR(?), TO_VECTOR(?))
            """

            self.cursor.executemany(sql, data)
            print("Successfully set up dictionary embeddings in the IRIS database.")

    def find_content_dictionary(self, keyword_dict: dict, category: str) -> dict:
        table_name = "ContentDictionaryJson"

        # Step 1: Convert the input text into its embedding.
        # input_embedding = self.embed_dictionary(keyword_dict)
        input_embedding = self.model.encode(json.dumps(keyword_dict), normalize_embeddings=True).tolist()
        # Step 2: Query the vector database for the closest match.
        sql = f"""
            SELECT TOP 1 content_dictionary
            FROM {self.schema_name}.{table_name}
            WHERE category='{category.upper()}'
            ORDER BY VECTOR_DOT_PRODUCT(description_embeddings, TO_VECTOR(?)) DESC
        """
        
        self.cursor.execute(sql, [str(input_embedding)])
        result, = self.cursor.fetchone()
        
        print(result)
        
        return json.loads(result)

    def clear_template_values(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively clear all string values in the template, replacing them with an empty string,
        while preserving the structure.
        """
        new_template = {}
        for key, value in template.items():
            if isinstance(value, str):
                new_template[key] = ""
            elif isinstance(value, dict):
                new_template[key] = self.clear_template_values(value)
            else:
                new_template[key] = value
        return new_template


    def iter_candidate_keys(self, template: Dict[str, Any]) -> List[Tuple[List[str], str]]:
        """
        Yield candidate key paths and their combined strings from the template.
        For each top-level key with a string value, yield ([key], "key: description").
        For each nested key (one level deep) where the value is a string, yield ([top_key, sub_key], "sub_key: description").
        """
        candidates = []
        for t_key, t_val in template.items():
            if isinstance(t_val, str):
                candidates.append(([t_key], f"{t_key}: {t_val}"))
            elif isinstance(t_val, dict):
                for sub_key, sub_val in t_val.items():
                    if isinstance(sub_val, str):
                        candidates.append(([t_key, sub_key], f"{sub_key}: {sub_val}"))
        return candidates
    
    def set_value_at_path(self, template: Dict[str, Any], path: List[str], value: Any) -> Dict[str, Any]:
        """
        Set the given value in the template at the location specified by the path (list of keys).
        If the key does not exist, create an empty dict for that key.
        Returns the updated template.
        """
        current = template
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value
        return template

    def get_value_at_path(self, template: Dict[str, Any], path: List[str]) -> Any:
        """
        Retrieve the value at the given key path in the template.
        Returns None if any key is missing.
        """
        current = template
        for key in path:
            current = current.get(key)
            if current is None:
                return None
        return current

    def append_value_at_path(self, template: Dict[str, Any], path: List[str], new_value: Any) -> Dict[str, Any]:
        """
        Append new_value to the field specified by path in the template.
        If the field already contains a nonempty string, new_value is appended
        with a comma separator.
        Returns the updated template.
        """
        current = template
        for key in path[:-1]:
            current = current.setdefault(key, {})
        existing = current.get(path[-1], "")
        if existing:
            current[path[-1]] = f"{existing}, {new_value}"
        else:
            current[path[-1]] = str(new_value)
        return template

    def merge_flat_keywords_into_template(self, feature_dict: Dict[str, Any],
                                            template: Dict[str, Any],
                                            threshold: float = 0.5) -> Dict[str, Any]:
        """
        Merge a flat feature dictionary into a nested content template (nested one level deep),
        using a prototype-based approach that considers both the key and its value.
        
        For each key in feature_dict (e.g., "term", "duration", "adjectives", "quantities"):
        - Build a combined string from the feature (e.g. "term: severe headache").
        - For every candidate field from the template—both top-level keys (if its value is a string)
            and nested keys (one level only)—build candidate strings using the key and its description.
        - Compute cosine similarity between the feature string and each candidate.
        - If the best candidate (similarity >= threshold) is found, update that field in a working
            template (whose string values have been cleared).  
            * If the feature value is a list, process each element iteratively, appending each
            element to the field.
        - Otherwise, store the key/value pair in an "additional_content" bucket.
        
        Returns a new dictionary with the same structure (and order) as the template, updated with feature data.
        """
        similarity_template = copy.deepcopy(template)
        working_template = self.clear_template_values(copy.deepcopy(template))
        extras = {}

        # Build a list of candidate keys from the template (one level deep)
        # Each candidate is a tuple: (path, candidate_string)
        candidates = []
        for t_key, t_val in similarity_template.items():
            if isinstance(t_val, str):
                candidates.append(([t_key], f"{t_key}: {t_val}"))
            elif isinstance(t_val, dict):
                for sub_key, sub_val in t_val.items():
                    if isinstance(sub_val, str):
                        candidates.append(([t_key, sub_key], f"{sub_key}: {sub_val}"))
        
        for f_key, f_value in feature_dict.items():
            if f_key == "label":
                continue

            # Process list values iteratively.
            if isinstance(f_value, list):
                for element in f_value:
                    element_str = str(element)
                    feature_candidate = f"{f_key}: {element_str}"
                    best_sim = -1.0
                    best_path = []
                    for path, candidate in candidates:
                        sim = cosine_similarity(self.embed_text(feature_candidate), self.embed_text(candidate))
                        if sim > best_sim:
                            best_sim = sim
                            best_path = path
                    if best_path:
                        # If there's already content, append; otherwise, set the value.
                        current_value = self.get_value_at_path(working_template, best_path)
                        if current_value:
                            working_template = self.append_value_at_path(working_template, best_path, element_str)
                        else:
                            working_template = self.set_value_at_path(working_template, best_path, element_str)
                    else:
                        extras.setdefault(f_key, []).append(element_str)
            else:
                if f_key == "term" and isinstance(f_value, str):
                    if working_template.get("Main Symptom") != None and f_value != "symptoms" and f_value != "feverishness" and f_value != "painful":
                        best_path = ["Main Symptom", "name"]
                        working_template = self.set_value_at_path(working_template, best_path, f_value)
                else:
                    f_val_str = str(f_value)
                    feature_candidate = f"{f_key}: {f_val_str}"
                    best_sim = -1.0
                    best_path = []
                    for path, candidate in candidates:
                        sim = cosine_similarity(self.embed_text(feature_candidate), self.embed_text(f"{path[1]}: ${candidate}"))
                        if sim > best_sim:
                            best_sim = sim
                            best_path = path
                    if best_path and best_sim >= threshold:
                        working_template = self.set_value_at_path(working_template, best_path, f_val_str)
                    else:
                        extras[f_key] = f_value

        if extras:
            if "additional_content" in working_template and isinstance(working_template["additional_content"], dict):
                working_template["additional_content"].update(extras)
            else:
                working_template["additional_content"] = extras
                
        print("Final Working Template: " , working_template)
        return working_template