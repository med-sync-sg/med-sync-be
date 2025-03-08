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
IRIS_USER = environ.get('IRIS_USER')
IRIS_PASSWORD = environ.get('IRIS_PASSWORD')
IRIS_HOST = environ.get('IRIS_HOST')
IRIS_PORT = environ.get('IRIS_PORT')
IRIS_NAMESPACE = environ.get('IRIS_NAMESPACE')

CONNECTION_STRING = f"{IRIS_HOST}:{IRIS_PORT}/{IRIS_NAMESPACE}"

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2)

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
            
            # cls._instance.set_up_categories()
            # cls._instance.set_up_category_embeddings()
            # cls._instance.set_up_content_dictionary_embeddings()
            
        return cls._instance
    
    def set_up_categories(self):
        table_name = "TextCategories"
        print("Setting up text categories...")
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

    def _get_category(self, tui: str) -> str:
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
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} (cui VARCHAR(100), term VARCHAR(1000), category VARCHAR(1000), semantic_type VARCHAR(1000), description VARCHAR(20000), description_embeddings VECTOR(DOUBLE, 384))"
        self.cursor.execute(create_table_sql)
        
        # Load concepts, semantic types, and definitions.        
        concepts_with_sty_def_df = umls_df_dict["concepts_with_sty_def_df"]
        patient_information_df = umls_df_dict["patient_information_df"]
        
        full_df = pd.concat([concepts_with_sty_def_df, patient_information_df])
        
        description_embeddings = self.model.encode(full_df["DEF"].tolist(), normalize_embeddings=True).tolist()
        description_embeddings = [str(embedding) for embedding in description_embeddings]
        print("Successfully created embeddings.")
        
        categories = full_df["TUI"].apply(IrisDataStore.get_category)
        print(categories)
        data = {
            "cui": full_df["CUI"].tolist(),
            "term": full_df["STR"].tolist(),
            "category": categories,
            "semantic_type": full_df["STY"].tolist(),
            "description": full_df["DEF"].tolist(),
            "description_embeddings": description_embeddings
        }
        result = list(zip(
            data["cui"],
            data["term"],
            data["category"],
            data["semantic_type"],
            data["description"],
            data["description_embeddings"]
        ))
        
        
        print(f"Number of terms with embeddings: {len(result)}")
            
        sql = f"""
            INSERT INTO {self.schema_name}.{table_name}
            (cui, term, category, semantic_type, description, description_embeddings)
            VALUES (?, ?, ?, ?, ?, TO_VECTOR(?))
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
    
    def classify_text_category(self, text: str, threshold: float = 0.8) -> str:
        """
        Classify the input text by performing a vector search against stored category embeddings.
        If no result is found or the best distance is greater than the threshold, return the fallback category (OTHERS).

        Parameters:
            text (str): The input text to classify.
            threshold (float): Maximum acceptable distance for a match. If the best match's distance 
                            exceeds this value, fallback to OTHERS.
        
        Returns:
            str: The predicted category, or TextCategoryEnum.OTHERS if no good match is found.
        """
        # Step 1: Convert the input text into its embedding.
        input_embedding = str(self.embed_text(text).tolist())

        # Step 2: Query the vector database for the closest match.
        sql = f"""
            SELECT TOP 1 category
            FROM {self.schema_name}.TextCategoryEmbeddings
            ORDER BY VECTOR_DOT_PRODUCT(description_embeddings, TO_VECTOR(?)) DESC
        """
        self.cursor.execute(sql, [input_embedding])
        result = self.cursor.fetchone()

        # Step 3: Check if a result was returned.
        if result:
            category, = result
            print(f"Best match category: {category}")
            # If the distance is too high, the input text is considered dissimilar.
            # if distance > threshold:
            return TextCategoryEnum[category].name
            # else:
            #     return category
        else:
            # No result found, fallback to OTHERS.
            return TextCategoryEnum.OTHERS.name
    
    def set_up_content_dictionary_embeddings(self):
        print("Setting up content dictionary embeddings...")
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
        
        table_name = "ContentDictionaryJson"
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} (content_dictionary VARCHAR(32000), category VARCHAR(1000), description VARCHAR(20000), embeddings VECTOR(DOUBLE, 384), description_embeddings VECTOR(DOUBLE, 384))"
        self.cursor.execute(create_table_sql)
        
        description_1 = "Contains chief complaints or main symptoms of the patient."
        description_2 = "Contains patient information such as demographics."
        description_3 = "Contains the medical history information of the patient such as current medications or chronic illnesses."
        description_4 = "Any other information belongs to this section."
        
        data = [
            (chief_complaint_example, TextCategoryEnum.CHIEF_COMPLAINT.name, description_1, self.embed_dictionary(chief_complaint_example), str(self.embed_text(description_1).tolist())),
            (patient_information_example, TextCategoryEnum.PATIENT_INFORMATION.name, description_2, self.embed_dictionary(patient_information_example), str(self.embed_text(description_2).tolist())),
            (medical_history_example, TextCategoryEnum.PATIENT_MEDICAL_HISTORY.name, description_3, self.embed_dictionary(medical_history_example), str(self.embed_text(description_3).tolist())),
            (others_examples, TextCategoryEnum.OTHERS.name, description_4, self.embed_dictionary(others_examples), str(self.embed_text(description_4).tolist()))
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

    def find_best_key_path(self, keyword_key: str, template: Dict[str, Any], threshold: float) -> Tuple[List[str], float]:
        """
        Recursively search the template for the key path that best matches the keyword_key.
        Returns a tuple (path, similarity) where path is a list of keys indicating where in the template
        the keyword should be mapped. If no key reaches the threshold, returns ([], best_similarity).
        """
        best_path = []
        best_sim = -1.0
        for t_key, t_value in template.items():
            sim = cosine_similarity(self.embed_text(keyword_key), self.embed_text(t_key))
            if sim > best_sim:
                best_sim = sim
                best_path = [t_key]
            # If the current value is a nested dict, search deeper.
            if isinstance(t_value, dict):
                sub_path, sub_sim = self.find_best_key_path(keyword_key, t_value, threshold)
                if sub_sim > best_sim:
                    best_sim = sub_sim
                    best_path = [t_key] + sub_path
        return best_path, best_sim

    def set_value_at_path(self, template: Dict[str, Any], path: List[str], new_value: Any) -> None:
        """
        Given a nested dictionary (template) and a list of keys (path), update the value at the deepest key.
        """
        current = template
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = new_value
        
    def append_value_at_path(self, template: Dict[str, Any], path: List[str], new_value: Any) -> None:
        """
        Given a nested dictionary (template) and a list of keys (path), update the value at the deepest key.
        """
        current = template
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] += f", {new_value}" 
        
    def recursive_fill_content_dictionary(self, keyword_dict: Dict[str, Any],
                                            content_dictionary: Dict[str, Any],
                                            threshold: float = 0.1) -> Dict[str, Any]:
        """
        Merge a flat keyword dictionary into a nested content template.
        
        For each key in keyword_dict (excluding keys like "label"):
          - Recursively search the template for the best matching key path using semantic similarity.
          - If a matching path is found (similarity above threshold), set that field to the keyword's value.
          - Otherwise, add the key/value pair to an "additional_content" bucket.
        
        Returns a new dictionary with the same structure (and order) as the template, updated with the keyword data.
        """
        # Make a deep copy of the content template to preserve its structure.
        merged = copy.deepcopy(content_dictionary)
        extras = {}

        if not keyword_dict:
            raise Exception("Keyword dictionary cannot be empty or None.")
        if not content_dictionary:
            raise Exception("Content dictionary cannot be empty or None.")

        # Process each key from the flat keyword dictionary.
        for k_key, k_value in keyword_dict.items():
            if k_key == "label":
                continue
            path, sim = self.find_best_key_path(k_key, merged, threshold)
            print("Path: ", path)
            if path:
                if isinstance(k_value, list):
                    added = False
                    self.set_value_at_path(merged, path, "")
                    for val in k_value:
                        adj_path, adj_sim = self.find_best_key_path(val, merged, threshold)
                        if adj_sim >= threshold:
                            self.append_value_at_path(merged, adj_path, val)
                            added = True
                    if not added:
                        self.set_value_at_path(merged, path, "N/A")
                elif isinstance(k_value, str):
                    if sim >= threshold:
                        self.set_value_at_path(merged, path, k_value)
                    else:
                        self.set_value_at_path(merged, path, "N/A")
                else:
                    self.set_value_at_path(merged, path, "N/A")
            else:
                extras[k_key] = k_value

        if extras:
            if "additional_content" in merged and isinstance(merged["additional_content"], dict):
                merged["additional_content"].update(extras)
            else:
                merged["additional_content"] = extras
        print(merged)
        return merged