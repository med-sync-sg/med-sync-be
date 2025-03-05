from os import environ
import iris
from app.schemas.section import TextCategoryEnum
from sentence_transformers import SentenceTransformer
import pandas as pd
from app.db.umls_data_loader import umls_df_dict
import numpy as np

IRIS_USER = environ.get('IRIS_USER')
IRIS_PASSWORD = environ.get('IRIS_PASSWORD')
IRIS_HOST = environ.get('IRIS_HOST')
IRIS_PORT = environ.get('IRIS_PORT')
IRIS_NAMESPACE = environ.get('IRIS_NAMESPACE')

CONNECTION_STRING = f"{IRIS_HOST}:{IRIS_PORT}/{IRIS_NAMESPACE}"

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
            # cls._instance.set_up_category_embeddings()
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

    def get_category(tui: str) -> str:
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
        
    def embed_text(self, text: str) -> list:
        """
        Generate an embedding for the input text using the preloaded model.
        
        Parameters:
            text (str): The input text.
        
        Returns:
            list: The embedding vector as a list of floats.
        """
        # Encode the text (wrap text in a list to get a batch output)
        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        return embedding
    
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
        input_embedding = self.embed_text(text)
        embedding_str = str(input_embedding)  # Ensure formatting matches what TO_VECTOR expects

        # Step 2: Query the vector database for the closest match.
        sql = f"""
            SELECT TOP 1 category, VECTOR_DISTANCE(description_embeddings, ?) AS distance
            FROM {self.schema_name}.TextCategoryEmbeddings
            ORDER BY distance ASC
        """
        self.cursor.execute(sql, (embedding_str,))
        result = self.cursor.fetchone()

        # Step 3: Check if a result was returned.
        if result:
            category, distance = result
            print(f"Best match category: {category}, distance: {distance}")
            # If the distance is too high, the input text is considered dissimilar.
            if distance > threshold:
                return TextCategoryEnum.OTHERS.name
            else:
                return category
        else:
            # No result found, fallback to OTHERS.
            return TextCategoryEnum.OTHERS.name