import os
import kagglehub
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L12-v2')

# Paths
DATA_FILE = "cleaned_jobs.pkl"
FAISS_FILE = "faiss_index.bin"

# Required columns for safe processing
REQUIRED_COLUMNS = [ 
    'job_id', 'job_title', 'company_location', 'salary_usd',
    'experience_level', 'employment_type', 'remote_ratio',
    'company_name', 'required_skills'
]

# Load, clean, embed, and return data
def load_and_prepare_data():
    global model

    # Load from disk if already processed
    if os.path.exists(DATA_FILE) and os.path.exists(FAISS_FILE):
        df = pd.read_pickle(DATA_FILE)
        index = faiss.read_index(FAISS_FILE)
        return df, index, model

    # Download and load CSV from Kaggle
    path = kagglehub.dataset_download("bismasajjad/global-ai-job-market-and-salary-trends-2025")
    file_path = os.path.join(path, "ai_job_dataset.csv")
    df = pd.read_csv(file_path)

    # Drop rows missing key information
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    # Standardize experience descriptions
    EXP_LEVEL_MAP = {
        "EN": "Entry level",
        "MI": "Mid level",
        "SE": "Senior level",
        "EX": "Executive level"
    }
    df['experience_desc'] = df['experience_level'].map(EXP_LEVEL_MAP)

    # Construct strong text embedding input
    df['full_text'] = (
        "Title: " + df['job_title'] + ". "
        + "Company: " + df['company_name'] + ". "
        + "Location: " + df['company_location'] + ". "
        + "Experience Required: " + df['experience_desc'] + ". "
        + "Skills Needed: " + df['required_skills'].fillna('') + ". "
        + "Employment Type: " + df['employment_type'] + ". "
        + df['remote_ratio'].astype(str) + "% remote. "
        + "Salary Estimate: $" + df['salary_usd'].astype(int).astype(str) + "."
    )

    # Drop duplicates for indexing quality
    df.drop_duplicates(subset=['full_text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Embed job descriptions
    texts = df['full_text'].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = normalize(embeddings, axis=1)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Cache to disk
    df.to_pickle(DATA_FILE)
    faiss.write_index(index, FAISS_FILE)

    return df, index, model
