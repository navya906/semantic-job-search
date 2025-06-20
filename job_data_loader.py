import os
import kagglehub
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Cached filenames
DATA_FILE = "cleaned_jobs.pkl"
FAISS_FILE = "faiss_index.bin"

# Required columns for data cleaning
REQUIRED_COLUMNS = [
    'job_id', 'job_title', 'company_location', 'salary_usd',
    'experience_level', 'employment_type', 'remote_ratio',
    'company_name', 'required_skills'
]

def load_and_prepare_data():
    global model

    # ✅ Load from cache if available
    if os.path.exists(DATA_FILE) and os.path.exists(FAISS_FILE):
        print("✅ Loaded cached dataset and FAISS index.")
        df = pd.read_pickle(DATA_FILE)
        index = faiss.read_index(FAISS_FILE)
        return df, index, model

    # ⏳ Step 1: Download the dataset
    print("⏬ Downloading dataset from KaggleHub...")
    try:
        path = kagglehub.dataset_download("bismasajjad/global-ai-job-market-and-salary-trends-2025")
        file_path = os.path.join(path, "ai_job_dataset.csv")
        df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download or load dataset: {e}")

    # 🧹 Step 2: Clean and filter
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    # 📝 Step 3: Construct searchable text
    df['full_text'] = (
        "Job ID: " + df['job_id'].astype(str) + ". "
        + "Title: " + df['job_title'] + ". "
        + "Keywords: AI, ML, Machine Learning, Data Science, Remote. "
        + "Company: " + df['company_name'] + ". "
        + "Location: " + df['company_location'] + ". "
        + "Experience: " + df['experience_level'] + ". "
        + "Skills: " + df['required_skills'].fillna('') + ". "
        + "Employment: " + df['employment_type'] + ". "
        + df['remote_ratio'].astype(str) + "% remote. "
        + "Salary: $" + df['salary_usd'].astype(int).astype(str) + "."
    )

    df.drop_duplicates(subset=['full_text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 📐 Step 4: Embed and normalize
    print("🧠 Generating sentence embeddings...")
    texts = df['full_text'].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = normalize(embeddings, axis=1)

    # 🔍 Step 5: Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 💾 Step 6: Cache for future reuse
    df.to_pickle(DATA_FILE)
    faiss.write_index(index, FAISS_FILE)
    print(f"✅ FAISS index created and cached with {len(df)} jobs.")

    return df, index, model
