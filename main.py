import os
import faiss
from sklearn.preprocessing import normalize
import job_data_loader
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables and Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load the data, embeddings, and model
df, index, model = job_data_loader.load_and_prepare_data()

# Gemini summarizer
def query_gemini(prompt):
    try:
        gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini API Error: {e}]"

# Main semantic job search with relevance filtering
def search_jobs_with_summary(user_query, k=20):
    experience_filter = None
    remote_filter = None
    employment_filter = None
    soft_prompt = ""

    query_lower = user_query.lower()

    # Basic intent extraction from query
    if "entry level" in query_lower or "fresher" in query_lower:
        experience_filter = "EN"
        soft_prompt += " This role is for freshers or recent graduates."

    if "senior" in query_lower or "lead" in query_lower:
        experience_filter = "SE"
        soft_prompt += " The user is looking for senior-level roles."

    if "remote" in query_lower:
        remote_filter = 100
        soft_prompt += " The job should be fully remote."

    if "part time" in query_lower or "part-time" in query_lower:
        employment_filter = "PT"
        soft_prompt += " This is a part-time position."

    if "full time" in query_lower or "full-time" in query_lower:
        employment_filter = "FT"
        soft_prompt += " This is a full-time position."

    # Apply filters
    df_filtered = df.copy()
    if experience_filter:
        df_filtered = df_filtered[df_filtered['experience_level'] == experience_filter]
    if remote_filter is not None:
        df_filtered = df_filtered[df_filtered['remote_ratio'] == remote_filter]
    if employment_filter:
        df_filtered = df_filtered[df_filtered['employment_type'] == employment_filter]

    if df_filtered.empty:
        return [], "❌ No matching job listings found for your query."

    # Embed and index the filtered job texts
    texts = df_filtered['full_text'].tolist()
    sub_embeddings = model.encode(texts, convert_to_numpy=True)
    sub_embeddings = normalize(sub_embeddings, axis=1)

    dim = sub_embeddings.shape[1]
    temp_index = faiss.IndexFlatIP(dim)
    temp_index.add(sub_embeddings)

    # Embed the user's query with intent
    full_query = user_query + soft_prompt
    query_vec = model.encode([full_query], convert_to_numpy=True)
    query_vec = normalize(query_vec, axis=1)

    distances, indices = temp_index.search(query_vec, k)

    # Keep only relevant matches
    SIMILARITY_THRESHOLD = 0.670
    jobs = []
    for i, score in zip(indices[0], distances[0]):
        if score >= SIMILARITY_THRESHOLD:
            job = df_filtered.iloc[i].to_dict()
            job['similarity'] = round(float(score), 3)
            jobs.append(job)

    if not jobs:
        return [], "❌ No semantically relevant job listings found."

    # Sort by similarity descending
    jobs.sort(key=lambda x: x['similarity'], reverse=True)

    # Create a Gemini prompt from top 5 matches
    context = "\n\n".join(
        f"{j['job_title']} at {j['company_name']} ({j['company_location']}), "
        f"${j['salary_usd']}, {j['experience_desc']}, {j['employment_type']} type, {j['remote_ratio']}% remote."
        for j in jobs[:5]
    )

    prompt = (
        f"You are a helpful assistant. Based only on the job listings below, answer the user’s question.\n\n"
        f"{context}\n\n"
        f"User's question: {full_query}"
    )

    summary = query_gemini(prompt)
    if summary.startswith("[Gemini API Error"):
        summary += "\n\nYou can still view the results listed below."

    return jobs[:5], summary
