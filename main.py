import os
from sklearn.preprocessing import normalize
import job_data_loader
import google.generativeai as genai
from dotenv import load_dotenv
import faiss


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

df, index, model = job_data_loader.load_and_prepare_data()

def query_gemini(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")  # fixed!
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini API Error: {e}]"


def search_jobs_with_summary(user_query, k=5):
    experience_filter = None
    remote_filter = None
    employment_filter = None
    soft_prompt = ""

    query_lower = user_query.lower()

    if "entry level" in query_lower or "fresher" in query_lower:
        experience_filter = "EN"
        soft_prompt += " This role is for freshers or recent graduates with no prior experience."

    if "remote" in query_lower:
        remote_filter = 100
        soft_prompt += " The job offers full remote work options."

    if "part time" in query_lower or "part-time" in query_lower:
        employment_filter = "PT"
        soft_prompt += " This is a part-time position."

    if "full time" in query_lower or "full-time" in query_lower:
        employment_filter = "FT"
        soft_prompt += " This is a full-time role."

    df_filtered = df.copy()

    if experience_filter:
        df_filtered = df_filtered[df_filtered['experience_level'] == experience_filter]

    if remote_filter is not None:
        df_filtered = df_filtered[df_filtered['remote_ratio'] == remote_filter]

    if employment_filter:
        df_filtered = df_filtered[df_filtered['employment_type'] == employment_filter]

    if df_filtered.empty:
        return [], "❌ No matching job listings found for your query."

    texts = df_filtered['full_text'].tolist()
    sub_embeddings = model.encode(texts, convert_to_numpy=True)
    sub_embeddings = normalize(sub_embeddings, axis=1)

    dim = sub_embeddings.shape[1]
    temp_index = faiss.IndexFlatIP(dim)
    temp_index.add(sub_embeddings)

    full_query = user_query + soft_prompt
    query_vec = model.encode([full_query], convert_to_numpy=True)
    query_vec = normalize(query_vec, axis=1)

    distances, indices = temp_index.search(query_vec, k)

    if len(indices[0]) == 0 or all(d == 0.0 for d in distances[0]):
        return [], "❌ No semantically similar job listings found."

    jobs = df_filtered.iloc[indices[0]].to_dict(orient='records')

    context = "\n\n".join(
        f"{j['job_title']} at {j['company_name']} ({j['company_location']}), "
        f"${j['salary_usd']}, {j['experience_desc']}, {j['employment_type']} type, {j['remote_ratio']}% remote."
        for j in jobs
    )

    prompt = (
        f"You are a helpful assistant. Based only on the job listings below, "
        f"answer the user’s question.\n\n"
        f"{context}\n\n"
        f"User's question: {full_query}"
    )

    summary = query_gemini(prompt)
    return jobs, summary

