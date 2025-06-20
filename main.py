import os
from sklearn.preprocessing import normalize
import job_data_loader
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load once
df, index, model = job_data_loader.load_and_prepare_data()

def query_gemini(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")  # fixed!
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini API Error: {e}]"


def search_jobs_with_summary(query, k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    query_vec = normalize(query_vec, axis=1)
    distances, indices = index.search(query_vec, k)
    jobs = df.iloc[indices[0]].to_dict(orient='records')

    context = "\n\n".join(
    f"{j['job_title']} at {j['company_name']} ({j['company_location']}), ${j['salary_usd']}, ${j['remote_ratio']}"
    for j in jobs
)


    prompt = (
        f"You are a helpful assistant. Based only on the job listings below, "
        f"answer the user’s question.\n\n"
        f"{context}\n\n"
        f"User's question: {query}"
    )

    summary = query_gemini(prompt)
    return jobs, summary
