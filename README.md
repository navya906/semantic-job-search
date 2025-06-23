# Semantic AI Job Search Engine 💼🔍

An AI-powered job search tool that helps users find relevant tech jobs using **natural language** queries and provides **smart summaries** powered by **Google Gemini 1.5 Flash**.

## 🚀 Features

- 🔍 **Semantic search** using SentenceTransformers and FAISS
- 💬 **Contextual summaries** using Gemini (Google Generative AI)
- 🌐 Natural language query support (e.g., *"remote AI jobs in Europe with high salary"*)
- 📊 Highlights key job details: title, company, salary, location, and remote ratio
- 🖥️ Simple and clean **Flask web interface**

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Flask** | Web framework |
| **SentenceTransformers** | Embedding job listings & user queries |
| **FAISS** | Fast similarity search |
| **Gemini 1.5 Flash** | Generating smart summaries |
| **Kaggle Dataset** | [Global AI Job Market & Salary Trends 2025](https://www.kaggle.com/datasets) |

## 📂 Project Structure
```
├── app.py # Main Flask app
├── job_data_loader.py # Loads and caches job data + FAISS index
├── main.py # Handles search logic and summaries
├── templates/
│ └── index.html # UI for job search
├── .env # Contains GEMINI_API_KEY (not uploaded)
├── .gitignore # Ignores cache, venv, env files, etc.
```


## 🧠 How It Works

1. User types a **natural language query**
2. Query is converted into an embedding via **SentenceTransformer**
3. FAISS searches for the top-matching jobs based on vector similarity
4. **Gemini** generates a helpful response or summary from the top results
5. Results and the summary are shown to the user via the web interface

## 🛡️ Environment Variables

Create a `.env` file with:

```env
GEMINI_API_KEY=your_google_api_key
```

## 🧪 Run Locally
```
git clone https://github.com/navya906/semantic-job-search.git
cd semantic-job-search
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000` in your browser!

## 📌 Notes
-The .env file is excluded for safety (API keys should never be pushed).
-You can use any SentenceTransformer model of your choice.
-This is a development version — not deployed yet.

## 🙋‍♀️ About Me
I'm a Computer Science undergrad at VIT, passionate about Data Science, AI, and building meaningful tech.
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/navya-g-a97051314/)

## ⭐️ License
This project is open-source and free to use for educational purposes.

