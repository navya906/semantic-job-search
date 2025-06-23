from flask import Flask, render_template, request
from job_data_loader import load_and_prepare_data
from main import search_jobs_with_summary

app = Flask(__name__)

# Load model and data only once at startup
df, index, model = load_and_prepare_data()


@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    summary = ""
    query = ""

    if request.method == 'POST':
        query = request.form.get('query', '').strip()

        if query:
            try:
                results, summary = search_jobs_with_summary(query, k=5)
            except Exception as e:
                summary = f"⚠️ Something went wrong: {str(e)}"
        else:
            summary = "Please enter a job-related query to begin."

    return render_template('index.html', query=query, results=results, summary=summary)


if __name__ == '__main__':
    # Use debug=True for local dev, False for deployment
    app.run(debug=True)
