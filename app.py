from flask import Flask, render_template, request
from job_data_loader import load_and_prepare_data
from main import search_jobs_with_summary 

app = Flask(__name__)

# Load model and data only once
df, index, model = load_and_prepare_data()


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    summary = ""
    query = ""

    if request.method == 'POST':
        query = request.form['query']
        try:
            results, summary = search_jobs_with_summary(query, k=5)
        except Exception as e:
            summary = f"⚠️ Something went wrong: {str(e)}"

    return render_template('index.html', query=query, results=results, summary=summary)


if __name__ == '__main__':
    app.run(debug=True)
