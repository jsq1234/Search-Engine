from main import SearchEngine
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
engine = SearchEngine('index.txt')

@app.post('/api/search')
def search():
    data = request.get_json()
    query = data.get('query', ' ')
    k = data.get('k', 10)
    ranked_docs = engine.searchQuery(query, k)
    results = engine.get_results(ranked_docs)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
