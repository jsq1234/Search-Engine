from main import SearchEngine
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
engine = SearchEngine('index.txt')

@app.get('/api/search')
def search():
    query = request.args.get('query') or ''
    k = int(request.args.get('k')) or 10
    ranked_docs = engine.searchQuery(query, k)
    results = engine.get_results(ranked_docs)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
