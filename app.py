from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

   # Fetch dataset, initialize vectorizer and LSA
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(newsgroups.data)
lsa = TruncatedSVD(n_components=100)
X_lsa = lsa.fit_transform(X)

def search_engine(query):
    query_vec = vectorizer.transform([query])
    query_lsa = lsa.transform(query_vec)
    cosine_similarities = cosine_similarity(query_lsa, X_lsa).flatten()
    indices = np.argsort(cosine_similarities)[::-1][:5]
    documents = [newsgroups.data[i] for i in indices]
    similarities = cosine_similarities[indices]
    return documents, similarities.tolist(), indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)