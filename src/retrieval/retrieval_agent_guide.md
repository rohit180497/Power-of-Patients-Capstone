# Retrieval Agent Usage Guide

## Basic Usage

```python
from retrieval_agent import retrieve_articles

# Basic query with anonymous user
results = retrieve_articles(
    query="What are the symptoms of a concussion?",
    top_k=5
)

# Query with user ID (for tracking)
results = retrieve_articles(
    query="How long does recovery from TBI take?",
    top_k=3,
    user_id="patient_123"  # or "doctor_456"
)

# Print results
for result in results:
    print(f"Title: {result['title']}")
    print(f"Read time: {result['read_time']}")
    print(f"Summary: {result['summary']}")
    print(f"URL: {result['url']}")
    print("---")
```

## Environment Setup

Create a `.env` file with your credentials:

```
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=us-east1-aws
PINECONE_INDEX_NAME=power-of-patients-tbi
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
QUERY_LOG_FILE=query_history.jsonl
```

## Query Logs Format

Each query is logged in JSONL format with:

```json
{
  "timestamp": "2023-05-13T15:32:47.123456",
  "query": "What are the symptoms of concussion?",
  "user_id": "patient_123",
  "results": [
    {
      "id": "1042",
      "score": 0.892,
      "title": "Understanding Concussions and Their Effects",
      "summary": "Concussions are a form of traumatic brain injury...",
      "url": "https://www.powerofpatients.com/blog/understanding-concussions",
      "date": "Mar 30, 2023",
      "author": "Power Of Patients",
      "read_time": "5 min read",
      "categories": "concussion, tbi, symptoms"
    },
    // More results...
  ]
}
```

## Integration with LLMs for Q&A

```python
from retrieval_agent import retrieve_articles
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def answer_patient_question(query, patient_id):
    # Retrieve relevant articles
    results = retrieve_articles(
        query=query, 
        top_k=3, 
        user_id=f"patient_{patient_id}"
    )
    
    # Format the results as context
    context = ""
    for i, result in enumerate(results, 1):
        context += f"Article {i}: {result['title']}\n"
        context += f"Summary: {result['summary']}\n"
        context += f"URL: {result['url']}\n\n"
    
    if not context:
        return "I don't have enough information about that topic yet."
    
    # Create prompt for LLM
    system_msg = """You are a helpful medical assistant specializing in TBI.
    Use ONLY the information from the articles to answer the question.
    Always mention the read time of articles in your response.
    If the articles don't contain enough information, say so."""
    
    user_msg = f"Based on these articles:\n\n{context}\n\nAnswer this patient question: {query}"
    
    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content

# Example usage
answer = answer_patient_question(
    "How long should I rest after a concussion?", 
    patient_id="12345"
)
print(answer)
```

## Importing Query History to Database

```python
import json
import sqlite3

def import_query_logs_to_db(log_file="query_history.jsonl", db_file="tbi_queries.db"):
    # Create/connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        query TEXT,
        user_id TEXT,
        error TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id INTEGER,
        article_id TEXT,
        score REAL,
        title TEXT,
        summary TEXT,
        url TEXT,
        read_time TEXT,
        FOREIGN KEY (query_id) REFERENCES queries (id)
    )
    ''')
    
    # Import data from JSONL file
    with open(log_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            
            # Insert query record
            cursor.execute(
                "INSERT INTO queries (timestamp, query, user_id, error) VALUES (?, ?, ?, ?)",
                (
                    entry["timestamp"],
                    entry["query"],
                    entry.get("user_id"),
                    entry.get("error")
                )
            )
            query_id = cursor.lastrowid
            
            # Insert result records
            for result in entry.get("results", []):
                cursor.execute(
                    """
                    INSERT INTO results 
                    (query_id, article_id, score, title, summary, url, read_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        result["id"],
                        result["score"],
                        result["title"],
                        result["summary"],
                        result["url"],
                        result["read_time"]
                    )
                )
    
    conn.commit()
    conn.close()
    print(f"Imported query logs to {db_file}")

# Example usage
import_query_logs_to_db()
```

## Creating a Flask API

```python
from flask import Flask, request, jsonify
from retrieval_agent import retrieve_articles

app = Flask(__name__)

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data['query']
    top_k = data.get('top_k', 5)
    user_id = data.get('user_id')
    
    results = retrieve_articles(query, top_k, user_id=user_id)
    
    return jsonify({
        "query": query,
        "results": results
    })

@app.route('/api/answer', methods=['POST'])
def answer():
    # Similar implementation as the answer_patient_question function
    # but integrated with the Flask API
    pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```