from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import os

app = Flask(__name__)
CORS(app)

# ✅ Load FAISS Index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.load_local(
    r"C:\Users\S Sri Hari\Major_project\final\frontend\chatbot-ui\faiss",
    embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ FAISS Index Loaded Successfully!")

# ✅ Load API Key (Either Hardcoded or from Environment)
API_KEY = os.getenv("TOGETHER_AI_KEY", "")  # Change the second argument to your actual key

# ✅ FAISS Search Function
def search_ipc(query: str, top_k: int = 3):
    results = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": top_k}).get_relevant_documents(query)
    
    seen = set()
    unique_results = []
    for doc in results:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_results.append(content)

    formatted_output = "**📌 Relevant IPC Sections:**\n\n"
    for i, result in enumerate(unique_results, 1):
        formatted_output += f"🔹 **Section {i}:** {result}\n\n"

    return formatted_output.strip()

# ✅ Function to Query Together AI
def query_together_ai(query: str, ipc_text: str):
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [
            {"role": "system", "content": "You are a legal assistant providing answers based on the Indian Penal Code (IPC)."},
            {"role": "user", "content": f"Here is relevant IPC information:\n{ipc_text}\n\nNow answer the user's query: {query}"}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.together.xyz/v1/chat/completions", json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.text}"

# ✅ Flask API Endpoint (DOES NOT Expect API Key)
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    ipc_text = search_ipc(user_query)
    ai_response = query_together_ai(user_query, ipc_text)

    return jsonify({
        "query": user_query,
        "ipc_sections": ipc_text,
        "response": ai_response
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
