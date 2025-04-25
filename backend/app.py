from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import os
import json
from Tagger import tagger, search_and_load_json
from caseSummarizer import summarizer, chunk_text

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

# ✅ Load API Key
API_KEY = os.getenv("TOGETHER_AI_KEY", "37d7a04f4f855143791edb2733d20b461f460c8a50ce75407210d049db58649e")

# ✅ Memory Handling
MEMORY_FILE = "backend/memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def append_to_memory(user_query, ipc_text, ai_response):
    memory = load_memory()
    memory.append({
        "user": user_query,
        "ipc": ipc_text,
        "response": ai_response
    })
    save_memory(memory)
    return memory[-6:]  # Latest 6 entries for context

# ✅ FAISS Search
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

# ✅ Together AI Query Function
def query_together_ai(query: str, ipc_text: str, memory_context: list):
    messages = [{"role": "system", "content": "You are a legal assistant providing answers based on the Indian Penal Code (IPC)."}]

    for entry in memory_context:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["response"]})

    messages.append({
        "role": "user",
        "content": f"Here is some IPC context:\n{ipc_text}\n\nNow answer the user's query: {query}"
    })

    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": messages,
        "max_tokens": 1000,
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

# ✅ Chat Endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    ipc_text = search_ipc(user_query)

    # Append to memory with placeholder response first
    memory = load_memory()
    memory.append({"user": user_query, "ipc": ipc_text, "response": ""})
    save_memory(memory)

    memory_context = memory[-6:-1]  # Get last 5 previous messages for context

    ai_response = query_together_ai(user_query, ipc_text, memory_context)

    # Update the last memory entry with actual response
    memory[-1]["response"] = ai_response
    save_memory(memory)

    return jsonify({
        "query": user_query,
        "ipc_sections": ipc_text,
        "response": ai_response
    })

# ✅ Tagging & Case Search
@app.route("/get_case", methods=["POST"])
def get_case():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    tags = tagger(user_query)

    root_folder = r"C:\Users\S Sri Hari\Major_project\final\frontend\chatbot-ui\backend\criminal law -IPC"
    result = search_and_load_json(tags, root_folder)

    return jsonify({
        "query": user_query,
        "tags": tags,
        "results": result
    })

# ✅ Case Summarization
@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Failed to parse JSON: {str(e)}"}), 400

    auth = request.headers.get("Authorization")

    if not auth or not auth.startswith("hf_"):
        return jsonify({"error": "Unauthorized"}), 401

    inputs = data.get("inputs")
    if not inputs:
        return jsonify({"error": "Missing 'inputs' field"}), 400

    final_summary = summarizer(inputs)
    return jsonify({"summary": final_summary})

# ✅ Run Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
