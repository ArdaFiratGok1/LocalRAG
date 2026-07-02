from flask import Flask, app, jsonify, render_template, request
from werkzeug.utils import secure_filename
import os
import sqlite3
import json
import math
from foundry_local_sdk import Configuration, FoundryLocalManager
from werkzeug.utils import secure_filename


app = Flask(__name__)
DB_NAME = "RAGSqlite.db"
UPLOAD_FOLDER = "temp_uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

embedding_client = None
chat_client = None



def load_and_split(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return []
    files = os.listdir(folder_path)
    chunks = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                paragraphs = content.split("\n\n")

                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if paragraph and not paragraph.startswith("---") and len(paragraph) > 3:
                        chunks.append({ "source": file, "content": paragraph })


    return chunks

def sqlite_cosine_similarity(query_emb_json, db_emb_str):
    vec_a = json.loads(query_emb_json)
    vec_b = json.loads(db_emb_str)
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(a * a for a in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def ingest_documents(db_name, embedding_client, folder_path):
    total_chunks = load_and_split(folder_path)
    
    conn = sqlite3.connect(db_name)
    conn.create_function("COS_SIM", 2, sqlite_cosine_similarity)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   text TEXT NOT NULL,
                   embedding TEXT NOT NULL
        )
    """)
    conn.commit()

    if not total_chunks:
        conn.close()
        print("No documents to index. Database initialized as empty.")
        return

    response = embedding_client.generate_embeddings([chunk['content'] for chunk in total_chunks])
    chunk_embeddings = [item.embedding for item in response.data]
    print(f"Indexed {len(chunk_embeddings)} documents.")

    for chunk, embedding in zip(total_chunks, chunk_embeddings):
        cursor.execute("INSERT INTO documents (text, embedding) VALUES (?, ?)", (chunk['content'], json.dumps(embedding)))

    conn.commit()
    conn.close()

def get_top_k_chunks(db_name, query, embedding_client, top_k=3):

    conn = sqlite3.connect(db_name)
    conn.create_function("COS_SIM",2,sqlite_cosine_similarity)
    cursor = conn.cursor()

    response = embedding_client.generate_embeddings([query])
    query_embedding = response.data[0].embedding #Sorgu vektörünü alıyoruz

    #Sorgu vektörünü SQL'in işlemesi için JSON string'e çeviriyoruz
    query_emb_json = json.dumps(query_embedding)

    cursor.execute("""
        SELECT text, COS_SIM(?, embedding) AS score
                   FROM documents
                   ORDER BY score DESC
                   LIMIT ?""", (query_emb_json,top_k))
    
    
    results=cursor.fetchall()

    return results

def answer_query(db_name, chat_client,contextString, query):

    conn = sqlite3.connect(db_name)
    

    system_prompt=f"""You are a helpful assistant. 
Your task is to answer the user's question using ONLY the provided context below.
If the answer cannot be found in the context, say "I couldn't find the answer in the uploaded documents." Do not make up facts.

---
CONTEXT:
{contextString}
---"""

    """
    results = find_relevant(query_embedding, rows, top_k=3)
    """
    user_prompt = query

    chat_response = chat_client.complete_chat([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
    ])

    return chat_response.choices[0].message.content


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file_api():
    if "files" not in request.files:
        return jsonify({"error": "No file part"}), 400
    uploaded_files = request.files.getlist("files")
    saved_count = 0

    target_folder = r"C:\Users\arda fırat\Desktop\rag_testfiles"##for testing, we keep the uploaded files in a specific folder. You can change this path as needed.
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file in uploaded_files:
        if file.filename == "":
            continue
        filename = secure_filename(file.filename)

        if not filename:
            filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
            
        if not filename:
            filename = "uploaded_doc.txt"

        file_path = os.path.join(target_folder, filename)
        file.save(file_path)
        saved_count += 1

    if saved_count == 0:
        return jsonify({"error": "No valid files uploaded"}), 400
    
    try:
        ingest_documents(DB_NAME, embedding_client, target_folder)
        return jsonify({"message": f"{saved_count} files uploaded and processed successfully."}), 200
    except Exception as e:
        return jsonify({"error": f"Error processing files: {str(e)}"}), 500

    

@app.route("/deleteall", methods=["POST"])
def delete_all_files_api():
    folder_path = r"C:\Users\arda fırat\Desktop\rag_testfiles"
    try:
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS documents")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       text TEXT NOT NULL,
                       embedding TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        return jsonify({"message": "All documents deleted successfully."}), 200
    except Exception as e:
        return jsonify({"message": f"Error occurred while deleting all files: {str(e)}"}), 500
    

@app.route("/delete", methods=["POST"])
def delete_file_api():
    global embedding_client
    data = request.get_json()
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"message": "Filename is required."}), 400
        
    folder_path = r"C:\Users\arda fırat\Desktop\rag_testfiles"
    file_path = os.path.join(folder_path, secure_filename(filename))
    
    if not os.path.exists(file_path):
        file_path = os.path.join(folder_path, filename)

    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            os.remove(file_path)
        else:
            return jsonify({"message": f"File {filename} not found on disk."}), 404
            
        ingest_documents(DB_NAME, embedding_client, folder_path)
        return jsonify({"message": f"File {filename} deleted successfully and database updated."}), 200
    except Exception as e:
        return jsonify({"message": f"Error occurred while deleting file: {str(e)}"}), 500
    
@app.route("/documents", methods=["GET"])
def documents_api():
    folder_path = r"C:\Users\arda fırat\Desktop\rag_testfiles"
    if os.path.exists(folder_path):
        try:
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            return jsonify({"documents": files})
        except Exception as e:
            return jsonify({"documents": [], "error": str(e)})
    else:
        return jsonify({"documents": ["aslan.txt", "fil.txt", "kutup_Ayisi.txt", "yunus.txt", "zurafa.txt"]})


@app.route("/ask", methods=["POST"])
def ask_api():
    global embedding_client, chat_client

    data = request.get_json()
    user_query = data.get("question", "")

    results = get_top_k_chunks(DB_NAME, user_query, embedding_client, top_k=3)

    contextString = "\n\n".join([text for text, score in results])
    answer = answer_query(DB_NAME, chat_client, contextString, user_query)

    return jsonify({"answer": answer, "references": results})



if __name__ == "__main__":
    config = Configuration(app_name="localRAG")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    #Load the embedding model
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()


    if not os.path.exists(DB_NAME):
        print("Database not found. Ingesting documents...")
        ingest_documents(DB_NAME, embedding_client, r"C:\Users\arda fırat\Desktop\rag_testfiles")


    chat_model = manager.catalog.get_model("qwen3.5-2b-text")
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )

    chat_model.load()
    chat_client = chat_model.get_chat_client()

    app.run(host="127.0.0.1", port=5000, debug=False)

