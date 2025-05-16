
# 기존소스 - IndexFlatL2 사용으로 단순한 벡터 인덱스임. LangChain과 포맷오류 발생하여 신규 소스 작성함 build_faiss_with_metadata.py

import os
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
import pickle

# 현재 스크립트 기준 절대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "db")

# db 디렉토리 없으면 생성
os.makedirs(DB_DIR, exist_ok=True)


# 1. 문서 읽기
def read_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def read_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        text += f"\n[Page {page_num}]\n{page_text}"
    return text

# 2. 문서 불러오기
def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            text = read_pdf(file_path)
        elif file.endswith(".docx"):
            text = read_docx(file_path)
        else:
            continue
        documents.append({"filename": file, "text": text})
    return documents

# 3. 문장 단위 분할 (줄바꿈 기준)
def split_documents(documents):
    chunks = []
    for doc in documents:
        lines = [line.strip() for line in doc["text"].split("\n") if len(line.strip()) > 20]
        for line in lines:
            chunks.append({
                "text": line,
                "source": doc["filename"]
            })
    return chunks

# 4. 벡터 DB 구축
def build_vector_index():
    documents = load_documents(DOCS_DIR)
    chunks = split_documents(documents)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # 저장 경로를 db/ 하위로 변경
    faiss_index_path = os.path.join(DB_DIR, "faiss_index.idx")
    chunks_path = os.path.join(DB_DIR, "chunks.pkl")

    faiss.write_index(index, faiss_index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f" FAISS 인덱스 저장 완료: {faiss_index_path}")
    print(f" 문장 메타데이터 저장 완료: {chunks_path}")


    print("🎉 벡터 DB 생성 완료!")

# 5. 검색 예시
def search_query(query, top_k=3):
    # 모델, 인덱스, chunks 불러오기
    print(f"\n🔍 검색어: {query}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(os.path.join(DB_DIR, "faiss_index.idxs"))
    with open(os.path.join(DB_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    print("\n📌 검색 결과:")
    for rank, idx in enumerate(indices[0], start=1):
        print(f"[{rank}] {chunks[idx]['text']} (출처: {chunks[idx]['source']})")

# 메인 실행
if __name__ == "__main__":
    build_vector_index()
    search_query("기준열량이 무엇인가요?")