import os
import fitz  # PyMuPDF
from docx import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "db")
INDEX_FAISS_PATH = os.path.join(DB_DIR, "index.faiss")
INDEX_PKL_PATH = os.path.join(DB_DIR, "index.pkl")

# 필요한 폴더 생성
os.makedirs(DB_DIR, exist_ok=True)

# === 문서 읽기 ===
def read_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def read_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        text += f"\n[Page {page_num}]\n" + page.get_text()
    return text

# === 문서 불러오기 ===
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

# === 문장 분할 ===
def split_documents(documents):
    chunks = []
    for doc in documents:
        lines = [line.strip() for line in doc["text"].split("\n") if len(line.strip()) > 20]
        for line in lines:
            chunks.append({"text": line, "source": doc["filename"]})
    return chunks

# === 인덱스 생성 ===
def build_langchain_vector_index():
    print("📂 문서 로딩 중...")
    documents = load_documents(DOCS_DIR)
    if not documents:
        print("❌ docs 폴더에 유효한 문서가 없습니다.")
        return

    chunks = split_documents(documents)
    if not chunks:
        print("❌ 문장에서 유의미한 텍스트를 추출하지 못했습니다.")
        return

    lc_documents = [
        LCDocument(page_content=chunk["text"], metadata={"source": chunk["source"]})
        for chunk in chunks
    ]

    print(f"✅ 문서에서 총 {len(lc_documents)}개의 문장 분할 및 임베딩 준비 완료")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(lc_documents, embedding_model)
    vectorstore.save_local(DB_DIR)

    print(f"💾 인덱스 저장 완료 → {INDEX_FAISS_PATH}")
    print(f"💾 메타데이터 저장 완료 → {INDEX_PKL_PATH}")
    print("🎉 LangChain 호환 벡터 DB 생성 완료")

# === 검색 예시 ===
def search_query(query, top_k=3):
    print(f"\n🔍 검색어: {query}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # ✅ pickle 파일 로딩 허용
    vectorstore = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)

    results = vectorstore.similarity_search(query, k=top_k)

    print("\n📌 검색 결과:")
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] {doc.page_content}")
        print(f"     ⤷ 출처: {doc.metadata['source']}")
        print("-" * 60)


# === 메인 실행 ===
if __name__ == "__main__":
    build_langchain_vector_index()
    search_query("기준열량이 무엇인가요?")
