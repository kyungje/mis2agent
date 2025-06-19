import os
import fitz  # PyMuPDF
from docx import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gc
import time
import re
from typing import List
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# === 경로 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "db")
INDEX_FAISS_PATH = os.path.join(DB_DIR, "index.faiss")
INDEX_PKL_PATH = os.path.join(DB_DIR, "index.pkl")

# 필요한 폴더 생성
os.makedirs(DB_DIR, exist_ok=True)

# === OpenAI 임베딩 모델 초기화 ===
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",  # OpenAI 임베딩 모델
    chunk_size=1000  # 배치 크기
)

# === 텍스트 분할기 초기화 ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# === 문서 읽기 ===
def read_docx(path):
    """DOCX 문서를 읽어서 텍스트를 추출합니다."""
    doc = Document(path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    print(f"[DOCX] {path} 추출 텍스트 예시:", text[:100])
    return text

def read_pdf(path):
    """PDF 문서를 읽어서 전체 텍스트를 추출합니다."""
    doc = fitz.open(path)
    full_text = ""
    
    # 페이지별로 처리
    for page_num, page in enumerate(doc, start=1):
        try:
            # 보다 구조화된 텍스트 추출 방식으로 변경
            blocks = page.get_text("blocks")
            page_text = ""
            for block in blocks:
                if block[6] == 0:  # 텍스트 블록만 선택 (이미지 블록 제외)
                    # 블록 텍스트에 공백 추가
                    block_text = block[4]
                    # 한글 텍스트 정규화
                    block_text = normalize_text(block_text)
                    page_text += block_text + "\n"
                    
            full_text += page_text + "\n"
        except Exception as e:
            print(f"페이지 {page_num} 처리 중 오류: {str(e)}")
    
    # 메모리 해제
    doc.close()
    if full_text:
        print(f"[PDF] {path} 텍스트 예시:", full_text[:100])
    return full_text

# === 텍스트 정규화 ===
def normalize_text(text):
    """텍스트 정규화 (한글-영문 공백, 특수문자 정리)"""
    # 한글-영문 공백 정리
    text = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', text)
    
    # 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    # 특수문자 정리 (문장부호, 괄호 등 유지)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    
    return text.strip()

# === 문서 처리 및 청킹 ===
def process_document(file_path):
    """문서를 읽고 LangChain 텍스트 분할기를 사용하여 청킹합니다."""
    filename = os.path.basename(file_path)
    
    # 문서 읽기
    if file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = read_docx(file_path)
    else:
        return []
    
    # 텍스트 정규화
    text = normalize_text(text)
    
    # LangChain 텍스트 분할기 사용
    chunks = text_splitter.split_text(text)
    
    # 메타데이터가 포함된 LangChain 문서로 변환
    documents = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 50:  # 최소 길이 필터 (50자 이상)
            doc = LCDocument(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "chunk_id": i,
                    "file_path": file_path
                }
            )
            documents.append(doc)
    
    print(f"[CHUNKS] {filename} → {len(documents)}개 생성")
    return documents

# === 벡터 인덱스 구축 ===
def build_langchain_vector_index():
    """문서들을 읽어서 벡터 인덱스를 구축합니다."""
    print("📂 문서 인덱싱 시작")

    # 문서 목록 불러오기
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".docx"))]
    if not files:
        print("❌ 인덱싱할 문서 없음")
        return

    print(f"총 {len(files)}개 문서 발견")

    all_documents = []
    total_files = len(files)

    # 문서별로 처리
    for i, file_name in enumerate(files, 1):
        file_path = os.path.join(DOCS_DIR, file_name)
        print(f"[{i}/{total_files}] 처리 중: {file_name}")
        
        documents = process_document(file_path)
        all_documents.extend(documents)
        
        # 메모리 최적화
        del documents
        gc.collect()
        time.sleep(0.1)  # 시스템 여유 주기

    print(f"\n📊 전체 문서 수: {len(files)}개")
    print(f"🔖 전체 청크 수: {len(all_documents)}")

    if len(all_documents) == 0:
        print("경고: 인덱싱할 문서가 없습니다!")
        return

    print(f"최종 인덱싱 대상 문서 수: {len(all_documents)}")
    print("예시 문서:", all_documents[0].page_content[:200])

    # 전체 청크로부터 인덱스 생성
    print("\n🔧 인덱스 생성 중...")
    vectorstore = FAISS.from_documents(all_documents, embedding_model)

    # 인덱스 저장
    try:
        vectorstore.save_local(DB_DIR)
    except Exception as e:
        print(f"인덱스 저장 오류: {e}")

    print(f"\n💾 인덱스 저장 완료: {INDEX_FAISS_PATH}")
    print(f"💾 메타데이터 저장 완료: {INDEX_PKL_PATH}")
    print("🎉 모든 문서 인덱싱 완료")

# === 검색 시스템 ===
def search_query(query, top_k=10):
    """벡터 인덱스에서 유사한 문서를 검색합니다."""
    print(f"\n🔍 검색어: {query}")
    
    vectorstore = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    print("✅ 벡터스토어 로딩 완료")
    print(f"총 벡터 수: {vectorstore.index.ntotal}")

    # 검색 결과를 더 많이 가져와서 정렬
    results = vectorstore.similarity_search_with_score(query, k=20)
    results = sorted(results, key=lambda x: x[1])  # 낮은 점수일수록 유사도 높음

    print("\n📌 검색 결과:")
    for i, (doc, score) in enumerate(results[:top_k], start=1):
        print(f"[{i}] 점수: {score:.4f}")
        # 검색어 주변 컨텍스트를 포함하여 출력
        content = doc.page_content
        if query in content:
            idx = content.find(query)
            start = max(0, idx - 150)  # 컨텍스트 범위 확장
            end = min(len(content), idx + 150)  # 컨텍스트 범위 확장
            print(f"...{content[start:end]}...")
        else:
            # 검색어가 없는 경우, 문장 단위로 출력
            sentences = content.split('.')
            print('. '.join(sentences[:3]) + '.')  # 처음 3개 문장만 출력
        print(f"     ⤷ 출처: {doc.metadata.get('source')} / 청크 ID: {doc.metadata.get('chunk_id')}")
        print("-" * 60)

# === 메인 실행 ===
if __name__ == "__main__":
    build_langchain_vector_index()
    
    # 인덱스 생성 완료 후 예시 검색 수행
    #search_query("도시가스 공급 규정에서 도시가스회사의 의무는 무엇인가요?")
    #search_query("가스요금은 어떻게 계산되나요?")
    search_query("기준열량")
    #search_query("최고열량")
