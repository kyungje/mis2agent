import os
import fitz  # PyMuPDF
from docx import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument
import gc
import time
import re

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
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text

def read_pdf(path):
    doc = fitz.open(path)
    text_by_page = []
    
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
                    block_text = normalize_korean_text(block_text)
                    page_text += block_text + "\n"
                    
            text_by_page.append((page_num, page_text))
        except Exception as e:
            print(f"페이지 {page_num} 처리 중 오류: {str(e)}")
    
    # 메모리 해제
    doc.close()
    return text_by_page

def normalize_korean_text(text):
    """한글 텍스트 정규화: 띄어쓰기 및 문장 구조 개선"""
    import re
    
    # 1. 기본 정규화
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. 한글 단어 사이에 띄어쓰기 추가
    # 한글 자음+모음 패턴 (가-힣)
    text = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', text)  # 한글 뒤에 영숫자
    text = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', text)  # 영숫자 뒤에 한글
    
    # 3. 붙어있는 한글 단어들 사이에 띄어쓰기 추가 (패턴 기반)
    patterns = [
        (r'([.!?])([가-힣])', r'\1 \2'),  # 문장 부호 뒤에 띄어쓰기
        (r'([가-힣])([(){}[\]<>])', r'\1 \2'),  # 한글과 괄호 사이
        (r'([(){}[\]<>])([가-힣])', r'\1 \2'),  # 괄호와 한글 사이
    ]
    
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    
    # 4. 특수 패턴 처리 (예: 가스공급규정 -> 가스 공급 규정)
    # 6글자 이상 연속된 한글을 확인하여 3~4글자 단위로 분리 시도
    def split_long_korean(match):
        long_word = match.group(0)
        if len(long_word) >= 6:
            # 긴 단어를 3글자 단위로 분리 시도
            parts = []
            i = 0
            while i < len(long_word):
                if i + 3 <= len(long_word):
                    parts.append(long_word[i:i+3])
                else:
                    parts.append(long_word[i:])
                i += 3
            return ' '.join(parts)
        return long_word
    
    text = re.sub(r'[가-힣]{6,}', split_long_korean, text)
    
    return text

# === 문서 불러오기 ===
def load_document(file_path):
    """단일 문서를 로드합니다."""
    if file_path.endswith(".pdf"):
        return read_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = read_docx(file_path)
        # DOCX는 페이지 정보가 없으므로 하나의 페이지로 취급
        return [(1, text)]
    return []

# === 문장 분할 - 간소화 버전 ===
def split_text_to_chunks(text_by_page, source_name, chunk_size=100):
    """
    문서를 더 작은 청크로 분할합니다(단순화된 버전).
    """
    chunks = []
    
    for page_num, page_text in text_by_page:
        # 텍스트 정규화: 불필요한 공백 제거 및 줄바꿈 표준화
        page_text = re.sub(r'\s+', ' ', page_text)
        
        # 줄바꿈으로 분할
        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        
        current_chunk = ""
        
        # 각 줄을 처리
        for line in lines:
            # 줄 정규화: 단어 사이에 공백이 없는 경우 추가
            line = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', line)  # 한글 뒤에 영숫자
            line = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', line)  # 영숫자 뒤에 한글
            
            # 현재 줄 추가 시 chunk_size를 초과하는지 확인
            if len(current_chunk) + len(line) > chunk_size and current_chunk:
                # 청크 추가 후 새 청크 시작
                chunks.append({
                    "text": current_chunk,
                    "source": source_name,
                    "page": page_num
                })
                current_chunk = line
            else:
                # 현재 청크에 줄 추가
                if current_chunk:
                    current_chunk += " " + line
                else:
                    current_chunk = line
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "source": source_name,
                "page": page_num
            })
    
    return chunks

# === 인덱스 생성 - 배치 처리 ===
def build_langchain_vector_index():
    print("📂 문서 로딩 및 인덱싱 시작...")
    
    # 문서 목록 가져오기
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.pdf', '.docx'))]
    if not files:
        print("❌ docs 폴더에 유효한 문서가 없습니다.")
        return
    
    print(f"총 {len(files)}개 문서 발견")
    
    # 임베딩 모델 초기화
    print("임베딩 모델 초기화 중...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 기존 인덱스 존재 시 로드, 없으면 새로 생성
    if os.path.exists(os.path.join(DB_DIR, "index.faiss")):
        print("기존 인덱스 로드 중...")
        vectorstore = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("새 인덱스 생성 중...")
        # 빈 문서 리스트로 초기화
        vectorstore = FAISS.from_documents([
            LCDocument(page_content="초기화 문서", metadata={"source": "초기화", "page": 0})
        ], embedding_model)
    
    # 파일별로 처리 (배치 처리)
    total_chunks = 0
    for i, file_name in enumerate(files, 1):
        print(f"\n처리 중: {file_name} ({i}/{len(files)})")
        file_path = os.path.join(DOCS_DIR, file_name)
        
        # 단일 문서 로드
        text_by_page = load_document(file_path)
        print(f"  - {len(text_by_page)}페이지 로드됨")
        
        # 메모리 관리를 위해 텍스트를 청크로 분할
        chunks = split_text_to_chunks(text_by_page, file_name, chunk_size=100)
        print(f"  - {len(chunks)}개 청크로 분할됨")
        total_chunks += len(chunks)
        
        # 메모리 최적화: 텍스트 페이지 데이터 해제
        del text_by_page
        gc.collect()
        
        # 청크를 작은 배치로 처리
        batch_size = 50  # 한 번에 처리할 청크 수
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            current_batch = chunks[batch_start:batch_end]
            
            print(f"  - 배치 처리 중: {batch_start+1}~{batch_end}/{len(chunks)}")
            
            # 현재 배치를 LangChain 문서로 변환
            lc_documents = [
                LCDocument(
                    page_content=chunk["text"],
                    metadata={"source": chunk["source"], "page": chunk["page"]}
                )
                for chunk in current_batch
            ]
            
            # 기존 인덱스에 추가
            vectorstore.add_documents(lc_documents)
            
            # 메모리 최적화
            del lc_documents
            gc.collect()
            
            # 잠시 대기하여 시스템에 여유 부여
            time.sleep(0.1)
        
        # 현재 파일 처리 완료 후 인덱스 저장
        print("  - 현재까지의 인덱스 저장 중...")
        vectorstore.save_local(DB_DIR)
        
        # 배치 처리 후 메모리 정리
        del chunks
        gc.collect()
    
    print(f"\n✅ 모든 문서 처리 완료. 총 {total_chunks}개 청크가 인덱싱됨")
    print(f"💾 최종 인덱스 저장 완료 → {INDEX_FAISS_PATH}")
    print(f"💾 메타데이터 저장 완료 → {INDEX_PKL_PATH}")
    print("🎉 LangChain 호환 벡터 DB 생성 완료")

# === 검색 예시 ===
def search_query(query, top_k=5):
    print(f"\n🔍 검색어: {query}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # pickle 파일 로딩 허용
    vectorstore = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)

    results = vectorstore.similarity_search(query, k=top_k)

    print("\n📌 검색 결과:")
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] {doc.page_content}")
        
        # 메타데이터 안전하게 출력
        source = doc.metadata.get('source', '알 수 없음')
        page = doc.metadata.get('page', '알 수 없음')
        print(f"     ⤷ 출처: {source}, 페이지: {page}")
        print("-" * 60)


# === 메인 실행 ===
if __name__ == "__main__":
    build_langchain_vector_index()
    
    # 인덱스 생성 완료 후 예시 검색 수행
    search_query("도시가스 공급 규정에서 도시가스회사의 의무는 무엇인가요?")
    search_query("가스요금은 어떻게 계산되나요?")
