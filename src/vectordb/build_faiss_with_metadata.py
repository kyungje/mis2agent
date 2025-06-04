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
    return [(1, text)]  # 페이지 번호를 1로 지정한 튜플 리스트 반환

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
def split_text_to_chunks(text_by_page, source_name, chunk_size=200):
    """
    문서를 더 작은 청크로 분할합니다(줄 기준 + 의미 단위 유지 + 공백 정리).
    """
    chunks = []

    for page_num, page_text in text_by_page:
        #print(f"--- 페이지 {page_num} ---")
        #print(page_text[:300])  # 디버깅용: 앞부분 미리보기

        # 1. 줄바꿈 유지하며 줄 단위 분리
        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        # 2. 각 줄의 내부 공백 정리
        lines = [re.sub(r'\s+', ' ', line) for line in lines]

        current_chunk = ""

        for line in lines:
            # 한글-영문 공백 정리
            line = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', line)
            line = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', line)

            if len(current_chunk) + len(line) > chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "source": source_name,
                    "page": page_num
                })
                current_chunk = line
            else:
                current_chunk = current_chunk + " " + line if current_chunk else line

        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "source": source_name,
                "page": page_num
            })

        # 앞부분이 빈 페이지일 때 최소 1개 청크라도 생성하도록 처리
        if not chunks:
            chunks.append({
                "text": page_text[:chunk_size],
                "source": source_name,
                "page": page_num
            })

    print(f"→ 생성된 청크 수: {len(chunks)}")
    return chunks

# === 인덱스 생성 - 배치 처리 ===
# === 벡터 인덱스 구축 ===
def build_langchain_vector_index():
    print("📂 문서 인덱싱 시작")

    # 문서 목록 불러오기
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".docx"))]
    if not files:
        print("❌ 인덱싱할 문서 없음")
        return

    print(f"총 {len(files)}개 문서 발견")

    # 임베딩 모델 초기화 (배치 사이즈 축소로 메모리 절약)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 32}  # 기본은 32 → 16으로 줄여 메모리 사용 감소
    )

    all_chunks = []
    total_pages = 0

    # 문서별로 청크 수 추적 및 로그 출력
    for file_name in files:
        file_path = os.path.join(DOCS_DIR, file_name)
        if file_name.endswith(".pdf"):
            text_by_page = read_pdf(file_path)
        else:
            text_by_page = read_docx(file_path)

        total_pages += len(text_by_page)

        chunks = split_text_to_chunks(text_by_page, file_name)
        all_chunks.extend(chunks)
        print(f" {file_name} → {len(chunks)}개 청크")
        
        # 메모리 최적화
        del text_by_page, chunks
        gc.collect()
        time.sleep(0.1)  # 시스템 여유 주기

    print(f"\n📊 전체 문서 수: {len(files)}개")
    print(f"📄 전체 페이지 수: {total_pages}")
    print(f"🔖 전체 청크 수: {len(all_chunks)}")

    #if len(all_chunks) > 100_000:
    #    print("⚠️ 청크 수가 매우 많습니다. 메모리 부족 가능성이 있으므로 배치 처리를 고려하세요.")

    # LangChain 문서 형식으로 변환
    lc_documents = [
        LCDocument(page_content=chunk["text"], metadata={"source": chunk["source"], "page": chunk["page"]})
        for chunk in all_chunks
    ]

    # 전체 청크로부터 인덱스 생성
    print("\n🔧 인덱스 생성 중...")
    vectorstore = FAISS.from_documents(lc_documents, embedding_model)

   
    # 인덱스 저장
    try:
        vectorstore.save_local(DB_DIR)
    except Exception as e:
        print(f"인덱스 저장 오류: {e}")

    print(f"\n💾 인덱스 저장 완료: {INDEX_FAISS_PATH}")
    print(f"💾 메타데이터 저장 완료: {INDEX_PKL_PATH}")
    print("🎉 모든 문서 인덱싱 완료")


# === 검색 예시 ===
def search_query(query, top_k=5):
    print(f"\n🔍 검색어: {query}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # pickle 파일 로딩 허용
    #vectorstore = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    print("📁 벡터스토어 로딩 중...")
    vectorstore = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    print("✅ 벡터스토어 로딩 완료")

    # 실제 벡터 개수 확인
    if hasattr(vectorstore, "index") and hasattr(vectorstore.index, "ntotal"):
        print(f"🔍 인덱스에 저장된 벡터 수: {vectorstore.index.ntotal}")
    else:
        print("⚠️ 인덱스 정보 없음")

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
