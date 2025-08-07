# chat_ui_docinsight.py (DocInsight AI 스타일 적용 전체 버전)
import streamlit as st
import requests
import time
import re
import os
from dotenv import load_dotenv

# 문서 업로드 관련 추가 import
import json
import pandas as pd
from typing import List, Dict, Any
import tempfile
import shutil
from pathlib import Path
import unicodedata

# 문서 처리 및 벡터 인덱스 생성을 위한 모듈들
import pdfplumber
from docx import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LCDocument
from langchain_experimental.text_splitter import SemanticChunker
import gc

# 페이지 설정
st.set_page_config(page_title="DocInsight AI", page_icon="📄", layout="wide")
load_dotenv()
API_URL = "http://localhost:8000/chat"
RELOAD_API_URL = "http://localhost:8000/reload-indexes"

# === 문서 업로드를 위한 함수들 (chat_ui.py에서 복사) ===

# 경로 설정
BASE_DIR = Path(__file__).parent.parent / "vectordb"
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "db"

# 인덱스 분류별 경로 설정
GAS_INDEX_DIR = DB_DIR / "gas_index"
POWER_INDEX_DIR = DB_DIR / "power_index"
OTHER_INDEX_DIR = DB_DIR / "other_index"

# 필요한 폴더 생성
DB_DIR.mkdir(exist_ok=True)
GAS_INDEX_DIR.mkdir(exist_ok=True)
POWER_INDEX_DIR.mkdir(exist_ok=True)
OTHER_INDEX_DIR.mkdir(exist_ok=True)

# OpenAI 임베딩 모델 초기화
def create_embedding_model():
    """동적으로 배치 크기를 조정하여 임베딩 모델을 생성합니다."""
    chunk_size = 1000
    
    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            chunk_size=chunk_size,
            max_retries=3
        )
        return embedding_model
    except Exception as e:
        if "max_tokens_per_request" in str(e):
            chunk_size = 500
            st.info(f"토큰 제한으로 인해 배치 크기를 {chunk_size}로 조정합니다.")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                chunk_size=chunk_size,
                max_retries=3
            )
        else:
            raise e

# 텍스트 분할기 초기화 - SemanticChunker 사용
def create_text_splitter():
    """SemanticChunker를 생성합니다."""
    embedding_model = create_embedding_model()
    text_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",  # 기본값
        buffer_size=1
    )
    return text_splitter

# 문서 읽기 함수들
def read_docx(path):
    """DOCX 문서를 읽어서 텍스트를 추출합니다."""
    doc = Document(path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    text = latex_to_text(text)
    return text

def read_pdf(path):
    """PDF 문서를 읽어서 전체 텍스트를 추출합니다."""
    full_text = ""
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text:
                    page_text = latex_to_text(page_text)
                    page_text = normalize_text(page_text)
                    full_text += page_text + "\n"
            except Exception as e:
                st.warning(f"페이지 {page_num} 처리 중 오류: {str(e)}")
    return full_text

def read_txt(path):
    """TXT 파일을 읽어서 텍스트를 추출합니다."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = latex_to_text(text)
    return text

# 텍스트 정규화
def normalize_text(text):
    """텍스트 정규화 (한글-영문 공백, 수식 기호 보존)"""
    # 한글-영문 공백 정리
    text = re.sub(r'([가-힣])([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z0-9])([가-힣])', r'\1 \2', text)

    # 수식 기호 보존: √ ± ≈ ∞ × ÷ π ² ³ ^ / = % 등
    math_symbols = "√±≈∞×÷π²³^=/%"

    # 더 관대한 정규화: 한글, 영문, 숫자, 공백, 문장부호, 수식 기호만 유지
    # 한글: 가-힣
    # 영문: A-Za-z
    # 숫자: 0-9
    # 공백: \s
    # 문장부호: .,!?;:-()[]{}
    # 수식 기호: √±≈∞×÷π²³^=/%
    # 추가 허용 문자: + (플러스 기호)
    
    allowed_pattern = r'[가-힣A-Za-z0-9\s.,!?;:\-\(\)\[\]\{\}' + re.escape(math_symbols + '+') + r']'
    text = re.sub(rf'[^{allowed_pattern}]', ' ', text)

    # 연속 공백 정리
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# 파일명 정규화 함수
def normalize_filename(file_name):
    """파일명을 정규화하여 분류에 사용합니다."""
    # 파일 확장자 제거
    name_without_ext = os.path.splitext(file_name)[0]
    
    # 특수문자 제거 (하이픈, 언더스코어, 플러스 등은 공백으로 변환)
    # 대괄호는 제거하되, 키워드가 포함된 부분은 보존
    normalized = re.sub(r'[\[\]\(\)\+\-\_]', ' ', name_without_ext)
    
    # 한글이 분해되지 않도록 NFC로 정규화
    normalized = unicodedata.normalize('NFC', normalized)
    
    # 키워드가 포함된 부분이 공백으로 분리되었는지 확인하고 복구
    if ' 전력 ' in normalized or normalized.startswith('전력 ') or normalized.endswith(' 전력'):
        # 전력 키워드 주변의 공백을 제거하여 복구
        normalized = re.sub(r'\s+전력\s+', '전력', normalized)
        normalized = re.sub(r'^전력\s+', '전력', normalized)
        normalized = re.sub(r'\s+전력$', '전력', normalized)
    
    if ' 도시가스 ' in normalized or normalized.startswith('도시가스 ') or normalized.endswith(' 도시가스'):
        # 도시가스 키워드 주변의 공백을 제거하여 복구
        normalized = re.sub(r'\s+도시가스\s+', '도시가스', normalized)
        normalized = re.sub(r'^도시가스\s+', '도시가스', normalized)
        normalized = re.sub(r'\s+도시가스$', '도시가스', normalized)
    
    # 연속 공백을 단일 공백으로 변환
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 앞뒤 공백 제거
    normalized = normalized.strip()
    
    return normalized

# 파일 분류 함수
def classify_file(file_name):
    """파일명에 따라 분류를 결정합니다."""
    # 파일명 정규화
    normalized_name = normalize_filename(file_name)
    
    # 키워드 검색 결과 (여러 방법으로 검색)
    gas_found = '도시가스' in normalized_name
    power_found = '전력' in normalized_name
    
    # 대안 검색 방법 (인코딩 문제 해결을 위해)
    gas_found_alt = normalized_name.find('도시가스') != -1
    power_found_alt = normalized_name.find('전력') != -1
    
    # 최종 결과 결정 (둘 중 하나라도 True면 True)
    gas_found = gas_found or gas_found_alt
    power_found = power_found or power_found_alt
    
    # 분류 결정
    if gas_found:
        return 'gas'
    elif power_found:
        return 'power'
    else:
        # 원본 파일명으로 재시도
        original_gas_found = '도시가스' in file_name
        original_power_found = '전력' in file_name
        
        if original_gas_found:
            return 'gas'
        elif original_power_found:
            return 'power'
        else:
            # 마지막 안전장치: 파일명의 모든 부분을 개별적으로 확인
            file_parts = re.split(r'[_\-\s\[\]\(\)]', file_name)
            
            for part in file_parts:
                if '도시가스' in part:
                    return 'gas'
                elif '전력' in part:
                    return 'power'
            
            return 'other'

def get_index_dir(category):
    """분류에 따른 인덱스 디렉토리를 반환합니다."""
    if category == 'gas':
        return GAS_INDEX_DIR
    elif category == 'power':
        return POWER_INDEX_DIR
    else:
        return OTHER_INDEX_DIR

def extract_metadata_from_filename(filename):
    base_name = os.path.splitext(filename)[0]
    parts = re.split(r'[_\-\s\+\(\)]', base_name)  # +와 ()도 구분자로 추가

    # 버전 추출 개선: 2024, 24.01.01, 2024.7.1 등 다양한 형식 지원
    version = None
    for part in parts:
        # 4자리 연도 포함 날짜 (2024.7.1) - 우선순위 1
        if re.match(r'20\d{2}\.\d{1,2}\.\d{1,2}', part):
            version = part  # 전체 날짜 보존
            break
        # 2자리 연도 포함 날짜 (24.01.01) - 우선순위 2
        elif re.match(r'\d{2}\.\d{1,2}\.\d{1,2}', part):
            year = part.split('.')[0]
            if int(year) >= 20:  # 20년 이후
                version = f"20{part}"  # 2024.01.01 형태로 변환
            break
        # 4자리 연도만 (2024) - 우선순위 3
        elif re.match(r'20\d{2}$', part):
            version = part
            break
    
    region_list = ['서울', '부산', '대구', '광주', '인천', '대전', '울산','경기도', '강원도', '충청북도', '충청남도' ,'전라남도','전북특별자치도', '경상남도', '경상북도']
    region = next((p for p in parts if p in region_list), None)
    organization_map = {'도시가스': '도시가스', '전력': '전력'}
    organization = next((p for p in organization_map if p in parts), "기타")

    return {
        "version": version,
        "region": region,
        "organization": organization,
        "title": base_name
    }

# 문서 처리 및 청킹
def process_document(file_path):
    """문서를 읽고 SemanticChunker를 사용하여 청킹합니다."""
    filename = os.path.basename(file_path)
    
    # 문서 읽기
    if file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = read_docx(file_path)
    elif file_path.endswith(".txt"):
        text = read_txt(file_path)
    else:
        return []
    
    # 텍스트가 비어있으면 처리 중단
    if not text:
        st.warning(f"⚠️ {filename}에서 텍스트를 추출하지 못했습니다. 건너뜁니다.")
        return []
    
    # 텍스트 정규화
    original_text = text
    text = normalize_text(text)
    
    # 정규화 과정에서 텍스트가 너무 많이 제거되었는지 확인
    if len(text) < len(original_text) * 0.1:  # 90% 이상 제거된 경우
        st.warning(f"  ⚠️ 텍스트가 너무 많이 제거되었습니다. 원본: {len(original_text)}자 → 정규화: {len(text)}자")
        # 정규화를 건너뛰고 원본 텍스트 사용
        text = original_text
    
    # SemanticChunker 사용
    text_splitter = create_text_splitter()
    chunks = text_splitter.split_text(text)
    
    # 파일명에서 추가 메타데이터 추출
    additional_metadata = extract_metadata_from_filename(filename)
    
    # 메타데이터가 포함된 LangChain 문서로 변환
    documents = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 50:  # 최소 길이 필터 (50자 이상)
            # 기본 메타데이터와 추가 메타데이터를 결합
            metadata = {
                "source": filename,
                "chunk_id": i,
                "file_path": file_path,
                "version": additional_metadata["version"],
                "region": additional_metadata["region"],
                "organization": additional_metadata["organization"],
                "title": additional_metadata["title"]
            }
            doc = LCDocument(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
    
    return documents

# 벡터 인덱스 구축
def build_vector_index_from_uploaded_files(uploaded_files):
    """업로드된 파일들로부터 벡터 인덱스를 구축합니다."""
    if not uploaded_files:
        st.warning("업로드된 파일이 없습니다.")
        return False
    
    # 단계별 메시지를 위한 플레이스홀더들
    main_status_placeholder = st.empty()
    file_status_placeholder = st.empty()
    index_status_placeholder = st.empty()
    
    main_status_placeholder.info("📂 문서 인덱싱 시작")
    
    # 문서 저장 디렉토리 생성
    docs_dir = Path(__file__).parent.parent.parent / "vectordb" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # 임베딩 모델 초기화
    embedding_model = create_embedding_model()
    
    # 분류별로 문서 그룹화
    gas_documents = []
    power_documents = []
    other_documents = []
    
    total_files = len(uploaded_files)
    
    # 파일별로 처리 및 분류
    for i, uploaded_file in enumerate(uploaded_files, 1):
        file_status_placeholder.info(f"[{i}/{total_files}] 처리 중: {uploaded_file.name}")
        
        # 실제 파일을 docs 디렉토리에 저장
        file_path = docs_dir / uploaded_file.name
        
        # 이미 동일한 파일명이 존재하는지 확인
        if file_path.exists():
            file_status_placeholder.empty()
            main_status_placeholder.empty()
            st.warning(f"⚠️ '{uploaded_file.name}' 파일이 이미 존재합니다. 인덱스 생성을 건너뜁니다.")
            return False
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # 파일 분류
            category = classify_file(uploaded_file.name)
            
            documents = process_document(str(file_path))
            
            # 분류별로 문서 추가
            if category == 'gas':
                gas_documents.extend(documents)
            elif category == 'power':
                power_documents.extend(documents)
            else:
                other_documents.extend(documents)
            
            # 메모리 최적화
            del documents
            gc.collect()
            
        except Exception as e:
            file_status_placeholder.empty()
            main_status_placeholder.empty()
            st.error(f"❌ 파일 처리 오류: {e}")
            # 오류 발생 시 저장된 파일 삭제
            if file_path.exists():
                file_path.unlink()
            continue
    
    # 파일 처리 완료 메시지 삭제
    file_status_placeholder.empty()
    
    # 분류별로 인덱스 생성
    categories = [
        ('gas', gas_documents, 'Gas'),
        ('power', power_documents, 'Power'),
        ('other', other_documents, 'Other')
    ]
    
    success_count = 0
    for category, documents, category_name in categories:
        if len(documents) == 0:
            continue
            
        index_status_placeholder.info(f"🔧 {category_name} 인덱스 생성 중... (문서 수: {len(documents)}개)")
        
        # 배치 단위로 임베딩 처리
        try:
            vectorstore = FAISS.from_documents(documents, embedding_model)
        except Exception as e:
            if "max_tokens_per_request" in str(e):
                try:
                    medium_embedding_model = OpenAIEmbeddings(
                        model="text-embedding-3-small",
                        chunk_size=500,
                        max_retries=3
                    )
                    vectorstore = FAISS.from_documents(documents, medium_embedding_model)
                except Exception as e2:
                    if "max_tokens_per_request" in str(e2):
                        small_embedding_model = OpenAIEmbeddings(
                            model="text-embedding-3-small",
                            chunk_size=100,
                            max_retries=3
                        )
                        vectorstore = FAISS.from_documents(documents, small_embedding_model)
                    else:
                        raise e2
            else:
                raise e

        # 인덱스 저장
        index_dir = get_index_dir(category)
        try:
            vectorstore.save_local(str(index_dir))
            index_status_placeholder.success(f"💾 {category_name} 인덱스 저장 완료: {index_dir}")
            success_count += 1
        except Exception as e:
            index_status_placeholder.empty()
            main_status_placeholder.empty()
            st.error(f"❌ {category_name} 인덱스 저장 오류: {e}")
            return False
    
    # 모든 단계 완료 후 플레이스홀더들 정리
    main_status_placeholder.empty()
    index_status_placeholder.empty()
    
    if success_count > 0:
        st.success(f"🎉 {success_count}개 분류별 인덱스 생성 완료")
        return True
    else:
        st.error("❌ 인덱스 생성에 실패했습니다.")
        return False

def reload_backend_indexes():
    """백엔드의 인덱스를 다시 로드합니다."""
    # 로딩 메시지를 위한 플레이스홀더 생성
    loading_placeholder = st.empty()
    
    try:
        loading_placeholder.info("🔄 백엔드 인덱스 리로드 중...")
        response = requests.post(RELOAD_API_URL)
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            loading_placeholder.empty()  # 로딩 메시지 삭제
            st.success("✅ 백엔드 인덱스 리로드 완료!")
            return True
        else:
            loading_placeholder.empty()  # 로딩 메시지 삭제
            st.error(f"❌ 백엔드 인덱스 리로드 실패: {result.get('message', '알 수 없는 오류')}")
            return False
    except Exception as e:
        loading_placeholder.empty()  # 로딩 메시지 삭제
        st.error(f"❌ 백엔드 인덱스 리로드 중 오류: {str(e)}")
        st.warning("⚠️ FastAPI 서버가 실행 중인지 확인해주세요.")
        return False

def latex_to_text(text):
    """
    LaTeX 수식을 사람이 읽는 텍스트 수식으로 변환
    예: \frac{a}{b} → (a) / (b)
    """
    # \frac 변환 함수
    def frac_repl(match):
        return f"({match.group(1)}) / ({match.group(2)})"

    # LaTeX 블록(\[...\], $$...$$, $...$)을 찾아서 변환
    def latex_block_repl(match):
        latex_expr = match.group(1)
        # \left와 \right 제거 (괄호 크기 조정 명령어) - 더 포괄적으로 처리
        latex_expr = re.sub(r'\\left\s*\(', '(', latex_expr)
        latex_expr = re.sub(r'\\right\s*\)', ')', latex_expr)
        latex_expr = re.sub(r'\\left\s*\[', '[', latex_expr)
        latex_expr = re.sub(r'\\right\s*\]', ']', latex_expr)
        latex_expr = re.sub(r'\\left\s*\\{', '{', latex_expr)
        latex_expr = re.sub(r'\\right\s*\\}', '}', latex_expr)
        # \frac 변환
        latex_expr = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', frac_repl, latex_expr)
        # \times 변환
        latex_expr = latex_expr.replace(r'\times', '×')
        # 중괄호 제거
        latex_expr = latex_expr.replace('{', '').replace('}', '')
        return latex_expr

    # \[ ... \] 블록 변환
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: latex_block_repl(m), text, flags=re.DOTALL)
    # $$ ... $$ 블록 변환
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: latex_block_repl(m), text, flags=re.DOTALL)
    # $ ... $ 블록 변환
    text = re.sub(r'\$(.*?)\$', lambda m: latex_block_repl(m), text, flags=re.DOTALL)

    # 인라인 변환 (혹시 남아있을 경우)
    # \left와 \right 제거 - 더 포괄적으로 처리
    text = re.sub(r'\\left\s*\(', '(', text)
    text = re.sub(r'\\right\s*\)', ')', text)
    text = re.sub(r'\\left\s*\[', '[', text)
    text = re.sub(r'\\right\s*\]', ']', text)
    text = re.sub(r'\\left\s*\\{', '{', text)
    text = re.sub(r'\\right\s*\\}', '}', text)
    # \frac 변환
    text = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', frac_repl, text)
    text = text.replace(r'\times', '×')
    text = text.replace('{', '').replace('}', '')

    return text

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"
    if "api_processing" not in st.session_state:
        st.session_state.api_processing = False

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def stream_response(response_text: str, loading_placeholder):
    message_placeholder = st.empty()
    full_response = ""
    loading_placeholder.empty()
    for char in response_text:
        full_response += char
        message_placeholder.markdown(full_response + "▌")
        time.sleep(0.01)
    message_placeholder.markdown(full_response)
    return full_response

def send_message(user_input: str):
    if not user_input:
        return
    
    # API 처리 상태 시작 (즉시 설정)
    st.session_state.api_processing = True
    
    # 애니메이션 상태 초기화
    if "animation_step" not in st.session_state:
        st.session_state.animation_step = 0
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 로딩 메시지를 위한 플레이스홀더 
    loading_placeholder = st.empty()
    
    # 메시지를 세션에 추가
    st.session_state.messages.append({"role": "user", "content": user_input})

    # API 요청 데이터 준비
    request_data = {
        "messages": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]
    }
    
    try:
        # 연속된 애니메이션과 API 호출을 동시에 처리
        dot_patterns = ["", ".", "..", "...", "...."]
        
        # API 호출을 별도 스레드에서 실행
        import concurrent.futures
        import threading
        
        result_container = {"response": None, "error": None, "completed": False}
        
        def api_call():
            try:
                response = requests.post(API_URL, json=request_data)
                response.raise_for_status()
                result_container["response"] = response.json()["response"]
            except Exception as e:
                result_container["error"] = str(e)
            finally:
                result_container["completed"] = True
        
        # API 호출 스레드 시작
        api_thread = threading.Thread(target=api_call, daemon=True)
        api_thread.start()
        
        # 애니메이션 실행 (API 완료까지)
        cycle = 0
        while not result_container["completed"]:
            dots = dot_patterns[cycle % len(dot_patterns)]
            loading_placeholder.markdown(
                f"<div style='font-size: 1rem; color: #6b7280;'>🔍 AI가 응답을 생성하고 있습니다{dots}</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.4)
            cycle += 1
        
        # API 호출 완료 대기
        api_thread.join()
        
        if result_container["error"]:
            raise Exception(result_container["error"])
        
        # 응답 처리
        assistant_response = result_container["response"]
        
        # 수식 변환 적용
        assistant_response = latex_to_text(assistant_response)
        
        # API 처리 완료 상태로 변경
        st.session_state.api_processing = False
        
        # 스트리밍 방식으로 응답 표시
        with st.chat_message("assistant"):
            streamed_response = stream_response(assistant_response, loading_placeholder)
            st.session_state.messages.append({"role": "assistant", "content": streamed_response})
        
    except Exception as e:
        # 오류 발생 시에도 API 처리 완료 상태로 변경
        st.session_state.api_processing = False
        loading_placeholder.empty()
        st.error(f"Error: {str(e)}")
    
    # 최종적으로 API 처리 상태 확실히 종료
    st.session_state.api_processing = False

def show_upload_page():
    """문서 업로드 페이지를 표시합니다."""
    # 내부 로고 영역 (채팅 화면과 동일한 스타일)
    st.markdown("""
        <div style="background-color:#ffffff; padding: 1rem 2rem; border-radius: 6px; margin-bottom: 1rem; display: flex; align-items: center; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <img src="https://img.icons8.com/ios-filled/50/2d9bf0/document--v1.png" width="24px" style="margin-right: 10px;" />
            <span style="font-size: 1.3rem; font-weight: bold; color: #111827;">문서 업로드 </span>
        </div>
        <div style="color: #666666; font-size: 0.95rem; margin-bottom: 1.5rem;">
            문서를 업로드 해 AI를 학습시켜보세요.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color: rgba(30, 41, 59, 0.9); padding: 1rem 1.5rem; border-radius: 10px; border: 1px solid #1f2937; color: #f3f4f6; font-size: 0.95rem; line-height: 1.7;">
        <strong style="font-size: 1.1rem; color: #ffffff;">📌 파일 분류 규칙</strong>
        <ul style="list-style-type: '📂 '; padding-left: 1.2em; margin: 0;">
            <li><b>도시가스</b> 키워드가 파일명에 포함 → <span style="color: #38bdf8;"><b>Gas</b></span> 분류</li>
            <li><b>전력</b> 키워드가 파일명에 포함 → <span style="color: #38bdf8;"><b>Power</b></span> 분류</li>
            <li>그 외 키워드일 경우 → <span style="color: #facc15;"><b>Other</b></span> 분류</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")  # 공백 추가
    
    uploaded_files = st.file_uploader(
        "문서 파일을 선택하세요",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="여러 파일을 동시에 업로드할 수 있습니다.",
        label_visibility="collapsed"
    )
    # st.caption("PDF, DOCX, TXT 파일을 업로드 가능 합니다. (최대 200MB)")

    st.write("")  # 공백 추가

    if uploaded_files:
        file_data = []

        for file in uploaded_files:
            file_name = file.name
            file_size_kb = len(file.getvalue()) / 1024
            category = classify_file(file_name)

            warning_msg = ""
            if category == "other":
                warning_msg = "⚠️ 분류 불확실"

            file_data.append({
                "파일명": file_name,
                "크기": f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{file_size_kb/1024:.2f} MB",
                "분류": warning_msg if warning_msg else category
            })

        df = pd.DataFrame(file_data)
        st.markdown("""
            <div style="background-color:#ffffff; padding: 1rem 2rem; border-radius: 6px; margin-bottom: 1rem; display: flex; align-items: center; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
                <img src="https://img.icons8.com/ios-filled/50/2d9bf0/database--v1.png" width="24px" style="margin-right: 10px;" />
                <span style="font-size: 1.3rem; font-weight: bold; color: #111827;">📄 업로드된 문서</span>
            </div>
        """, unsafe_allow_html=True)
        st.table(df)
 
    if st.button("▶️ AI 문서 학습 시작", type="primary", disabled=not uploaded_files):
        try:
            with st.spinner("문서를 처리하고 인덱스를 생성하고 있습니다..."):
                success = build_vector_index_from_uploaded_files(uploaded_files)
                if success:
                    st.success("✅ 인덱스 생성이 완료되었습니다!")
                    
                    # 백엔드 인덱스 리로드
                    reload_success = reload_backend_indexes()
                    if reload_success:
                        st.success("🎉 모든 작업이 완료되었습니다! 이제 채팅 탭에서 질문할 수 있습니다.")
                    else:
                        st.warning("⚠️ 인덱스는 생성되었지만 백엔드 리로드에 실패했습니다. FastAPI 서버를 재시작하거나 수동으로 학습 문서 정보 리로드] 버튼을 눌러 주세요.")
                else:
                    st.error("❌ 인덱스 생성에 실패했습니다. 다시 시도해주세요.")
        except Exception as e:
            st.error(f"❌ 인덱스 생성 중 오류가 발생했습니다: {str(e)}")
            st.error("API 서버가 실행되지 않았을 수 있습니다. FastAPI 서버를 먼저 실행해주세요.")
    
    st.write("")  # 공백 추가

    # 기존 학습 정보 헤더
    st.markdown("""
        <div style="background-color:#ffffff; padding: 1rem 2rem; border-radius: 6px; margin-bottom: 1rem; display: flex; align-items: center; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <img src="https://img.icons8.com/ios-filled/50/2d9bf0/database--v1.png" width="24px" style="margin-right: 10px;" />
            <span style="font-size: 1.3rem; font-weight: bold; color: #111827;">학습 완료된 문서 정보</span>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Gas 인덱스 카드
    with col1:
        if GAS_INDEX_DIR.exists() and any(GAS_INDEX_DIR.iterdir()):
            # 저장된 문서 수 계산
            docs_dir = Path(__file__).parent.parent.parent / "vectordb" / "docs"
            gas_docs = [f for f in docs_dir.iterdir() if f.is_file() and classify_file(f.name) == 'gas'] if docs_dir.exists() else []
            doc_count = len(gas_docs)
            
            # 실제 마지막 업데이트 날짜 계산
            import datetime
            index_files = list(GAS_INDEX_DIR.iterdir())
            if index_files:
                latest_time = max(f.stat().st_mtime for f in index_files)
                last_update = datetime.datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d")
            else:
                last_update = "알 수 없음"
            
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                    <div style="font-weight: bold; font-size: 1.1rem; color: #333; margin-bottom: 0.5rem;">✅ Gas 문서 학습 완료</div>
                    <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.3rem;">학습 완료 문서 {doc_count}건</div>
                    <div style="color: #999; font-size: 0.8rem;">마지막 업데이트: {last_update}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                    <div style="font-weight: bold; font-size: 1.1rem; color: #333; margin-bottom: 0.5rem;">⚠️ Gas 문서 학습 미완료</div>
                    <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.3rem;">학습 완료 문서 0건</div>
                    <div style="color: #999; font-size: 0.8rem;"></div>
                </div>
            """, unsafe_allow_html=True)
    
    # Power 인덱스 카드
    with col2:
        if POWER_INDEX_DIR.exists() and any(POWER_INDEX_DIR.iterdir()):
            # 저장된 문서 수 계산
            docs_dir = Path(__file__).parent.parent.parent / "vectordb" / "docs"
            power_docs = [f for f in docs_dir.iterdir() if f.is_file() and classify_file(f.name) == 'power'] if docs_dir.exists() else []
            doc_count = len(power_docs)
            
            # 실제 마지막 업데이트 날짜 계산
            import datetime
            index_files = list(POWER_INDEX_DIR.iterdir())
            if index_files:
                latest_time = max(f.stat().st_mtime for f in index_files)
                last_update = datetime.datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d")
            else:
                last_update = "알 수 없음"
            
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                    <div style="font-weight: bold; font-size: 1.1rem; color: #333; margin-bottom: 0.5rem;">✅ Power 문서 학습 완료</div>
                    <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.3rem;">학습 완료 문서 {doc_count}건</div>
                    <div style="color: #999; font-size: 0.8rem;">마지막 업데이트: {last_update}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                    <div style="font-weight: bold; font-size: 1.1rem; color: #333; margin-bottom: 0.5rem;">⚠️ Power 문서 학습 미완료</div>
                    <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.3rem;">학습 완료 문서 0건</div>
                    <div style="color: #999; font-size: 0.8rem;"></div>
                </div>
            """, unsafe_allow_html=True)
    
    # Other 인덱스 카드
    with col3:
        if OTHER_INDEX_DIR.exists() and any(OTHER_INDEX_DIR.iterdir()):
            # 저장된 문서 수 계산
            docs_dir = Path(__file__).parent.parent.parent / "vectordb" / "docs"
            other_docs = [f for f in docs_dir.iterdir() if f.is_file() and classify_file(f.name) == 'other'] if docs_dir.exists() else []
            doc_count = len(other_docs)
            
            # 실제 마지막 업데이트 날짜 계산
            import datetime
            index_files = list(OTHER_INDEX_DIR.iterdir())
            if index_files:
                latest_time = max(f.stat().st_mtime for f in index_files)
                last_update = datetime.datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d")
            else:
                last_update = "알 수 없음"
            
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                    <div style="font-weight: bold; font-size: 1.1rem; color: #333; margin-bottom: 0.5rem;">✅ Other 문서 학습 완료</div>
                    <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.3rem;">학습 완료 문서 {doc_count}건</div>
                    <div style="color: #999; font-size: 0.8rem;">마지막 업데이트: {last_update}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 1px solid #e9ecef; text-align: center;">
                    <div style="font-weight: bold; font-size: 1.1rem; color: #333; margin-bottom: 0.5rem;">⚠️ Other 문서 학습 미완료</div>
                    <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.3rem;">학습 완료 문서 0건</div>
                    <div style="color: #999; font-size: 0.8rem;"></div>
                </div>
            """, unsafe_allow_html=True)
    
    
    st.write("")  # 공백 추가
    
    # 저장된 문서 파일 목록 표시
    docs_dir = Path(__file__).parent.parent.parent / "vectordb" / "docs"
    if docs_dir.exists() and any(docs_dir.iterdir()):
        # 숨김 파일 제외하고 파일 목록 가져오기
        files = [f for f in docs_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        if files:
            # st.info(f"📂 총 {len(files)}개 파일이 저장되어 있습니다:")
            
            # 파일 데이터를 표 형태로 준비
            file_data = []
            for file in sorted(files):
                file_size = file.stat().st_size
                file_size_kb = file_size / 1024
                category = classify_file(file.name)
                
                file_data.append({
                    "파일명": file.name,
                    "사이즈": f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{file_size_kb/1024:.2f} MB",
                    "분류": category
                })
            
            df = pd.DataFrame(file_data)
            st.table(df)
        else:
            st.info("📂 저장된 파일이 없습니다.")
    else:
        st.info("📂 문서 저장 디렉토리가 없습니다.")

    # 수동 리로드 버튼
    if st.button("🔄 학습 문서 정보 리로드", type="secondary"):
        reload_backend_indexes()

def show_chat_page():
    """채팅 페이지를 표시합니다."""
    # 내부 로고 영역
    st.markdown("""
        <div style="background-color:#ffffff; padding: 1rem 2rem; border-radius: 6px; margin-bottom: 1rem; display: flex; align-items: center; border: 1px solid #e5e7eb; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
            <img src="https://img.icons8.com/ios-filled/50/2d9bf0/document--v1.png" width="24px" style="margin-right: 10px;" />
            <span style="font-size: 1.3rem; font-weight: bold; color: #111827;">DocInsight AI</span>
        </div>
        <div style="color: #666666; font-size: 0.95rem; margin-bottom: 1.5rem;">
           다양한 문서를 학습한 AI가 질문에 답변합니다.
        </div>
    """, unsafe_allow_html=True)

    display_chat_history()

    user_input = st.chat_input("서울시 도시가스 요금 산정 방식은?")
    if user_input:
        send_message(user_input)
        st.rerun()

def main():
    initialize_session_state()

    with st.sidebar:
        st.markdown("""
            <style>
            .sidebar-title {
                font-size: 1.3rem;
                font-weight: bold;
                margin-bottom: 1rem;
                color: white !important;
            }
            </style>
            <div class="sidebar-title">문서관리</div>
        """, unsafe_allow_html=True)
        
        # API 처리 중인지 확인
        is_processing = st.session_state.get("api_processing", False)
        
        # 문서업로드 버튼
        upload_clicked = st.button(
            "📂 문서 업로드", 
            use_container_width=True,
            disabled=is_processing,
            key="upload_button"
        )
        
        if upload_clicked and not is_processing:
            st.session_state.current_page = "upload"
            st.rerun()
        
        # 뒤로가기 버튼은 문서 업로드 화면에서만 표시
        if st.session_state.current_page == "upload":
            # 문서 업로드 버튼 바로 아래에 뒤로가기 버튼 배치
            back_clicked = st.button(
                "← 뒤로가기", 
                use_container_width=True,
                disabled=is_processing,
                key="back_button"
            )
            
            if back_clicked and not is_processing:
                st.session_state.current_page = "chat"
                st.rerun()

    # 상단 전체 헤더 영역 어둡게 처리
    st.markdown("""
        <style>
        header[data-testid="stHeader"] {
            background-color: #111827;
        }

        section[data-testid="stSidebar"] {
            width: 187px !important;
            background-color: #111827 !important;
            color: white !important;
            overflow-y: hidden !important;  /* 사이드바 세로 스크롤 비활성화 */
            overflow-x: hidden !important;  /* 사이드바 가로 스크롤 비활성화 */
            height: 100vh !important;       /* 사이드바 높이를 화면 높이로 고정 */
            max-height: 100vh !important;   /* 최대 높이 제한 */
        }

        /* 사이드바 내부 컨테이너도 스크롤 방지 */
        section[data-testid="stSidebar"] > div {
            overflow-y: hidden !important;
            overflow-x: hidden !important;
            height: 100vh !important;
            max-height: 100vh !important;
        }

        /* 사이드바 내부 모든 요소들의 스크롤 방지 */
        section[data-testid="stSidebar"] * {
            overflow-y: visible !important;
            overflow-x: visible !important;
        }

        /* 사이드바 버튼이 disabled 상태에서도 색상 유지 */
        section[data-testid="stSidebar"] .stButton > button {
            background-color: #333333 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: #555555 !important;
        }
        section[data-testid="stSidebar"] .stButton > button:disabled {
            background-color: #333333 !important;
            color: white !important;
            opacity: 0.6 !important;
            cursor: not-allowed !important;
        }

        /* 사이드바 토글 버튼 강조 */
        button[data-testid="collapsedControl"] {
            border: 2px solid #9ca3af !important;
            border-radius: 50% !important;
            background-color: #f3f4f6 !important;
            width: 36px !important;
            height: 36px !important;
            display: flex;
            align-items: center;
            justify-content: center;
            position: fixed;
            top: 1.25rem;
            left: 0.75rem;
            z-index: 10000;
        }
        button[data-testid="collapsedControl"] svg {
            stroke: #374151 !important;
            width: 20px !important;
            height: 20px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 현재 페이지에 따라 다른 내용 표시
    if st.session_state.current_page == "upload":
        show_upload_page()
    else:
        show_chat_page()

    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Pretendard', sans-serif !important;
        }
        /* 메인 콘텐츠 영역의 버튼만 스타일 적용 (사이드바 제외) */
        .main .stButton button {
            background-color: #333333 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
        }
        .main .stButton button:hover {
            background-color: #555555 !important;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
