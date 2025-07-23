# chat_ui.py (UI 변경된 버전)
import streamlit as st
import requests
import time
import re
import os
from dotenv import load_dotenv

# 페이지 설정을 가장 먼저 실행
st.set_page_config(page_title="DocInsight AI", page_icon="📄", layout="wide")

# .env 파일 로드
load_dotenv()

# API 엔드포인트 설정
API_URL = "http://localhost:8000/chat"

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

# 세션 초기화
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "streaming" not in st.session_state:
        st.session_state.streaming = False

# 채팅 기록 출력
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 스트리밍 출력
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

# 메시지 전송
def send_message(user_input: str):
    if not user_input:
        return
    with st.chat_message("user"):
        st.markdown(user_input)
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.markdown('🔍 AI가 응답을 생성하고 있습니다...')
    st.session_state.messages.append({"role": "user", "content": user_input})

    request_data = {
        "messages": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]
    }
    try:
        response = requests.post(API_URL, json=request_data)
        response.raise_for_status()
        assistant_response = response.json()["response"]
        
        # === 여기에서 수식 변환 적용 ===
        assistant_response = latex_to_text(assistant_response)
        
        with st.chat_message("assistant"):
            streamed_response = stream_response(assistant_response, loading_placeholder)
            st.session_state.messages.append({"role": "assistant", "content": streamed_response})
    except requests.exceptions.ConnectionError:
        loading_placeholder.empty()
        st.error("❌ FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
        st.info("서버 실행: `poetry run python -m src.api.chat`")
    except requests.exceptions.HTTPError as e:
        loading_placeholder.empty()
        st.error(f"❌ HTTP 오류: {e}")
        if e.response.status_code == 500:
            st.error("서버 내부 오류가 발생했습니다. 백엔드 로그를 확인해주세요.")
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Error: {str(e)}")

# 메인 함수
def main():
    initialize_session_state()

    # 좌측 사이드바 (문서 업로드 및 인덱스 관리용 유지)
    with st.sidebar:
        st.title("문서관리")
        st.button("📂 문서 업로드", use_container_width=True)
        st.button("🔄 인덱스 재생성", use_container_width=True)

    # 메인 화면 (채팅 영역)
    st.markdown("""
        <style>
            .doc-title { font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem; }
            .doc-subtitle { color: #888; margin-bottom: 2rem; }
            .chat-box { padding: 2rem 2rem 8rem 2rem; border-radius: 10px; background-color: #fff; }
            .chat-footer { position: fixed; bottom: 0; left: 270px; width: calc(100% - 270px); padding: 1rem; background: white; border-top: 1px solid #e0e0e0; z-index: 999; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="chat-box">
            <div class="doc-title">📘 DocInsight AI</div>
            <div class="doc-subtitle">문서를 업로드하고 질문하세요. 인공지능이 요약과 검색을 도와줍니다.</div>
    """, unsafe_allow_html=True)

    display_chat_history()

    st.markdown("""
        </div>
        <div class="chat-footer">
    """, unsafe_allow_html=True)

    user_input = st.chat_input("질문을 입력하세요")
    if user_input:
        send_message(user_input)
        st.rerun()

    st.markdown("""
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* 전체 폰트 통일 */
    html, body, [class*="css"]  {
        font-family: 'Pretendard', sans-serif !important;
    }

    /* 사이드바 스타일 */
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] span {
        color: white !important;
    }

    /* 사이드바 버튼 스타일 */
    .stButton button {
        background-color: #333333 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton button:hover {
        background-color: #555555 !important;
    }

    /* 채팅 박스 영역 여백 및 배경 */
    .chat-box {
        padding: 2rem 2rem 8rem 2rem;
        border-radius: 10px;
        background-color: #ffffff;
    }

    /* 채팅 입력창 고정 하단 */
    .chat-footer {
        position: fixed;
        bottom: 0;
        left: 270px;
        width: calc(100% - 270px);
        padding: 1rem;
        background: white;
        border-top: 1px solid #e0e0e0;
        z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 