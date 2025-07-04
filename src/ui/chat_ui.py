import streamlit as st
import requests
import json
from typing import List, Dict, Any
import time
import re  # re 모듈 추가

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

def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "streaming" not in st.session_state:
        st.session_state.streaming = False

def display_chat_history():
    """채팅 기록 표시"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def stream_response(response_text: str, loading_placeholder):
    """스트리밍 방식으로 응답 표시"""
    message_placeholder = st.empty()
    full_response = ""
    
    # 로딩 메시지 제거
    loading_placeholder.empty()
    
    # 문자를 하나씩 표시
    for char in response_text:
        full_response += char
        message_placeholder.markdown(full_response + "▌")
        time.sleep(0.01)  # 타이핑 효과를 위한 지연
    
    # 최종 응답 표시
    message_placeholder.markdown(full_response)
    return full_response

def send_message(user_input: str):
    """메시지 전송 및 응답 처리"""
    if not user_input:
        return

    # 사용자 메시지를 먼저 화면에 표시
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 로딩 메시지 표시를 위한 플레이스홀더
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.markdown('🤖 AI가 응답을 생성하고 있습니다...')
    
    # 사용자 메시지를 세션에 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # API 요청 데이터 준비
    request_data = {
        "messages": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]
    }
    
    try:
        # API 호출
        response = requests.post(API_URL, json=request_data)
        response.raise_for_status()
        
        # 응답 처리
        assistant_response = response.json()["response"]
        
        # === 여기에서 수식 변환 적용 ===
        assistant_response = latex_to_text(assistant_response)
        
        # 스트리밍 방식으로 응답 표시
        with st.chat_message("assistant"):
            streamed_response = stream_response(assistant_response, loading_placeholder)
            st.session_state.messages.append({"role": "assistant", "content": streamed_response})
        
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Error: {str(e)}")

def main():
    """메인 함수"""
    st.set_page_config(
        page_title="Chat",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Chat")
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 사이드바
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This is a chat interface built with Streamlit.
        It uses OpenAI's GPT model through a FastAPI backend.
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # 채팅 인터페이스
    display_chat_history()
    
    # 메시지 입력
    if prompt := st.chat_input("What's on your mind?"):
        send_message(prompt)
        st.rerun()

if __name__ == "__main__":
    main() 