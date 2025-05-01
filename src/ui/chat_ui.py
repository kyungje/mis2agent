import streamlit as st
import requests
import json
from typing import List, Dict, Any
import time

# API 엔드포인트 설정
API_URL = "http://localhost:8000/chat"

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