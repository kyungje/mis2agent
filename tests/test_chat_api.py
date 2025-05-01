from fastapi.testclient import TestClient
from src.api.chat import app, Message, ChatRequest
import pytest

client = TestClient(app)

def test_chat_endpoint():
    """채팅 엔드포인트 테스트"""
    # 테스트 메시지 준비
    messages = [
        Message(role="user", content="Hello, how are you?")
    ]
    request_data = ChatRequest(messages=messages)
    
    # API 호출
    response = client.post("/chat", json=request_data.dict())
    
    # 응답 검증
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)

def test_chat_endpoint_empty_messages():
    """빈 메시지 리스트 테스트"""
    request_data = ChatRequest(messages=[])
    
    response = client.post("/chat", json=request_data.dict())
    assert response.status_code == 200

def test_chat_endpoint_invalid_request():
    """잘못된 요청 테스트"""
    # 잘못된 형식의 요청 데이터
    invalid_data = {"messages": "invalid"}
    
    response = client.post("/chat", json=invalid_data)
    assert response.status_code == 422  # Validation Error 