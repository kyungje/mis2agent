from fastapi.testclient import TestClient
from src.api.example import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_get_example():
    """GET /example 엔드포인트 테스트"""
    response = client.get("/example/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "This is an example API endpoint"
    assert data["status"] == "success"

def test_echo_message():
    """POST /example/echo 엔드포인트 테스트"""
    message = "Hello, API!"
    response = client.post("/example/echo", json={"message": message})
    assert response.status_code == 200
    data = response.json()
    assert data["echo"] == message

def test_get_error():
    """GET /example/error 엔드포인트 테스트"""
    response = client.get("/example/error")
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "This is an example error" 