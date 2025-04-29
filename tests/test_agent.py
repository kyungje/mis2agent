import pytest
from src.agent.example import ExampleAgent

@pytest.fixture
def agent():
    return ExampleAgent()

@pytest.mark.asyncio
async def test_process_message(agent):
    """메시지 처리 테스트"""
    message = "Hello, Agent!"
    response = await agent.process_message(message)
    assert response == f"Example Agent received: {message}"

@pytest.mark.asyncio
async def test_get_context(agent):
    """컨텍스트 조회 테스트"""
    context = await agent.get_context()
    assert context["name"] == "Example Agent"
    assert context["version"] == "1.0.0"

@pytest.mark.asyncio
async def test_update_context(agent):
    """컨텍스트 업데이트 테스트"""
    new_context = {"status": "active"}
    await agent.update_context(new_context)
    context = await agent.get_context()
    assert context["status"] == "active" 