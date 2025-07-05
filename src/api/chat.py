from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import logging
from dotenv import load_dotenv
from src.agent.agent import Agent
from src.config.logging_config import setup_logging

# 로깅 설정
logger = setup_logging()

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# 도메인별 FAISS 인덱스 경로 설정
gas_index_path = os.getenv("GAS_INDEX_PATH")
power_index_path = os.getenv("POWER_INDEX_PATH")
others_index_path = os.getenv("OTHERS_INDEX_PATH")

if not gas_index_path:
    logger.warning("GAS_INDEX_PATH is not set. Gas-related queries will use dummy data.")

if not power_index_path:
    logger.warning("POWER_INDEX_PATH is not set. Power-related queries will use general agent.")

if not others_index_path:
    logger.warning("OTHERS_INDEX_PATH is not set. General queries will use basic conversation.")

# Agent 초기화 (도메인별 인덱스 경로 전달)
agent = Agent(
    gas_index_path=gas_index_path or "dummy",
    power_index_path=power_index_path,
    others_index_path=others_index_path
)

app = FastAPI(title="Chat API")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str

class ReloadResponse(BaseModel):
    message: str
    success: bool

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 메시지를 처리하고 응답을 반환합니다."""
    try:
        # 메시지 형식 변환
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Agent를 통해 응답 생성
        response = await agent.get_response(messages)
        
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-indexes", response_model=ReloadResponse)
async def reload_indexes():
    """인덱스를 다시 로드합니다."""
    try:
        logger.info("Reloading indexes...")
        
        # 새로운 Agent 인스턴스 생성 (최신 인덱스 로드)
        global agent
        agent = Agent(
            gas_index_path=gas_index_path or "dummy",
            power_index_path=power_index_path,
            others_index_path=others_index_path
        )
        
        logger.info("Indexes reloaded successfully")
        return ReloadResponse(
            message="인덱스가 성공적으로 다시 로드되었습니다.",
            success=True
        )
    except Exception as e:
        logger.error(f"Error reloading indexes: {str(e)}")
        return ReloadResponse(
            message=f"인덱스 리로드 중 오류가 발생했습니다: {str(e)}",
            success=False
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 