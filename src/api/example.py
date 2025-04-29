from fastapi import APIRouter, HTTPException
from typing import Dict, Any

router = APIRouter(prefix="/example", tags=["example"])

@router.get("/")
async def get_example() -> Dict[str, Any]:
    """예제 API 엔드포인트"""
    return {
        "message": "This is an example API endpoint",
        "status": "success"
    }

@router.post("/echo")
async def echo_message(message: str) -> Dict[str, str]:
    """메시지를 에코하는 예제 엔드포인트"""
    return {"echo": message}

@router.get("/error")
async def get_error() -> None:
    """에러 발생 예제"""
    raise HTTPException(
        status_code=400,
        detail="This is an example error"
    ) 