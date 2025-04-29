from typing import Dict, Any
from .base import BaseAgent

class ExampleAgent(BaseAgent):
    """예제 에이전트 구현"""
    
    def __init__(self):
        self.context: Dict[str, Any] = {
            "name": "Example Agent",
            "version": "1.0.0"
        }
    
    async def process_message(self, message: str) -> str:
        """간단한 메시지 처리 예제"""
        return f"Example Agent received: {message}"
    
    async def get_context(self) -> Dict[str, Any]:
        """현재 컨텍스트 반환"""
        return self.context
    
    async def update_context(self, new_context: Dict[str, Any]) -> None:
        """컨텍스트 업데이트"""
        self.context.update(new_context) 