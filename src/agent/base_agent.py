from abc import ABC, abstractmethod
from typing import List, Optional

class BaseAgent(ABC):
    """모든 Agent의 기본 클래스"""
    
    @abstractmethod
    def run(self, question: str, chat_history: Optional[List] = None) -> str:
        """질문에 대한 답변을 생성합니다."""
        pass
    
    @abstractmethod
    def can_handle(self, question: str) -> bool:
        """이 Agent가 해당 질문을 처리할 수 있는지 확인합니다."""
        pass 