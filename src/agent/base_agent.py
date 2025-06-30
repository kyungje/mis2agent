from abc import ABC, abstractmethod
from typing import List, Optional

class BaseAgent(ABC):
    """모든 Agent의 기본 클래스"""
    
    @abstractmethod
    def run(self, question: str, chat_history: Optional[List] = None) -> str:
        """질문에 대한 답변을 생성합니다."""
        pass 