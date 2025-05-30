from openai import AsyncOpenAI
import os
import logging
from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent
from .legal_agent import LegalAgent
from .general_agent import GeneralAgent

logger = logging.getLogger(__name__)

class Agent:
    """여러 Agent를 관리하고 적절한 Agent를 선택하여 질문을 처리하는 메인 Agent 클래스"""
    
    def __init__(self, index_path: str):
        # OpenAI 클라이언트 초기화
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4"  # 기본 모델 설정
        
        # 각 Agent 초기화
        self.legal_agent = LegalAgent(index_path)
        self.general_agent = GeneralAgent()
        
        # Agent 목록
        self.agents: List[BaseAgent] = [
            self.legal_agent,
            self.general_agent
        ]
    
    def run(self, question: str, chat_history: Optional[List] = None) -> str:
        """질문에 대해 적절한 Agent를 선택하여 답변을 생성합니다."""
        if chat_history is None:
            chat_history = []
        
        # 서울특별시 도시가스회사 공급규정 관련 질문인지 확인
        if self.legal_agent.can_handle(question):
            logger.info("Selected Agent: Legal Agent")
            return self.legal_agent.run(question, chat_history)
        
        # 그 외의 경우 일반 Agent 사용
        logger.info("Selected Agent: General Agent")
        return self.general_agent.run(question, chat_history)

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        주어진 메시지 목록을 기반으로 AI 응답을 생성합니다.
        서울특별시 도시가스회사 공급규정 관련 질문인 경우 LegalAgent를 사용하고, 그 외의 경우 일반 대화를 처리합니다.
        
        Args:
            messages: 대화 메시지 목록 (role과 content를 포함하는 딕셔너리 리스트)
            
        Returns:
            str: AI의 응답 메시지
        """
        try:
            # 입력 메시지 로깅
            logger.info("Input messages:")
            for msg in messages:
                logger.info(f"Role: {msg['role']}, Content: {msg['content']}")
            
            # 마지막 사용자 메시지 추출
            last_user_message = next(
                (msg['content'] for msg in reversed(messages) if msg['role'] == 'user'),
                None
            )
            
            if last_user_message:
                # 대화 기록을 원본 형식 그대로 유지
                chat_history = messages[:-1]  # 마지막 메시지 제외
                
                # 서울특별시 도시가스회사 공급규정 관련 질문인지 확인
                if self.legal_agent.can_handle(last_user_message):
                    logger.info("Selected Agent: Legal Agent")
                    response = self.legal_agent.run(last_user_message, chat_history)
                    logger.info(f"Legal Agent Response: {response}")
                    return response
            
            # 관련 질문이 아니거나 Agent가 없는 경우 기본 응답 생성
            logger.info("Selected Agent: General Agent")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            
            # 응답 메시지 로깅
            response_content = response.choices[0].message.content
            logger.info(f"General Agent Response: {response_content}")
            
            return response_content
            
        except Exception as e:
            logger.error(f"Agent error: {str(e)}")
            raise 