from openai import AsyncOpenAI
import os
import logging
from typing import List, Dict, Any, Optional
from .base_agent import BaseAgent
from .legal_agent import LegalAgent
from .power_agent import PowerAgent
from .general_agent import GeneralAgent

logger = logging.getLogger(__name__)

class Agent:
    """여러 Agent를 관리하고 적절한 Agent를 선택하여 질문을 처리하는 메인 Agent 클래스"""
    
    def __init__(self, gas_index_path: str = None, power_index_path: str = None, others_index_path: str = None):
        # OpenAI 클라이언트 초기화
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4-turbo"  # 기본 모델 설정
        
        # 도메인별 인덱스 경로 설정
        self.gas_index_path = gas_index_path or os.getenv("GAS_INDEX_PATH")
        self.power_index_path = power_index_path or os.getenv("POWER_INDEX_PATH")
        self.others_index_path = others_index_path or os.getenv("OTHERS_INDEX_PATH")
        
        # 각 Agent 초기화 (도메인별로 다른 인덱스 사용)
        self.legal_agent = LegalAgent(self.gas_index_path)
        self.power_agent = PowerAgent(self.power_index_path) if self.power_index_path else None
        self.general_agent = GeneralAgent(self.others_index_path)
        
        # Agent 목록
        self.agents: List[BaseAgent] = [
            self.legal_agent,
            self.general_agent
        ]
        
        # PowerAgent가 초기화된 경우 목록에 추가
        if self.power_agent:
            self.agents.append(self.power_agent)
    
    async def _select_agent_with_ai(self, question: str) -> BaseAgent:
        """AI를 사용하여 질문에 가장 적합한 Agent를 선택합니다."""
        try:
            # AI 기반 Agent 선택을 위한 프롬프트
            prompt = f"""다음 질문에 가장 적합한 AI Agent를 선택해주세요.

사용 가능한 Agent:
1. legal_agent: 서울특별시 도시가스회사 공급규정 관련 질문 처리
   - 가스 공급, 요금, 계약, 안전, 설비 관련 질문
   - 압력 관련 질문 (최고압력, 공급압력, 압력 초과 등)
   - 도시가스 공급규정, 가스사용량 산정 관련 질문
   - 서울특별시 도시가스회사 관련 모든 질문

2. power_agent: 전력 관련 질문 처리
   - 전력, 전기, 요금, 계약, 안전, 설비 관련 질문
   - 한국전력, 전기사업법 관련 질문
   - 전기 사용, 계량, 송전, 배전, 발전 관련 질문

3. general_agent: 일반적인 대화, 위 두 도메인에 해당하지 않는 질문 처리

질문: {question}

위 질문에 가장 적합한 Agent를 다음 중 하나로만 선택해주세요:
- legal_agent
- power_agent  
- general_agent

답변은 반드시 위 세 가지 중 하나로만 해주세요."""

            # AI를 통해 Agent 선택
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # 일관성을 위해 낮은 temperature 사용
            )
            
            selected_agent_name = response.choices[0].message.content.strip().lower()
            logger.info(f"AI selected agent: {selected_agent_name}")
            
            # 선택된 Agent 반환
            if selected_agent_name == "legal_agent":
                return self.legal_agent
            elif selected_agent_name == "power_agent" and self.power_agent:
                return self.power_agent
            else:
                return self.general_agent
                
        except Exception as e:
            logger.error(f"Error in AI-based agent selection: {e}")
            # 오류 발생 시 GeneralAgent 반환
            return self.general_agent
    
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        주어진 메시지 목록을 기반으로 AI 응답을 생성합니다.
        AI가 질문을 분석하여 적절한 Agent를 선택합니다.
        
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
                
                # AI 기반으로 적절한 Agent 선택
                selected_agent = await self._select_agent_with_ai(last_user_message)
                logger.info(f"Selected Agent: {type(selected_agent).__name__}")
                
                # 선택된 Agent가 LegalAgent인 경우
                if isinstance(selected_agent, LegalAgent):
                    response = selected_agent.run(last_user_message, chat_history)
                    logger.info(f"Legal Agent Response: {response}")
                    return response
                
                # 선택된 Agent가 PowerAgent인 경우
                if isinstance(selected_agent, PowerAgent):
                    response = selected_agent.run(last_user_message, chat_history)
                    logger.info(f"Power Agent Response: {response}")
                    return response
                
                # 선택된 Agent가 GeneralAgent인 경우
                if isinstance(selected_agent, GeneralAgent):
                    response = selected_agent.run(last_user_message, chat_history)
                    logger.info(f"General Agent Response: {response}")
                    return response
            
            # 기본 응답 생성
            logger.info("Selected Agent: General Agent (fallback)")
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