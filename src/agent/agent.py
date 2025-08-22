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
    
    async def _detect_ambiguous_question(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """질문이 애매한지 판단하고 가능한 에이전트들을 반환합니다."""
        try:
            # 대화 기록 컨텍스트 생성
            context_str = ""
            if chat_history:
                # 최근 5개 메시지만 컨텍스트로 사용
                recent_history = chat_history[-5:]
                context_parts = []
                for msg in recent_history:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        context_parts.append(f"{msg['role']}: {msg['content']}")
                
                if context_parts:
                    context_str = f"\n\n이전 대화 기록:\n{chr(10).join(context_parts)}\n"
            
            prompt = f"""다음 질문을 분석하고, 어떤 AI Agent가 적합한지 판단해주세요.
{context_str}
사용 가능한 Agent:
1. legal_agent: 도시가스회사 공급규정 관련 질문
   - 가스 공급, 요금, 계약, 안전, 설비 관련 질문
   - 도시가스 압력 관련 질문 (최고압력, 공급압력, 압력 초과 등)
   - 도시가스 공급규정, 가스사용량 산정 관련 질문

2. power_agent: 전력 관련 질문
   - 전력, 전기, 요금, 계약, 안전, 설비 관련 질문
   - 전기 압력, 전압 관련 질문
   - 한국전력, 전기사업법 관련 질문
   - 발전, 송전, 배전, 복합발전기, 기동비용 등 전력 시스템 관련 질문

3. general_agent: 일반적인 대화, 위 두 도메인에 해당하지 않는 질문

현재 질문: {question}

**중요: 이전 대화 맥락을 고려하여 판단하세요. 후속 질문의 경우 이전에 어떤 주제에 대해 이야기했는지 참고하세요.**

다음 형식으로 답변해주세요:
confidence: [높음/중간/낮음] (이 질문이 어떤 도메인인지 확신하는 정도)
primary_agent: [legal_agent/power_agent/general_agent] (가장 적합한 에이전트)
possible_agents: [legal_agent, power_agent, general_agent 중에서 가능한 모든 에이전트들을 쉼표로 구분]
reason: 선택 이유를 간단히 설명

예시:
confidence: 낮음
primary_agent: legal_agent
possible_agents: legal_agent, power_agent
reason: "압력"이라는 단어가 도시가스 압력일 수도 있고 전기 압력(전압)일 수도 있어서 애매함"""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            logger.info(f"Ambiguity detection response: {content}")
            
            # 응답 파싱
            result = {
                "confidence": "중간",
                "primary_agent": "general_agent",
                "possible_agents": ["general_agent"],
                "reason": "파싱 오류"
            }
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('confidence:'):
                    result["confidence"] = line.split(':', 1)[1].strip()
                elif line.startswith('primary_agent:'):
                    result["primary_agent"] = line.split(':', 1)[1].strip()
                elif line.startswith('possible_agents:'):
                    agents_str = line.split(':', 1)[1].strip()
                    result["possible_agents"] = [agent.strip() for agent in agents_str.split(',')]
                elif line.startswith('reason:'):
                    result["reason"] = line.split(':', 1)[1].strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ambiguity detection: {e}")
            return {
                "confidence": "중간",
                "primary_agent": "general_agent", 
                "possible_agents": ["general_agent"],
                "reason": f"오류 발생: {str(e)}"
            }

    async def _select_agent_with_ai(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> BaseAgent:
        """AI를 사용하여 질문에 가장 적합한 Agent를 선택합니다."""
        try:
            # 대화 기록 컨텍스트 생성
            context_str = ""
            if chat_history:
                # 최근 5개 메시지만 컨텍스트로 사용
                recent_history = chat_history[-5:]
                context_parts = []
                for msg in recent_history:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        context_parts.append(f"{msg['role']}: {msg['content']}")
                
                if context_parts:
                    context_str = f"\n\n이전 대화 기록:\n{chr(10).join(context_parts)}\n"
            
            # AI 기반 Agent 선택을 위한 프롬프트
            prompt = f"""다음 질문에 가장 적합한 AI Agent를 선택해주세요.
{context_str}
사용 가능한 Agent:
1. legal_agent: 도시가스회사 공급규정 관련 질문 처리
   - 가스 공급, 요금, 계약, 안전, 설비 관련 질문
   - 압력 관련 질문 (최고압력, 공급압력, 압력 초과 등)
   - 도시가스 공급규정, 가스사용량 산정 관련 질문
   - 도시가스회사 관련 모든 질문

2. power_agent: 전력 관련 질문 처리
   - 전력, 전기, 요금, 계약, 안전, 설비 관련 질문
   - 한국전력, 전기사업법 관련 질문
   - 전기 사용, 계량, 송전, 배전, 발전 관련 질문
   - 복합발전기, 기동비용, 전력 시스템 운영 관련 질문

3. general_agent: 일반적인 대화, 위 두 도메인에 해당하지 않는 질문 처리

현재 질문: {question}

**중요 지침:**
- 이전 대화 맥락을 반드시 고려하세요
- 후속 질문("구체적인 수식", "더 자세히", "예시" 등)의 경우, 이전에 논의된 주제와 같은 도메인으로 분류하세요
- 애매한 단어라도 이전 맥락이 있다면 그 맥락을 우선시하세요

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

    def _get_agent_by_name(self, agent_name: str) -> BaseAgent:
        """에이전트 이름으로 해당 에이전트 객체를 반환합니다."""
        agent_name = agent_name.lower().strip()
        if agent_name == "legal_agent":
            return self.legal_agent
        elif agent_name == "power_agent" and self.power_agent:
            return self.power_agent
        else:
            return self.general_agent

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        주어진 메시지 목록을 기반으로 AI 응답을 생성합니다.
        AI가 질문을 분석하여 적절한 Agent를 선택하고, 애매한 경우 사용자에게 구체적인 정보를 요청합니다.
        
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
                
                # 질문의 애매함 판단 (대화 기록 포함)
                ambiguity_result = await self._detect_ambiguous_question(last_user_message, chat_history)
                
                # 확신도가 낮고 여러 에이전트가 가능한 경우만 사용자에게 구체적 정보 요청
                # 단, 이전 대화 기록이 있다면 맥락을 고려하여 더 관대하게 처리
                should_ask_clarification = (
                    ambiguity_result["confidence"] == "낮음" and 
                    len(ambiguity_result["possible_agents"]) > 1 and
                    len(chat_history) == 0  # 첫 번째 질문일 때만 명확화 요청
                )
                
                if should_ask_clarification:
                    logger.info(f"Ambiguous question detected: {ambiguity_result['reason']}")
                    
                    # 가능한 영역들을 설명하며 구체적인 정보 요청
                    possible_agents = ambiguity_result["possible_agents"]
                    clarification_response = self._generate_clarification_response(
                        last_user_message, possible_agents, ambiguity_result["reason"]
                    )
                    
                    return clarification_response
                
                # 확신도가 높거나 애매하지 않은 경우 또는 후속 질문인 경우 선택된 에이전트로 처리
                chosen_agent = self._get_agent_by_name(ambiguity_result["primary_agent"])
                logger.info(f"Selected Agent: {type(chosen_agent).__name__} (confidence: {ambiguity_result['confidence']}, context considered: {len(chat_history) > 0})")
                
                # 선택된 Agent로 응답 생성
                if isinstance(chosen_agent, LegalAgent):
                    response = chosen_agent.run(last_user_message, chat_history)
                    logger.info(f"Legal Agent Response: {response}")
                    return response
                
                if isinstance(chosen_agent, PowerAgent):
                    response = chosen_agent.run(last_user_message, chat_history)
                    logger.info(f"Power Agent Response: {response}")
                    return response
                
                if isinstance(chosen_agent, GeneralAgent):
                    response = chosen_agent.run(last_user_message, chat_history)
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

    def _generate_clarification_response(self, question: str, possible_agents: List[str], reason: str) -> str:
        """애매한 질문에 대해 구체적인 정보를 요청하는 응답을 생성합니다."""
        
        # 에이전트별 설명 매핑
        agent_descriptions = {
            'legal_agent': '🏛️ **도시가스 관련**: 도시가스회사의 공급규정, 가스 요금, 가스 압력 기준, 가스 안전 관련',
            'power_agent': '⚡ **전력 관련**: 전력의 전기 요금, 전압 기준, 전력 설비, 전기사업법 관련',
            'general_agent': '💬 **일반 질문**: 위 두 영역에 해당하지 않는 일반적인 질문'
        }
        
        clarification_text = f"""🤔 **질문이 다소 애매합니다!**

**질문**: "{question}"

**분석 결과**: {reason}

다음 중 어떤 영역에 대한 질문인지 좀 더 구체적으로 말씀해주시겠어요?

"""
        
        # 가능한 에이전트들에 대한 설명 추가
        for agent in possible_agents:
            if agent in agent_descriptions:
                clarification_text += f"{agent_descriptions[agent]}\n\n"
        
        clarification_text += """**💡 예시**:
- "도시가스 압력 기준은?" → 도시가스 관련
- "전기 압력(전압) 기준은?" → 전력 관련

좀 더 구체적으로 질문해주시면 정확한 답변을 드릴 수 있습니다! 😊"""
        
        return clarification_text 