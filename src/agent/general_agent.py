from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from .base_agent import BaseAgent
from ..validation.response_validator import ResponseValidator
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class GeneralAgent(BaseAgent):
    def __init__(self, others_index_path: str = None):
        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0.7
        )
        
        # 기타 문서 벡터 DB 초기화
        self.others_index_path = others_index_path
        self.rag_tool = None
        
        if self.others_index_path:
            try:
                from ..tools.others_rag_tools import OthersRAGTool
                self.rag_tool = OthersRAGTool(self.others_index_path)
                logger.info(f"Others RAG tool initialized with path: {self.others_index_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize Others RAG tool: {e}")
                self.rag_tool = None
        
        # 검증기 초기화
        self.validator = ResponseValidator()
        
        # Agent 프롬프트 로드
        self.prompt = load_prompt("src/agent/prompts/agentic-rag-prompt-general.yaml", encoding='utf-8')
        
        # Agent 생성
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=[self.rag_tool] if self.rag_tool else []
        )
        
        # Agent 실행기 생성
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[self.rag_tool] if self.rag_tool else [],
            verbose=True
        )
    
    def run(self, question: str, chat_history: Optional[List] = None) -> str:
        """질문에 대한 답변을 생성합니다."""
        if chat_history is None:
            chat_history = []
        
        # 대화 기록을 문자열 형식으로 변환하고 최근 3개만 유지
        formatted_history = []
        for msg in chat_history[-3:]:  # 최근 3개 메시지만 사용
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted_history.append(f"{msg['role']}: {msg['content']}")

        # 검증 및 재시도 로직
        max_retries = 1
        retry_count = 0
        search_strategies = ["default", "expanded", "keyword"]
        
        while retry_count <= max_retries:
            logger.info(f"GeneralAgent: Attempt {retry_count + 1} for question: {question[:50]}...")
            
            # 현재 시도에 맞는 검색 전략 선택
            current_strategy = search_strategies[min(retry_count, len(search_strategies) - 1)]
            logger.info(f"GeneralAgent: Using search strategy: {current_strategy}")
            
            # RAG 도구가 있는 경우 검색 수행
            if self.rag_tool:
                try:
                    # vectordb 검색 수행
                    search_result = self.rag_tool._run(question, search_strategy=current_strategy)
                    
                    # 검색 결과 관련성 검증
                    search_relevance, search_score = self.validator.validate_search_relevance(question, search_result)
                    logger.info(f"GeneralAgent: Search validation - Relevance: {search_relevance}, Score: {search_score}")
                    
                    # 검색 결과가 관련성이 없고 재시도 가능한 경우
                    if not search_relevance and retry_count < max_retries:
                        logger.info("GeneralAgent: Search results not relevant, retrying with different search strategy...")
                        retry_count += 1
                        continue
                    
                    # 검색 결과를 chat history에 추가
                    formatted_history_with_search = formatted_history.copy()
                    formatted_history_with_search.append(f"assistant: [참고 문서 원문]\n{search_result}")
                    
                    # 응답 생성
                    response = self.agent_executor.invoke({
                        "question": question,
                        "chat_history": formatted_history_with_search
                    })
                    generated_response = response["output"]
                    
                    # 응답 품질 검증
                    response_quality, response_score = self.validator.validate_response_quality(
                        question, search_result, generated_response
                    )
                    logger.info(f"GeneralAgent: Response validation - Quality: {response_quality}, Score: {response_score}")
                    
                    # 재시도 여부 결정
                    should_retry = self.validator.should_retry(
                        search_relevance, search_score,
                        response_quality, response_score,
                        retry_count
                    )
                    
                    if should_retry and retry_count < max_retries:
                        logger.info("GeneralAgent: Response quality insufficient, retrying with different search strategy...")
                        retry_count += 1
                        continue
                    else:
                        # 최종 응답 반환
                        if not response_quality:
                            logger.warning("GeneralAgent: Response quality validation failed, but returning response due to retry limit")
                            return f"{generated_response}\n\n[참고: 이 응답은 검증 기준을 완전히 충족하지 못할 수 있습니다.]"
                        else:
                            return generated_response
                            
                except Exception as e:
                    logger.error(f"GeneralAgent: Error in RAG search: {e}")
                    if retry_count < max_retries:
                        retry_count += 1
                        continue
                    else:
                        # RAG 실패 시 일반 대화로 처리
                        return self._generate_general_response(question, chat_history)
            else:
                # RAG 도구가 없는 경우 일반 대화로 처리
                logger.info("GeneralAgent: No RAG tool available, using general conversation")
                return self._generate_general_response(question, chat_history)
        
        # 모든 재시도가 실패한 경우
        return "죄송합니다. 적절한 답변을 생성할 수 없습니다. 질문을 다시 한 번 확인해주세요."
    
    def _generate_general_response(self, question: str, chat_history: List) -> str:
        """일반 대화로 응답을 생성합니다."""
        try:
            # 참고 문서가 있는지 확인
            reference_texts = []
            for msg in chat_history:
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "assistant"
                    and msg.get("content", "").startswith("[참고 문서")
                ):
                    reference_texts.append(msg["content"])
            
            # system 프롬프트 동적 생성
            if reference_texts:
                system_prompt = (
                    "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. "
                    "아래 참고 문서 내용을 반드시 참고하여 사용자의 질문에 답변하세요.\n"
                    f"참고 문서:\n{chr(10).join(reference_texts)}"
                )
            else:
                system_prompt = "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 최선을 다해 답변해주세요."
            
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # 이전 대화 기록 추가
            for msg in chat_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
            
            # 현재 질문 추가
            messages.append({"role": "user", "content": question})
            
            # LLM 호출
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in general response generation: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
    
    def can_handle(self, question: str) -> bool:
        """일반 질문인지 확인합니다."""
        # 법령 관련 키워드가 없는 경우 일반 질문으로 간주
        legal_keywords = [
            "법", "법령", "법조문", "조항", "법률", "규정", "법원", "판례",
            "민법", "형법", "상법", "행정법", "헌법", "소송", "계약", "손해배상",
            "책임", "권리", "의무", "처벌", "벌칙", "제재", "처분", "행정처분"
        ]
        
        return not any(keyword in question for keyword in legal_keywords) 