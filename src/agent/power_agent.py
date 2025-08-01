from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt, ChatPromptTemplate
from ..tools.power_rag_tools import PowerRAGTool
from ..validation.response_validator import ResponseValidator
from .base_agent import BaseAgent
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class PowerAgent(BaseAgent):
    def __init__(self, index_path: str):
        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0
        )
        
        # RAG 도구 초기화
        try:
            self.rag_tool = PowerRAGTool(index_path)
        except Exception as e:
            logger.warning(f"PowerRAGTool could not be loaded. Error: {e}")
            self.rag_tool = None
        
        # 검증기 초기화
        self.validator = ResponseValidator()
        
        # 프롬프트 템플릿 로드
        prompt_data = load_prompt("src/agent/prompts/agentic-rag-prompt-power.yaml", encoding='utf-8')
        self.prompt_template = prompt_data.template
    
    def run(self, question: str, chat_history: Optional[List] = None) -> str:
        """질문에 대한 답변을 생성합니다."""
        if chat_history is None:
            chat_history = []
            
        # chat history 포맷팅 - 최근 5개 메시지만 유지
        formatted_history = []
        for message in chat_history[-5:]:  # 최근 5개 메시지만 사용
            if isinstance(message, dict):
                role = message.get('role', 'user')
                content = message.get('content', '')
                formatted_history.append(f"{role}: {content}")
            else:
                formatted_history.append(f"user: {message}")
        
        max_retries = 0  # 재시도 횟수를 0으로 설정
        retry_count = 0
        
        # 비교 질문인지 확인
        is_comparison = self.validator.is_comparison_query(question)
        
        while retry_count <= max_retries:
            logger.info(f"Attempt {retry_count + 1} for question: {question[:50]}...")
            
            # 검색 전략 선택 - 비교 질문이면 comparison, 아니면 기본 MMR
            current_strategy = "comparison" if is_comparison else "default"
            logger.info(f"Using search strategy: {current_strategy}")
            
            # vectordb 검색 수행
            search_result = self.rag_tool._run(question, search_strategy=current_strategy)
            
            # 검색 결과 관련성 검증
            search_relevance, search_score = self.validator.validate_search_relevance(question, search_result)
            logger.info(f"Search validation - Relevance: {search_relevance}, Score: {search_score}")
            
            # 검색 결과가 관련성이 없고 재시도 가능한 경우
            if not search_relevance and retry_count < max_retries:
                logger.info("Search results not relevant, retrying with different search strategy...")
                retry_count += 1
                continue
            
            # 검색 결과를 chat history에 추가
            formatted_history_with_search = formatted_history.copy()
            formatted_history_with_search.append(f"assistant: [참고 문서 원문]\n{search_result}")

            # 응답 생성
            try:
                # 프롬프트 포맷팅
                formatted_prompt = self.prompt_template.format(
                    question=question,
                    chat_history="\n".join(formatted_history_with_search),
                    agent_scratchpad=""
                )
                
                # LLM 직접 호출
                response = self.llm.invoke([{"role": "user", "content": formatted_prompt}])
                generated_response = response.content
                
                # 응답 품질 검증
                response_quality, response_score = self.validator.validate_response_quality(
                    question, search_result, generated_response
                )
                logger.info(f"Response validation - Quality: {response_quality}, Score: {response_score}")
                
                # 재시도 여부 결정
                should_retry = self.validator.should_retry(
                    search_relevance, search_score,
                    response_quality, response_score,
                    retry_count
                )
                
                if should_retry and retry_count < max_retries:
                    logger.info("Response quality insufficient, retrying with different search strategy...")
                    retry_count += 1
                    continue
                else:
                    # 최종 응답 반환
                    if not response_quality:
                        logger.warning("Response quality validation failed, but returning response due to retry limit")
                        return f"{generated_response}\n\n[참고: 이 응답은 검증 기준을 완전히 충족하지 못할 수 있습니다.]"
                    else:
                        return generated_response
                        
            except Exception as e:
                logger.error(f"Error in response generation: {e}")
                if retry_count < max_retries:
                    retry_count += 1
                    continue
                else:
                    return "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        
        # 모든 재시도가 실패한 경우
        return "죄송합니다. 적절한 답변을 생성할 수 없습니다. 질문을 다시 한 번 확인해주세요." 