from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from ..tools.rag_tools import LegalRAGTool
from .base_agent import BaseAgent
from typing import List, Optional

class LegalAgent(BaseAgent):
    def __init__(self, index_path: str):
        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0
        )
        
        # RAG 도구 초기화
        try:
            self.rag_tool = LegalRAGTool(index_path)
        except Exception as e:
            print(f"Warning: FAISS index could not be loaded. Error: {e}")
            self.rag_tool = None
        
        # Agent 프롬프트 로드
        self.prompt = load_prompt("src/agent/prompts/agentic-rag-prompt-legal.yaml", encoding='utf-8')
        
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
        
        if not self.rag_tool:
            return "죄송합니다. 현재 서울특별시 도시가스회사 공급규정 데이터베이스가 설정되지 않아 관련 질문에 답변할 수 없습니다."
        
        if not self.can_handle(question):
            return "죄송합니다. 이 에이전트는 서울특별시 도시가스회사 공급규정에 대한 질문만 처리할 수 있습니다."
        
        # 대화 기록을 문자열 형식으로 변환하고 최근 3개만 유지
        formatted_history = []
        for msg in chat_history[-3:]:  # 최근 3개 메시지만 사용
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted_history.append(f"{msg['role']}: {msg['content']}")
        
        response = self.agent_executor.invoke({
            "question": question,
            "chat_history": formatted_history
        })
        
        return response["output"]
    
    def can_handle(self, question: str) -> bool:
        """서울특별시 도시가스회사 공급규정 관련 질문인지 AI를 통해 확인합니다."""
        try:
            # 질문 의도 파악을 위한 프롬프트
            prompt = f"""다음 질문이 서울특별시 도시가스회사 공급규정과 관련된 질문인지 판단해주세요.
            질문: {question}
            
            다음 중 하나라도 해당되면 'yes'를, 그렇지 않으면 'no'를 출력해주세요:
            1. 서울특별시 도시가스회사의 공급규정, 요금, 계약 등에 관한 질문
            2. 가스 공급, 사용, 안전, 설비 등에 관한 질문
            3. 도시가스 관련 법규나 규정에 관한 질문
            
            답변은 반드시 'yes' 또는 'no'로만 해주세요."""

            # LLM을 통해 질문 의도 파악
            response = self.llm.invoke(prompt)
            result = response.content.strip().lower()
            
            return result == 'yes'
            
        except Exception as e:
            print(f"Error in can_handle: {e}")
            # 오류 발생 시 기본 키워드 기반 판단으로 폴백
            keywords = [
                "서울특별시", "도시가스", "공급규정", "가스공급", "가스요금",
                "가스사용", "가스계량", "가스설비", "가스안전", "가스공급계약"
            ]
            return any(keyword in question for keyword in keywords) 