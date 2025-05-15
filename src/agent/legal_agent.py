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
        
        # 대화 기록을 문자열 형식으로 변환
        formatted_history = []
        for msg in chat_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted_history.append(f"{msg['role']}: {msg['content']}")
        
        response = self.agent_executor.invoke({
            "question": question,
            "chat_history": formatted_history
        })
        
        return response["output"]
    
    def can_handle(self, question: str) -> bool:
        """서울특별시 도시가스회사 공급규정 관련 질문인지 확인합니다."""
        # 관련 키워드 목록
        keywords = [
            "서울특별시", "도시가스", "공급규정", "가스공급", "가스요금",
            "가스사용", "가스계량", "가스설비", "가스안전", "가스공급계약"
        ]
        
        # 질문에 관련 키워드가 포함되어 있는지 확인
        return any(keyword in question for keyword in keywords) 