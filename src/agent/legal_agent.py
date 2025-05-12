from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from ..tools.rag_tools import LegalRAGTool
from .base_agent import BaseAgent
from typing import List, Optional

class LegalAgent(BaseAgent):
    def __init__(self, vectorstore_path: str):
        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0
        )
        
        # 더미 RAG 도구 초기화
        try:
            self.vectorstore = FAISS.load_local(vectorstore_path, OpenAIEmbeddings())
        except Exception as e:
            print(f"Warning: FAISS vectorstore could not be loaded. Using dummy vectorstore. Error: {e}")
            self.vectorstore = None  # 또는 더미 객체로 대체
        
        self.rag_tool = LegalRAGTool(self.vectorstore) if self.vectorstore else None
        
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
            return "죄송합니다. 현재 법령 데이터베이스가 설정되지 않아 법령 관련 질문에 답변할 수 없습니다. 일반적인 대화는 계속 가능합니다."
        
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
        """법령 관련 질문인지 확인합니다."""
        # 법령 관련 키워드 목록
        legal_keywords = [
            "법", "법령", "법조문", "조항", "법률", "규정", "법원", "판례",
            "민법", "형법", "상법", "행정법", "헌법", "소송", "계약", "손해배상",
            "책임", "권리", "의무", "처벌", "벌칙", "제재", "처분", "행정처분"
        ]
        
        # 질문에 법령 관련 키워드가 포함되어 있는지 확인
        return any(keyword in question for keyword in legal_keywords) 