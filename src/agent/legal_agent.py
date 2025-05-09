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
        
        # RAG 도구 초기화
        self.vectorstore = FAISS.load_local(vectorstore_path, OpenAIEmbeddings())
        self.rag_tool = LegalRAGTool(self.vectorstore)
        
        # Agent 프롬프트 로드
        self.prompt = load_prompt("src/agent/prompts/agentic-rag-prompt-legal.yaml")
        
        # Agent 생성
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=self.prompt,
            tools=[self.rag_tool]
        )
        
        # Agent 실행기 생성
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[self.rag_tool],
            verbose=True
        )
    
    def run(self, question: str, chat_history: Optional[List] = None) -> str:
        """질문에 대한 답변을 생성합니다."""
        if chat_history is None:
            chat_history = []
            
        response = self.agent_executor.invoke({
            "question": question,
            "chat_history": chat_history
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