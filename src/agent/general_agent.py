from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .base_agent import BaseAgent
from typing import List, Optional

class GeneralAgent(BaseAgent):
    def __init__(self):
        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7
        )
        
        # 프롬프트 템플릿 생성
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 최선을 다해 답변해주세요."),
            ("human", "{question}")
        ])
    
    def run(self, question: str, chat_history: Optional[List] = None) -> str:
        """질문에 대한 답변을 생성합니다."""
        if chat_history is None:
            chat_history = []
            
        chain = self.prompt | self.llm
        response = chain.invoke({"question": question})
        
        return response.content
    
    def can_handle(self, question: str) -> bool:
        """일반 질문인지 확인합니다."""
        # 법령 관련 키워드가 없는 경우 일반 질문으로 간주
        legal_keywords = [
            "법", "법령", "법조문", "조항", "법률", "규정", "법원", "판례",
            "민법", "형법", "상법", "행정법", "헌법", "소송", "계약", "손해배상",
            "책임", "권리", "의무", "처벌", "벌칙", "제재", "처분", "행정처분"
        ]
        
        return not any(keyword in question for keyword in legal_keywords) 