from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .base_agent import BaseAgent
from typing import List, Optional

class GeneralAgent(BaseAgent):
    def __init__(self):
        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo",
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
    
    def can_handle(self, question: str) -> bool:
        """일반 질문인지 확인합니다."""
        # 법령 관련 키워드가 없는 경우 일반 질문으로 간주
        legal_keywords = [
            "법", "법령", "법조문", "조항", "법률", "규정", "법원", "판례",
            "민법", "형법", "상법", "행정법", "헌법", "소송", "계약", "손해배상",
            "책임", "권리", "의무", "처벌", "벌칙", "제재", "처분", "행정처분"
        ]
        
        return not any(keyword in question for keyword in legal_keywords) 