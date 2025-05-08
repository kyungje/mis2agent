from openai import AsyncOpenAI
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4"  # 기본 모델 설정
        
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        주어진 메시지 목록을 기반으로 AI 응답을 생성합니다.
        
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
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # 응답 메시지 로깅
            response_content = response.choices[0].message.content
            logger.info(f"AI Response: {response_content}")
            
            return response_content
        except Exception as e:
            logger.error(f"Agent error: {str(e)}")
            raise 