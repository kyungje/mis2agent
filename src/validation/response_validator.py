from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

class ResponseValidator:
    """응답 검증을 위한 클래스"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0
        )
    
    def validate_search_relevance(self, query: str, search_results: str) -> Tuple[bool, float]:
        """
        vectordb 검색 결과가 질문과 관련성이 있는지 검증합니다.
        
        Args:
            query: 사용자 질문
            search_results: 검색된 문서 내용
            
        Returns:
            Tuple[bool, float]: (관련성 여부, 관련성 점수 0-1)
        """
        try:
            prompt = f"""다음 질문과 검색 결과의 관련성을 평가해주세요.

질문: {query}

검색 결과:
{search_results}

평가 기준:
1. 검색 결과가 질문에 대한 답변을 제공할 수 있는 정보를 포함하고 있는가?
2. 검색 결과의 내용이 질문의 주제와 일치하는가?
3. 검색 결과가 질문에 대한 구체적이고 유용한 정보를 제공하는가?

평가 결과를 다음 형식으로 출력해주세요:
관련성: [yes/no]
점수: [0.0-1.0 사이의 숫자]
이유: [간단한 설명]"""

            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # 응답 파싱
            lines = content.split('\n')
            relevance = False
            score = 0.0
            
            for line in lines:
                if line.startswith('관련성:'):
                    relevance = 'yes' in line.lower()
                elif line.startswith('점수:'):
                    try:
                        score = float(line.split(':')[1].strip())
                    except:
                        score = 0.0
            
            logger.info(f"Search relevance validation - Query: {query[:50]}..., Relevance: {relevance}, Score: {score}")
            return relevance, score
            
        except Exception as e:
            logger.error(f"Error in validate_search_relevance: {e}")
            return False, 0.0
    
    def validate_response_quality(self, query: str, search_results: str, response: str) -> Tuple[bool, float]:
        """
        생성된 응답의 품질을 검증합니다.
        
        Args:
            query: 사용자 질문
            search_results: 검색된 문서 내용
            response: 생성된 응답
            
        Returns:
            Tuple[bool, float]: (품질 통과 여부, 품질 점수 0-1)
        """
        try:
            prompt = f"""다음 응답의 품질을 평가해주세요.

질문: {query}

참고 문서:
{search_results}

생성된 응답:
{response}

평가 기준:
1. 응답이 질문에 정확히 답변하고 있는가?
2. 응답이 참고 문서의 내용을 정확히 반영하고 있는가?
3. 응답이 논리적이고 일관성이 있는가?
4. 응답이 명확하고 이해하기 쉬운가?
5. 응답이 참고 문서의 범위를 벗어나지 않는가?

평가 결과를 다음 형식으로 출력해주세요:
품질: [pass/fail]
점수: [0.0-1.0 사이의 숫자]
이유: [간단한 설명]"""

            response_eval = self.llm.invoke(prompt)
            content = response_eval.content.strip()
            
            # 응답 파싱
            lines = content.split('\n')
            quality_pass = False
            score = 0.0
            
            for line in lines:
                if line.startswith('품질:'):
                    quality_pass = 'pass' in line.lower()
                elif line.startswith('점수:'):
                    try:
                        score = float(line.split(':')[1].strip())
                    except:
                        score = 0.0
            
            logger.info(f"Response quality validation - Query: {query[:50]}..., Quality: {quality_pass}, Score: {score}")
            return quality_pass, score
            
        except Exception as e:
            logger.error(f"Error in validate_response_quality: {e}")
            return False, 0.0
    
    def should_retry(self, search_relevance: bool, search_score: float, 
                    response_quality: bool, response_score: float,
                    retry_count: int = 0) -> bool:
        """
        재시도 여부를 결정합니다.
        
        Args:
            search_relevance: 검색 결과 관련성
            search_score: 검색 결과 점수
            response_quality: 응답 품질 통과 여부
            response_score: 응답 품질 점수
            retry_count: 현재 재시도 횟수
            
        Returns:
            bool: 재시도 여부
        """
        # 최대 재시도 횟수 제한
        if retry_count >= 1:
            return False
        
        # 검색 결과가 관련성이 없거나 점수가 낮은 경우
        if not search_relevance or search_score < 0.5:
            return True
        
        # 응답 품질이 낮거나 점수가 낮은 경우
        if not response_quality or response_score < 0.6:
            return True
        
        return False 