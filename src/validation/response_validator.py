from typing import Dict, List, Tuple, Optional
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

class ResponseValidator:
    """응답 검증을 위한 클래스"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
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
            # 빠른 휴리스틱 검증 먼저 수행
            quick_relevance, quick_score = self._quick_relevance_check(query, search_results)
            
            # 휴리스틱 검증에서 명확하게 관련성이 있거나 없으면 LLM 호출 생략
            if quick_score > 0.8:
                logger.info(f"Quick validation passed - Query: {query[:50]}..., Score: {quick_score}")
                return True, quick_score
            elif quick_score < 0.2:
                logger.info(f"Quick validation failed - Query: {query[:50]}..., Score: {quick_score}")
                return False, quick_score
            
            # 경계선 케이스만 LLM으로 정확한 검증 수행
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
    
    def validate_comparison_response(self, query: str, search_results: str, response: str) -> Tuple[bool, float]:
        """
        출처별 비교 분석 응답의 품질을 검증합니다.
        
        Args:
            query: 사용자 질문
            search_results: 검색된 문서 내용
            response: 생성된 응답
            
        Returns:
            Tuple[bool, float]: (품질 통과 여부, 품질 점수 0-1)
        """
        try:
            prompt = f"""다음 비교 분석 응답의 품질을 평가해주세요.

질문: {query}

참고 문서:
{search_results}

생성된 응답:
{response}

평가 기준:
1. 응답이 여러 출처의 차이점을 명확히 비교하고 있는가?
2. 응답이 공통점과 차이점을 구분하여 설명하고 있는가?
3. 각 출처의 특성이 정확히 반영되고 있는가?
4. 비교 분석이 논리적이고 체계적인가?
5. 출처별 정보가 명확히 구분되어 있는가?
6. 응답이 참고 문서의 내용을 정확히 반영하고 있는가?

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
            
            logger.info(f"Comparison response validation - Query: {query[:50]}..., Quality: {quality_pass}, Score: {score}")
            return quality_pass, score
            
        except Exception as e:
            logger.error(f"Error in validate_comparison_response: {e}")
            return False, 0.0
    
    def is_comparison_query(self, query: str) -> bool:
        """
        질문이 비교 분석을 요구하는지 판단합니다.
        
        Args:
            query: 사용자 질문
            
        Returns:
            bool: 비교 분석 요구 여부
        """
        comparison_keywords = [
            "비교", "차이", "다른", "구분", "차이점", "공통점", "상이", "별도",
            "각각", "각기", "개별", "지역별", "시기별", "버전별", "개정",
            "변경", "수정", "업데이트", "최신", "이전", "구버전", "신규"
        ]
        
        return any(keyword in query for keyword in comparison_keywords)
    
    def should_retry(self, search_relevance: bool, search_score: float, 
                    response_quality: bool, response_score: float,
                    retry_count: int = 0, search_strategy: str = "default") -> bool:
        """
        재시도 여부를 결정합니다.
        
        Args:
            search_relevance: 검색 결과 관련성
            search_score: 검색 결과 점수
            response_quality: 응답 품질 통과 여부
            response_score: 응답 품질 점수
            retry_count: 현재 재시도 횟수
            search_strategy: 현재 사용된 검색 전략
            
        Returns:
            bool: 재시도 여부
        """
        # 최대 재시도 횟수 제한 (2회로 복구하여 3가지 전략 모두 사용)
        if retry_count >= 2:
            return False
        
        # comparison 전략은 특수한 목적이므로 재시도하지 않음
        if search_strategy == "comparison":
            logger.info("Comparison strategy used - no retry regardless of quality")
            return False
        
        # 더 관대한 기준 적용 (성능 향상을 위해)
        # 검색 결과가 완전히 관련성이 없는 경우만 재시도
        if not search_relevance and search_score < 0.3:
            return True
        
        # 응답 품질이 매우 낮은 경우만 재시도
        if not response_quality and response_score < 0.4:
            return True
        
        return False 
    
    def _quick_relevance_check(self, query: str, search_results: str) -> Tuple[bool, float]:
        """
        빠른 휴리스틱 방식으로 관련성을 검증합니다.
        """
        if not search_results or len(search_results.strip()) < 10:
            return False, 0.0
        
        # 질문에서 키워드 추출
        query_words = set(query.lower().split())
        results_lower = search_results.lower()
        
        # 키워드 매칭 점수 계산
        matched_words = 0
        for word in query_words:
            if len(word) > 1 and word in results_lower:
                matched_words += 1
        
        # 기본 점수
        match_score = matched_words / len(query_words) if query_words else 0
        
        # 문서 길이 보정 (너무 짧으면 관련성 낮음)
        length_bonus = min(len(search_results) / 500, 1.0) * 0.2
        
        # 최종 점수
        final_score = min(match_score + length_bonus, 1.0)
        
        return final_score > 0.5, final_score 