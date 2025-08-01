"""
성능 최적화 관련 설정
"""

# RAG 검색 설정
RAG_SEARCH_CONFIG = {
    "max_results": 2,  # 검색 결과 수 제한
    "chunk_size": 400,  # 텍스트 청크 크기
    "chunk_overlap": 80,  # 청크 중복 크기
    "embedding_batch_size": 1000,  # 임베딩 배치 크기
}

# 검증 설정
VALIDATION_CONFIG = {
    "max_retries": 1,  # 최대 재시도 횟수
    "quick_validation_enabled": True,  # 빠른 검증 활성화
    "relevance_threshold": 0.3,  # 관련성 임계값
    "quality_threshold": 0.4,  # 품질 임계값
}

# API 응답 설정
API_CONFIG = {
    "response_timeout": 30,  # 응답 타임아웃 (초)
    "max_concurrent_requests": 10,  # 최대 동시 요청 수
}

# UI 설정
UI_CONFIG = {
    "stream_chunk_size": 3,  # 스트리밍 청크 크기
    "stream_delay": 0.005,  # 스트리밍 지연시간 (초)
    "cache_enabled": True,  # 캐시 활성화
}

# 로깅 설정
LOGGING_CONFIG = {
    "performance_logging": True,  # 성능 로깅 활성화
    "detailed_timing": False,  # 상세 타이밍 로깅 (개발용)
} 