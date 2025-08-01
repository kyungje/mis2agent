"""
성능 모니터링 유틸리티
"""
import time
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

def measure_time(func_name: str = None):
    """함수 실행 시간을 측정하는 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                name = func_name or func.__name__
                logger.info(f"⚡ {name} 실행시간: {execution_time:.3f}초")
                
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                name = func_name or func.__name__
                logger.error(f"❌ {name} 실행실패 ({execution_time:.3f}초): {str(e)}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                name = func_name or func.__name__
                logger.info(f"⚡ {name} 실행시간: {execution_time:.3f}초")
                
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                name = func_name or func.__name__
                logger.error(f"❌ {name} 실행실패 ({execution_time:.3f}초): {str(e)}")
                raise
        
        # async 함수인지 확인
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class PerformanceMonitor:
    """성능 통계를 수집하는 클래스"""
    
    def __init__(self):
        self.stats = {
            "total_requests": 0,
            "total_time": 0,
            "average_time": 0,
            "min_time": float('inf'),
            "max_time": 0
        }
    
    def record_request(self, execution_time: float):
        """요청 통계를 기록합니다."""
        self.stats["total_requests"] += 1
        self.stats["total_time"] += execution_time
        self.stats["average_time"] = self.stats["total_time"] / self.stats["total_requests"]
        self.stats["min_time"] = min(self.stats["min_time"], execution_time)
        self.stats["max_time"] = max(self.stats["max_time"], execution_time)
    
    def get_stats(self) -> dict:
        """현재 통계를 반환합니다."""
        return self.stats.copy()
    
    def reset_stats(self):
        """통계를 초기화합니다."""
        self.stats = {
            "total_requests": 0,
            "total_time": 0,
            "average_time": 0,
            "min_time": float('inf'),
            "max_time": 0
        }

# 글로벌 성능 모니터 인스턴스
performance_monitor = PerformanceMonitor() 