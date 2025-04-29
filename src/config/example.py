from pydantic import BaseSettings
from typing import Optional

class ExampleSettings(BaseSettings):
    """예제 설정 클래스"""
    
    # 기본 설정
    APP_NAME: str = "AI Agent"
    DEBUG: bool = False
    
    # API 설정
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # OpenAI 설정
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True 