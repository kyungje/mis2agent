import os
import pytest
from src.config.example import ExampleSettings

def test_default_settings():
    """기본 설정 테스트"""
    settings = ExampleSettings()
    assert settings.APP_NAME == "AI Agent"
    assert settings.DEBUG is False
    assert settings.API_HOST == "0.0.0.0"
    assert settings.API_PORT == 8000
    assert settings.OPENAI_MODEL == "gpt-3.5-turbo"
    assert settings.LOG_LEVEL == "INFO"

def test_custom_settings():
    """사용자 정의 설정 테스트"""
    os.environ["APP_NAME"] = "Custom Agent"
    os.environ["DEBUG"] = "true"
    os.environ["API_PORT"] = "9000"
    
    settings = ExampleSettings()
    assert settings.APP_NAME == "Custom Agent"
    assert settings.DEBUG is True
    assert settings.API_PORT == 9000
    
    # 환경 변수 정리
    del os.environ["APP_NAME"]
    del os.environ["DEBUG"]
    del os.environ["API_PORT"]

def test_missing_required_settings():
    """필수 설정 누락 테스트"""
    with pytest.raises(ValueError):
        ExampleSettings(OPENAI_API_KEY=None) 