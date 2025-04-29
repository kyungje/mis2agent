# AI 에이전트 프로젝트

이 프로젝트는 Poetry를 사용하여 관리되는 AI 에이전트 구현입니다.

## 설치 방법

1. Poetry 설치 (아직 설치하지 않은 경우):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 프로젝트 의존성 설치:
```bash
poetry install
```

## 프로젝트 구조

```
.
├── src/
│   ├── agent/          # AI 에이전트 핵심 로직
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── openai_agent.py
│   ├── api/           # API 관련 코드
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── config/        # 설정 관련 코드
│   │   ├── __init__.py
│   │   └── settings.py
│   └── ui/           # 사용자 인터페이스 관련 코드
│       ├── __init__.py
│       └── components.py
├── tests/            # 테스트 코드
│   ├── __init__.py
│   └── test_agent.py
├── pyproject.toml    # Poetry 프로젝트 설정
└── README.md         # 프로젝트 문서
```

## 사용 방법

1. `.env` 파일을 생성하고 필요한 API 키를 설정합니다:
```
OPENAI_API_KEY=your_api_key_here
```

2. Poetry 환경에서 서버 실행:
```bash
poetry run python -m src.api.main
```

## 개발 도구

- 코드 포맷팅:
```bash
poetry run black .
poetry run isort .
```

- 타입 체크:
```bash
poetry run mypy .
```

- 린트:
```bash
poetry run flake8
```

- 테스트 실행:
```bash
poetry run pytest
```

## API 문서

서버가 실행되면 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 