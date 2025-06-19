# AI 에이전트 프로젝트

## 프로젝트 구조
```
ai-agent/
├── src/
│   ├── agent/      # AI 에이전트 관련 코드
│   ├── api/        # FastAPI 백엔드
│   ├── config/     # 설정 파일
│   ├── index/      # FAISS 인덱스 파일
│   └── ui/         # Streamlit UI
├── tests/          # 테스트 코드
├── .env           # 환경 변수
├── pyproject.toml # 프로젝트 설정
└── README.md      # 프로젝트 문서
```

## 설치 방법

1. Poetry 설치 (아직 설치하지 않은 경우):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 프로젝트 의존성 설치:
```bash
poetry install
```

3. 환경 변수 설정:
`.env` 파일을 생성하고 다음 내용을 추가합니다:

```
OPENAI_API_KEY=your_api_key_here
GAS_INDEX_PATH=src/vectordb/db/gas_index
POWER_INDEX_PATH=src/vectordb/db/power_index
OTHERS_INDEX_PATH=src/vectordb/db/other_index
```

**참고:**
- `GAS_INDEX_PATH`: 도시가스 관련 벡터 데이터베이스 경로
- `POWER_INDEX_PATH`: 전력 관련 벡터 데이터베이스 경로
- `OTHERS_INDEX_PATH`: 기타 문서들에 대한 벡터 데이터베이스 경로 (관련성 검증 후 사용)

## 추가 모듈 설치

프로젝트 실행에 필요한 추가 모듈을 설치하려면 다음 명령어를 사용하세요:
```bash
poetry add langchain_openai langchain_community
```

## 실행 방법

### FastAPI 서버 실행
다음 두 가지 방법 중 하나를 선택하여 FastAPI 서버를 실행할 수 있습니다:

1. Python 모듈로 실행:
```bash
poetry run python -m src.api.chat
```

2. Uvicorn으로 직접 실행 (코드 변경 시 자동 재시작):
```bash
poetry run uvicorn src.api.chat:app --host 0.0.0.0 --port 8000 --reload
```

서버가 실행되면 다음 URL에서 접속할 수 있습니다:
- API 서버: http://localhost:8000
- API 문서: http://localhost:8000/docs
- 대체 API 문서: http://localhost:8000/redoc

### Streamlit UI 실행
새 터미널을 열고 다음 명령어를 실행합니다:
```bash
poetry run streamlit run src/ui/chat_ui.py
```

Streamlit UI는 자동으로 브라우저에서 열리거나, 터미널에 표시된 URL(기본값: http://localhost:8501)로 접속할 수 있습니다.

## 개발 환경 설정

1. 가상 환경 활성화:
```bash
poetry shell
```

2. 코드 포맷팅:
```bash
poetry run black .
poetry run isort .
```

3. 타입 체크:
```bash
poetry run mypy .
```

4. 테스트 실행:
```bash
poetry run pytest
```

## 주의사항
- FastAPI 서버와 Streamlit UI를 동시에 실행하려면 두 개의 터미널이 필요합니다.
- FastAPI 서버가 실행 중이어야 Streamlit UI가 정상적으로 작동합니다.
- OpenAI API 키가 올바르게 설정되어 있어야 합니다.
- 도메인별 인덱스 경로가 설정되지 않은 경우, 해당 도메인의 질문은 일반 Agent로 처리됩니다.
- `OTHERS_INDEX_PATH`의 경우, 검색 결과가 질문과 관련이 없으면 벡터 DB를 사용하지 않고 일반 대화로 처리됩니다.





