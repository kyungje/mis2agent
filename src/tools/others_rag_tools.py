from typing import List, Dict, Any, Optional, ClassVar, Type, Union
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import Field, PrivateAttr
import os
import logging
import traceback
import re

logger = logging.getLogger(__name__)

def extract_region_from_query(query: str) -> Optional[str]:
    """질문에서 지역 정보를 추출합니다."""
    # 한국의 주요 지역들
    regions = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
        "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", 
        "대전광역시", "울산광역시", "세종특별자치시",
        "경기도", "강원도", "충청북도", "충청남도", "전라북도", "전라남도", 
        "경상북도", "경상남도", "제주특별자치도"
    ]
    
    for region in regions:
        if region in query:
            return region
    
    return None

def extract_region_from_filename(filename: str) -> Optional[str]:
    """파일명에서 지역 정보를 추출합니다."""
    # 한국의 주요 지역들
    regions = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
        "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", 
        "대전광역시", "울산광역시", "세종특별자치시",
        "경기도", "강원도", "충청북도", "충청남도", "전라북도", "전라남도", 
        "경상북도", "경상남도", "제주특별자치도"
    ]
    
    # 디버깅을 위한 로그 추가
    logger.info(f"Extracting region from filename: {filename}")
    
    # 파일명을 정규화 (URL 인코딩된 부분을 디코딩)
    normalized_filename = filename.replace('+', ' ').replace('%20', ' ')
    logger.info(f"Normalized filename: {normalized_filename}")
    
    # 더 구체적인 지역명을 우선하기 위해 길이순으로 정렬 (긴 것부터)
    sorted_regions = sorted(regions, key=len, reverse=True)
    for region in sorted_regions:
        if region in normalized_filename:
            logger.info(f"Found region '{region}' in filename")
            return region
    
    logger.warning(f"No region found in filename: {filename}")
    return None

class OthersRAGTool(BaseTool):
    name: ClassVar[str] = "others_document_search"
    description: ClassVar[str] = "기타 문서들에서 관련 정보를 검색합니다."
    
    # 프라이빗 속성으로 정의 (Pydantic 모델에 포함되지 않음)
    _embeddings: Any = PrivateAttr(default=None)
    _vectorstore: Any = PrivateAttr(default=None)
    _retriever: BaseRetriever = PrivateAttr(default=None)
    _history_aware_retriever: BaseRetriever = PrivateAttr(default=None)
    
    def __init__(self, index_path: str):
        super().__init__()
        logger.info(f"Initializing OthersRAGTool with index_path: {index_path}")
        
        try:
            # OpenAI 임베딩 모델 초기화 (FAISS 인덱스 생성 시와 동일한 모델 사용)
            logger.info("Initializing OpenAIEmbeddings model")
            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                chunk_size=1000
            )
            logger.info("OpenAIEmbeddings initialized successfully")
            
            # 인덱스 파일 경로 확인
            logger.info(f"Checking if index_path exists: {os.path.exists(index_path)}")
            if not os.path.exists(index_path):
                logger.warning(f"Index path does not exist: {index_path}")
                logger.info("OthersRAGTool will be initialized without vectorstore")
                self._vectorstore = None
                self._retriever = None
                self._history_aware_retriever = None
                return
            
            # 디렉토리 내용 확인
            try:
                contents = os.listdir(index_path)
                logger.info(f"Contents of index_path: {contents}")
                
                # index.faiss 파일이 있는지 확인
                faiss_files = [f for f in contents if f.endswith('.faiss')]
                if not faiss_files:
                    logger.warning(f"No .faiss files found in {index_path}")
                    logger.info("OthersRAGTool will be initialized without vectorstore")
                    self._vectorstore = None
                    self._retriever = None
                    self._history_aware_retriever = None
                    return
                    
            except Exception as e:
                logger.warning(f"Error reading directory contents: {e}")
                self._vectorstore = None
                self._retriever = None
                self._history_aware_retriever = None
                return
            
            # FAISS 벡터스토어 로드
            logger.info(f"Loading FAISS vectorstore from {index_path}")
            self._vectorstore = FAISS.load_local(
                index_path, 
                self._embeddings,
                allow_dangerous_deserialization=True  # 이 옵션은 안전한 환경에서만 사용
            )
            logger.info("FAISS vectorstore loaded successfully")
            
            # 검색기 설정
            logger.info("Initializing retriever")
            search_kwargs = {
                "k": 40,
                "score_threshold": 0.0
            }
            
            base_retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            
            # MultiQueryRetriever로 래핑하여 다양한 쿼리 생성
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            
            # 기타 문서 관련 커스텀 프롬프트
            QUERY_PROMPT = ChatPromptTemplate.from_messages([
                ("system", 
                 "당신은 기타 문서 검색 쿼리를 생성하는 AI입니다. 주어진 질문을 기반으로 검색 쿼리를 만드세요.\n\n"
                 "매우 중요한 규칙:\n"
                 "- 첫 번째 줄에는 반드시 주어진 질문을 완전히 그대로 써야 합니다.\n"
                 "- 그 다음에 2-3개의 관련 검색어를 추가하세요.\n"
                 "- 번호를 붙여서 각 줄에 하나씩 써주세요.\n\n"
                 "예시:\n"
                 "1. [주어진 질문을 완전히 그대로]\n"
                 "2. [관련 검색어 1]\n"
                 "3. [관련 검색어 2]"),
                ("human", "{question}"),
            ])
            
            self._retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
                prompt=QUERY_PROMPT
            )
            
            # 컨텍스트를 고려한 검색을 위한 history-aware retriever 생성
            contextualize_q_system_prompt = (
                "주어진 대화 기록과 최신 사용자 질문을 바탕으로 "
                "대화 기록의 맥락을 참조할 수 있는 독립적인 질문을 작성하세요. "
                "질문에 답변하지 말고, 필요한 경우에만 다시 공식화하고 "
                "그렇지 않으면 그대로 반환하세요."
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            
            self._history_aware_retriever = create_history_aware_retriever(
                llm, self._retriever, contextualize_q_prompt
            )
            
            logger.info("History-aware retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OthersRAGTool: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("OthersRAGTool will be initialized without vectorstore")
            self._vectorstore = None
            self._retriever = None
            self._history_aware_retriever = None
    
    def _run(self, query: str, search_strategy: str = "default", chat_history: Optional[List] = None) -> str:
        """
        RAG 검색을 수행하고 포맷된 결과를 반환합니다.
        
        Args:
            query: 검색할 질문
            search_strategy: 검색 전략 ("default", "expanded", "keyword", "comparison")
            chat_history: 이전 대화 기록 (컨텍스트 고려를 위해)
        """
        if self._vectorstore is None:
            logger.warning("Vectorstore is not initialized")
            return "벡터 데이터베이스가 초기화되지 않았습니다."
        
        try:
            logger.info(f"Running search query: {query} with strategy: {search_strategy}")
            
            # 질문에서 지역 정보 추출
            target_region = extract_region_from_query(query)
            
            # 현재 질문에 지역이 없으면 채팅 히스토리에서 지역 추출
            if not target_region and chat_history:
                for msg in chat_history[-3:]:  # 최근 3개 메시지에서 지역 찾기
                    if isinstance(msg, dict):
                        content = msg.get('content', '')
                        history_region = extract_region_from_query(content)
                        if history_region:
                            target_region = history_region
                            logger.info(f"Found region from chat history: {target_region}")
                            break
            
            if target_region:
                logger.info(f"Using target region: {target_region}")
            
            # chat_history를 LangChain 메시지 형식으로 변환
            formatted_chat_history = []
            if chat_history:
                for msg in chat_history[-5:]:  # 최근 5개 메시지만 사용
                    if isinstance(msg, dict):
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if role == 'user':
                            formatted_chat_history.append(HumanMessage(content=content))
                        elif role == 'assistant':
                            formatted_chat_history.append(AIMessage(content=content))
                    elif isinstance(msg, str):
                        formatted_chat_history.append(HumanMessage(content=msg))
            
            # 검색 전략에 따른 검색 수행
            if search_strategy == "expanded":
                # 확장된 검색 전략
                docs = self._search_with_expanded_strategy(query)
            elif search_strategy == "keyword":
                # 키워드 기반 검색
                docs = self._search_with_keyword_strategy(query)
            elif search_strategy == "comparison":
                # 출처별 비교를 위한 검색
                docs = self._search_with_comparison_strategy(query)
            else:
                # 기본 검색 - history-aware retriever 사용
                if formatted_chat_history and self._history_aware_retriever:
                    logger.info("Using history-aware retriever")
                    docs = self._history_aware_retriever.invoke({
                        "input": query,
                        "chat_history": formatted_chat_history
                    })
                else:
                    logger.info("Using standard MultiQueryRetriever")
                    docs = self._retriever.get_relevant_documents(query)
            
            # 지역별 필터링 적용
            if target_region and docs:
                logger.info(f"Filtering documents for region: {target_region}")
                filtered_docs = []
                for doc in docs:
                    source = doc.metadata.get('source', '')
                    doc_region = extract_region_from_filename(source)
                    if doc_region and target_region in doc_region:
                        filtered_docs.append(doc)
                        logger.info(f"  ✓ Included: {source} (region: {doc_region})")
                    else:
                        logger.info(f"  ✗ Excluded: {source} (region: {doc_region})")
                
                if filtered_docs:
                    # 채팅 히스토리 컨텍스트 고려하여 관련성 점수로 재정렬
                    if chat_history and len(chat_history) > 0:
                        logger.info("Re-ranking filtered documents with chat history context")
                        scored_docs = []
                        core_query = self._extract_core_query(query, target_region)
                        for doc in filtered_docs:
                            score = self._calculate_relevance_score(doc.page_content, core_query, chat_history)
                            scored_docs.append((doc, score))
                            logger.info(f"  Relevance score for {doc.metadata.get('source', 'unknown')}: {score}")
                        
                        # 점수순으로 정렬
                        scored_docs.sort(key=lambda x: x[1], reverse=True)
                        docs = [doc for doc, score in scored_docs[:10]]  # 상위 10개
                    else:
                        docs = filtered_docs
                    logger.info(f"After region filtering: {len(docs)} documents")
                else:
                    logger.warning(f"No documents found for region: {target_region}, using all documents")
            
            # 관련성 필터링 - 문서 내용이 질문과 관련이 있는지 확인
            if docs:
                logger.info("Filtering documents by relevance")
                relevant_docs = []
                query_keywords = self._extract_keywords(query)
                logger.info(f"Query keywords: {query_keywords}")
                
                for doc in docs:
                    content = doc.page_content.lower()
                    relevance_score = 0
                    
                    # 키워드 매칭으로 관련성 점수 계산
                    for keyword in query_keywords:
                        if keyword.lower() in content:
                            relevance_score += 1
                    
                    # 최소 1개 이상의 키워드가 매칭되면 포함
                    if relevance_score >= 1:
                        relevant_docs.append(doc)
                        logger.info(f"  ✓ Relevant: {doc.metadata.get('source', 'unknown')} (score: {relevance_score})")
                    else:
                        logger.info(f"  ✗ Not relevant: {doc.metadata.get('source', 'unknown')} (score: {relevance_score})")
                
                if relevant_docs:
                    docs = relevant_docs
                    logger.info(f"After relevance filtering: {len(docs)} documents")
                else:
                    logger.warning("No relevant documents found, using all documents")
            
            logger.info(f"Found {len(docs)} relevant documents")
            
            if not docs:
                logger.warning("No relevant documents found")
                return "관련 문서를 찾을 수 없습니다."
            
            formatted_result = self._format_docs(docs)
            logger.info("Documents formatted successfully")
            return formatted_result
        except Exception as e:
            logger.error(f"Error in _run: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"검색 중 오류가 발생했습니다: {str(e)}"
    
    def _search_with_expanded_strategy(self, query: str) -> List[Document]:
        """확장된 검색 전략: 더 많은 결과와 더 넓은 범위로 검색"""
        logger.info("Using expanded search strategy")
        
        # 더 많은 결과를 가져오고 MMR을 사용하여 다양성 확보
        expanded_retriever = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,  # 더 많은 후보
                "fetch_k": 20,  # 더 넓은 범위에서 검색
                "lambda_mult": 0.7  # 다양성과 관련성의 균형
            }
        )
        
        return expanded_retriever.get_relevant_documents(query)
    
    def _search_with_keyword_strategy(self, query: str) -> List[Document]:
        """키워드 기반 검색 전략: 질문에서 핵심 키워드를 추출하여 검색"""
        logger.info("Using keyword search strategy")
        
        # 질문에서 핵심 키워드 추출
        keywords = self._extract_keywords(query)
        logger.info(f"Extracted keywords: {keywords}")
        
        # 각 키워드로 개별 검색 후 결과 통합
        all_docs = []
        for keyword in keywords:
            try:
                docs = self._retriever.get_relevant_documents(keyword)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Error searching for keyword '{keyword}': {e}")
        
        # 중복 제거 및 정렬
        unique_docs = self._remove_duplicates(all_docs)
        return unique_docs[:5]  # 상위 5개 결과만 반환
    
    def _search_with_comparison_strategy(self, query: str) -> List[Document]:
        """출처별 비교를 위한 검색 전략: 여러 출처에서 관련 문서를 찾아 비교 가능하도록 함"""
        logger.info("Using comparison search strategy")
        
        # 지역 정보 추출
        target_region = extract_region_from_query(query)
        
        if target_region:
            logger.info(f"Applying region filtering for comparison: {target_region}")
            # 더 많은 결과를 가져와서 지역 필터링 적용
            comparison_retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 30}  # 필터링 전에 더 많은 결과
            )
            all_docs = comparison_retriever.get_relevant_documents(query)
            
            # 지역별 필터링
            docs = []
            for doc in all_docs:
                source = doc.metadata.get('source', '')
                doc_region = extract_region_from_filename(source)
                if doc_region and (target_region in doc_region or doc_region in target_region):
                    docs.append(doc)
                    logger.info(f"  ✓ Included for comparison: {source} (region: {doc_region})")
        else:
            # 더 많은 결과를 가져와서 출처별로 그룹화
            comparison_retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 15}  # 더 많은 결과
            )
            docs = comparison_retriever.get_relevant_documents(query)
        
        if not docs:
            logger.warning("No documents found in comparison search strategy")
            return []
        
        # 출처별로 그룹화
        source_groups = self._group_by_source(docs)
        
        # 각 출처에서 대표 문서 선택
        representative_docs = self._select_representative_docs(source_groups)
        
        return representative_docs
    
    def _group_by_source(self, docs: List[Document]) -> Dict[str, List[Document]]:
        """문서들을 출처별로 그룹화합니다."""
        source_groups = {}
        
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        return source_groups
    
    def _select_representative_docs(self, source_groups: Dict[str, List[Document]]) -> List[Document]:
        """각 출처에서 대표 문서를 선택합니다."""
        representative_docs = []
        
        for source, docs in source_groups.items():
            if docs:
                # 각 출처에서 가장 관련성 높은 문서 선택
                best_doc = docs[0]  # 이미 유사도 순으로 정렬되어 있음
                representative_docs.append(best_doc)
        
        # 최대 5개 출처까지만 반환
        return representative_docs[:5]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """질문에서 핵심 키워드를 추출합니다."""
        # 질문을 단어로 분할
        words = query.split()
        
        # 2글자 이상의 단어만 키워드로 사용 (조사, 접미사 등 제외)
        keywords = []
        for word in words:
            # 한글, 영문, 숫자만 포함된 단어만 선택
            if len(word) >= 2:
                # 한글, 영문, 숫자, 특수문자 중 하나라도 있으면 포함
                if any(c.isalnum() or c in '가-힣' for c in word):
                    # 너무 일반적인 단어들 제외
                    if word not in ['이것', '그것', '저것', '무엇', '어떤', '어떻게', '왜', '언제', '어디서', '누가']:
                        keywords.append(word)
        
        # 상위 5개 키워드만 반환
        return keywords[:5]
    
    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        """중복된 문서를 제거합니다."""
        seen_contents = set()
        unique_docs = []
        
        for doc in docs:
            # 내용의 해시를 사용하여 중복 확인
            content_hash = hash(doc.page_content[:100])  # 처음 100자만 사용
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _format_docs(self, docs: List[Document]) -> str:
        """검색된 문서들을 포맷팅합니다."""
        logger.info("Formatting documents")
        
        if not docs:
            logger.warning("No documents to format")
            return "관련 문서를 찾을 수 없습니다."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            formatted_doc = f"<document id='{i}'>\n"
            formatted_doc += f"<content>{doc.page_content}</content>\n"
            
            # 메타데이터 정보 추가
            if 'source' in doc.metadata:
                formatted_doc += f"<source>{doc.metadata['source']}</source>\n"
            if 'page' in doc.metadata:
                formatted_doc += f"<page>{doc.metadata['page']}</page>\n"
            if 'title' in doc.metadata:
                formatted_doc += f"<title>{doc.metadata['title']}</title>\n"
            if 'section' in doc.metadata:
                formatted_doc += f"<section>{doc.metadata['section']}</section>\n"
            if 'chapter' in doc.metadata:
                formatted_doc += f"<chapter>{doc.metadata['chapter']}</chapter>\n"
            if 'subsection' in doc.metadata:
                formatted_doc += f"<subsection>{doc.metadata['subsection']}</subsection>\n"
            if 'paragraph' in doc.metadata:
                formatted_doc += f"<paragraph>{doc.metadata['paragraph']}</paragraph>\n"
            if 'line' in doc.metadata:
                formatted_doc += f"<line>{doc.metadata['line']}</line>\n"
            if 'chunk_id' in doc.metadata:
                formatted_doc += f"<chunk_id>{doc.metadata['chunk_id']}</chunk_id>\n"
            if 'date' in doc.metadata:
                formatted_doc += f"<date>{doc.metadata['date']}</date>\n"
            if 'region' in doc.metadata:
                formatted_doc += f"<region>{doc.metadata['region']}</region>\n"
            formatted_doc += "</document>"
            formatted_docs.append(formatted_doc)
        
        logger.info(f"Formatted {len(formatted_docs)} documents")
        return "\n".join(formatted_docs)
    
    def _extract_core_query(self, query: str, region: str) -> str:
        """질문에서 지역명과 회사명을 제거하고 핵심 키워드만 추출합니다."""
        # 지역명과 회사명 패턴들
        region_patterns = [
            region, f"{region}시", f"{region}도", f"{region}지역",
            "시", "도", "지역", "회사", "공사", "기관"
        ]
        
        core_query = query
        for pattern in region_patterns:
            if pattern in core_query:
                core_query = core_query.replace(pattern, " ").strip()
        
        # 연속된 공백 정리
        core_query = " ".join(core_query.split())
        
        if not core_query:
            return query
        
        return core_query
    
    def _calculate_relevance_score(self, content: str, query: str, chat_history: Optional[List] = None) -> float:
        """문서 내용과 쿼리의 관련성 점수를 계산합니다."""
        content_lower = content.lower()
        query_words = query.lower().split()
        
        score = 0.0
        
        # 각 키워드별 매칭 점수
        for word in query_words:
            if len(word) >= 2:  # 2글자 이상만
                if word in content_lower:
                    score += 1.0
                    # 연속된 단어가 있으면 추가 점수
                    if len(word) >= 3:
                        score += 0.5

                # 간단한 동의어 가중치 (일반 문서 관련)
                synonyms_map = {
                    "승인": ["허가", "통지", "서면 통지", "서면으로 통지"],
                    "통지": ["서면 통지", "서면으로 통지"],
                    "신청": ["접수"],
                    "관리": ["운영", "관리운영"],
                    "규정": ["규칙", "기준"]
                }
                for base, syns in synonyms_map.items():
                    if base in word:
                        for syn in syns:
                            if syn in content_lower:
                                score += 0.5
        
        # 연속된 키워드 조합 점수
        for i in range(len(query_words) - 1):
            bigram = f"{query_words[i]} {query_words[i+1]}"
            if bigram in content_lower:
                score += 2.0

        # 한국어 규정 문구 패턴 가중치 (자주 쓰이는 표현)
        phrase_boosts = [
            "신청일로부터", "일 이내", "서면으로 통지", "서면 통지", "며칠 이내"
        ]
        for phrase in phrase_boosts:
            if phrase in content_lower:
                score += 1.5
        
        # 채팅 히스토리 컨텍스트 고려
        if chat_history and len(chat_history) > 0:
            try:
                # 이전 대화에서 언급된 키워드들
                context_keywords = []
                for msg in chat_history[-3:]:  # 최근 3개 메시지만 고려
                    if isinstance(msg, dict):
                        content_msg = msg.get('content', '')
                        # 주요 도메인 키워드 추출 (일반 문서 관련)
                        domain_keywords = ['계약', '연체료', '요금', '신청', '승인', '통지', '책임', '준수', '관리', '운영', '규정']
                        for keyword in domain_keywords:
                            if keyword in content_msg and keyword not in context_keywords:
                                context_keywords.append(keyword)
                
                # 컨텍스트 키워드가 문서에 있으면 추가 점수
                for context_keyword in context_keywords:
                    if context_keyword in content_lower:
                        score += 1.5  # 컨텍스트 일치 보너스
                        logger.debug(f"Context bonus for '{context_keyword}': +1.5")
            except Exception as e:
                logger.warning(f"Error calculating context score: {e}")
        
        return score 