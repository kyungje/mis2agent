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

class LegalRAGTool(BaseTool):
    name: ClassVar[str] = "gas_supply_regulation_search"
    description: ClassVar[str] = "도시가스 공급규정에서 관련 정보를 검색합니다."
    
    # 프라이빗 속성으로 정의 (Pydantic 모델에 포함되지 않음)
    _embeddings: Any = PrivateAttr(default=None)
    _vectorstore: Any = PrivateAttr(default=None)
    _retriever: BaseRetriever = PrivateAttr(default=None)
    _history_aware_retriever: BaseRetriever = PrivateAttr(default=None)
    
    def __init__(self, index_path: str):
        super().__init__()
        logger.info(f"Initializing LegalRAGTool with index_path: {index_path}")
        
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
                logger.info("LegalRAGTool will be initialized without vectorstore")
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
                    logger.info("LegalRAGTool will be initialized without vectorstore")
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
            
            # 검색 파라미터 설정 - 더 많은 결과를 가져와서 필터링
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
            
            # 지역 정보를 보존하는 커스텀 프롬프트
            QUERY_PROMPT = ChatPromptTemplate.from_messages([
                ("system", 
                 "당신은 검색 쿼리를 생성하는 AI입니다. 주어진 질문을 기반으로 검색 쿼리를 만드세요.\n\n"
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
                "검색에 최적화된 독립적인 질문을 작성하세요.\n\n"
                "중요한 원칙:\n"
                "1. 최신 사용자 질문이 최우선입니다 - 새로운 질문이면 대화 기록을 무시하고 그대로 사용하세요\n"
                "2. 대화 기록은 현재 질문이 애매하거나 불완전한 경우에만 참고하세요\n"
                "3. 지역명(서울특별시, 부산 등), 법령명, 조항명 등 구체적 정보는 반드시 보존하세요\n"
                "4. 질문이 명확하면 그대로 반환하고, 꼭 필요한 경우에만 다시 공식화하세요\n\n"
                "예시:\n"
                "- '구체적으로 설명해줘' → 이전 맥락 참고 필요\n"
                "- '서울특별시 공사비 부담에 대해 구체적으로 설명해줘' → 이미 명확함, 그대로 사용"
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
            logger.error(f"Error initializing LegalRAGTool: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("LegalRAGTool will be initialized without vectorstore")
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
                # 기본 검색 - 지역별 필터링 후 키워드 검색
                if target_region:
                    logger.info(f"Using region-specific search for: {target_region}")
                    # 지역별 검색: 해당 지역 문서에서만 검색 (채팅 히스토리 포함)
                    docs = self._search_in_region(query, target_region, chat_history)
                else:
                    # 일반 검색 - 질문이 명확하면 MultiQueryRetriever 우선 사용
                    query_keywords = query.lower().split()
                    specific_keywords = ['계약', '준수', '연체료', '요금', '신청', '승인', '통지', '분담금', '안전관리']
                    is_specific_query = any(keyword in ' '.join(query_keywords) for keyword in specific_keywords)
                    
                    if is_specific_query or len(query_keywords) >= 3:
                        logger.info("Using standard MultiQueryRetriever for specific query")
                        docs = self._retriever.get_relevant_documents(query)
                    elif formatted_chat_history and self._history_aware_retriever:
                        logger.info("Using history-aware retriever for general query")
                        docs = self._history_aware_retriever.invoke({
                            "input": query,
                            "chat_history": formatted_chat_history
                        })
                    else:
                        logger.info("Using standard MultiQueryRetriever as fallback")
                        docs = self._retriever.get_relevant_documents(query)
            
            # 지역별 검색이 아닌 경우에만 추가 필터링 적용
            if not target_region and docs:
                logger.info("Applying additional filtering for non-region-specific search")
                # 간단한 관련성 체크만 수행
                relevant_docs = []
                for doc in docs:
                    content = doc.page_content.lower()
                    # 기본적인 관련성 키워드가 있으면 포함
                    if any(word in content for word in ['가스', '도시가스', '안전', '책임', '관리']):
                        relevant_docs.append(doc)
                
                if relevant_docs:
                    docs = relevant_docs[:5]  # 상위 5개만 유지
            
            logger.info(f"Found {len(docs)} relevant documents")
            
            # 검색된 문서들의 출처를 상세히 로깅
            if docs:
                logger.info("=== Document Sources ===")
                for i, doc in enumerate(docs):
                    source = doc.metadata.get('source', 'unknown')
                    region = extract_region_from_filename(source)
                    logger.info(f"  Doc {i+1}: source={source}, region={region}")
                    # 문서 내용의 일부도 로깅 (디버깅용)
                    content_preview = doc.page_content[:200].replace('\n', ' ')
                    logger.info(f"    Content preview: {content_preview}...")
                    
                    # 유사도 점수도 로깅 (가능한 경우)
                    if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                        logger.info(f"    Similarity score: {doc.metadata['score']}")
            else:
                logger.warning("No documents found after all filtering steps!")
            
            if not docs:
                logger.warning("No relevant documents found")
                return "죄송합니다. 해당 질문에 관련된 정보를 찾을 수 없습니다."
            
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
            # 지역별 검색 적용
            docs = self._search_in_region(query, target_region)
            logger.info(f"After region search: found {len(docs)} documents")
        else:
            # MultiQueryRetriever를 사용하여 다양한 쿼리로 검색
            docs = self._retriever.get_relevant_documents(query)
            logger.info(f"After general search: found {len(docs)} documents")
        
        if not docs:
            logger.warning("No documents found in comparison search strategy")
            return []
        
        # 출처별로 그룹화
        source_groups = self._group_by_source(docs)
        
        # 각 출처에서 대표 문서 선택
        representative_docs = self._select_representative_docs(source_groups)
        
        return representative_docs
    
    def _search_in_region(self, query: str, target_region: str, chat_history: Optional[List] = None) -> List[Document]:
        """특정 지역의 문서에서만 검색합니다."""
        logger.info(f"Searching in region: {target_region}")
        
        # 1. 질문에서 지역명과 회사명을 제거하고 핵심 키워드만 추출
        core_query = self._extract_core_query(query, target_region)
        logger.info(f"Core query (without region/company): {core_query}")
        
        # 2. 채팅 히스토리가 있으면 컨텍스트 고려하여 core_query 보완
        if chat_history and len(chat_history) > 0:
            try:
                # 현재 질문이 매우 짧고 애매한 경우에만 컨텍스트 추가 (2단어 이하)
                # 지역이 있어도 질문이 충분히 구체적이면 컨텍스트 추가하지 않음
                should_add_context = len(core_query.split()) <= 2
                
                if should_add_context:
                    # 이전 대화에서 언급된 주요 키워드 추출 (최대 1개만)
                    previous_context = ""
                    for msg in chat_history[-2:]:  # 최근 2개 메시지만 고려
                        if isinstance(msg, dict) and not previous_context:
                            content = msg.get('content', '')
                            # 주요 키워드 추출 (계약, 연체료, 안전관리, 공사비 등)
                            context_keywords = ['계약', '연체료', '안전관리', '공사비', '분담금', '요금', '신청', '승인', '통지', '책임', '준수']
                            for keyword in context_keywords:
                                if keyword in content and keyword not in core_query:
                                    previous_context = f" {keyword}"
                                    break  # 첫 번째 키워드만 사용
                    
                    if previous_context.strip():
                        enhanced_query = f"{core_query}{previous_context}"
                        logger.info(f"Enhanced query with limited context: {enhanced_query}")
                        core_query = enhanced_query
                    else:
                        logger.info(f"Core query kept as original (no relevant context found): {core_query}")
                else:
                    logger.info(f"Core query kept as original (sufficient specificity): {core_query}")
            except Exception as e:
                logger.warning(f"Error enhancing query with context: {e}")
        
        # 3. 핵심 키워드로 검색 (지역 제한 없이)
        logger.info(f"Searching with core query: '{core_query}'")
        
        # 다양한 검색어로 시도
        search_queries = [
            core_query,
            "열량변경작업 가스사고 연소기 제조사 책임",
            "열량 변경 가스 사고 책임",
            "연소기 제조사 책임"
        ]
        
        all_docs = []
        for search_query in search_queries:
            logger.info(f"Trying search with: '{search_query}'")
            docs = self._vectorstore.similarity_search(search_query, k=20)
            all_docs.extend(docs)
            logger.info(f"  Found {len(docs)} documents")
        
        # 중복 제거
        seen_docs = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = (doc.metadata.get('source', ''), doc.page_content[:100])
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_docs.append(doc)
        
        all_docs = unique_docs[:50]  # 최대 50개로 제한
        logger.info(f"Vector search returned {len(all_docs)} documents")
        
        # 4. 지역별 필터링
        region_docs = []
        for doc in all_docs:
            source = doc.metadata.get('source', '')
            doc_region = extract_region_from_filename(source)
            logger.info(f"Checking document: {source}, extracted region: {doc_region}, target: {target_region}")
            
            # 더 관대한 지역 매칭
            region_match = False
            if doc_region:
                if target_region in doc_region or doc_region in target_region:
                    region_match = True
                # 추가: "강원" vs "강원도" 같은 케이스를 위한 더 관대한 매칭
                elif target_region == "강원" and "강원" in doc_region:
                    region_match = True
                elif doc_region == "강원" and "강원" in target_region:
                    region_match = True
                    
            if region_match:
                region_docs.append(doc)
                logger.info(f"  ✓ Found in region: {source}")
                # 문서 내용도 일부 로깅
                content_preview = doc.page_content[:150].replace('\n', ' ')
                logger.info(f"    Content preview: {content_preview}...")
            else:
                logger.info(f"  ✗ Not in target region: {source} (region: {doc_region})")
        
        if not region_docs:
            logger.warning(f"No documents found for region: {target_region}")
            logger.warning(f"Available documents: {[doc.metadata.get('source', '') for doc in all_docs[:10]]}")
            return []
        
        logger.info(f"Found {len(region_docs)} documents in {target_region}")
        
        # 5. 관련성 점수로 정렬 (채팅 히스토리 컨텍스트 고려)
        scored_docs = []
        for doc in region_docs:
            score = self._calculate_relevance_score(doc.page_content, core_query, chat_history)
            scored_docs.append((doc, score))
            logger.info(f"  Relevance score for {doc.metadata.get('source', 'unknown')}: {score}")
        
        # 점수순으로 정렬하고 상위 결과 반환
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:5]]
    
    def _extract_core_query(self, query: str, region: str) -> str:
        """질문에서 지역명과 회사명을 제거하고 핵심 키워드만 추출합니다."""
        # 지역명과 회사명 패턴들
        region_patterns = [
            f"{region}도시가스", f"{region}도 도시가스", f"{region}특별시 도시가스",
            f"{region}시 도시가스", f"{region}군 도시가스", f"{region}구 도시가스"
        ]
        
        # 회사명 패턴들
        company_patterns = [
            "도시가스", "가스회사", "가스공사", "가스공급", "가스공급사"
        ]
        
        core_query = query
        
        # 지역명 패턴 제거
        for pattern in region_patterns:
            core_query = core_query.replace(pattern, "")
        
        # 회사명 패턴 제거
        for pattern in company_patterns:
            core_query = core_query.replace(pattern, "")
        
        # 추가 정리
        core_query = core_query.replace("에서", "").replace("의", "").replace("에", "")
        core_query = core_query.strip()
        
        # 빈 문자열이 되면 원본 질문 반환
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

                # 간단한 동의어 가중치
                synonyms_map = {
                    "승인": ["허가", "통지", "서면 통지", "서면으로 통지"],
                    "통지": ["서면 통지", "서면으로 통지"],
                    "신청": ["접수"],
                    "책임": ["의무", "부담"],
                    "사고": ["안전사고", "가스안전사고", "위해"],
                    "열량": ["열량변경", "기준열량", "최고열량"],
                    "가스": ["도시가스", "가스공급"]
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
            "신청일로부터", "일 이내", "서면으로 통지", "서면 통지", "며칠 이내",
            "열량변경작업", "가스사고", "연소기 제조사", "제조사의 책임", "책임이 있습니다"
        ]
        for phrase in phrase_boosts:
            if phrase in content_lower:
                if phrase in ["열량변경작업", "가스사고", "연소기 제조사"]:
                    score += 3.0  # 높은 가중치
                else:
                    score += 1.5

        # 특별히 중요한 키워드들에 대한 추가 가중치
        high_priority_keywords = [
            "열량변경", "연소기", "제조사"
        ]
        for keyword in high_priority_keywords:
            if keyword in content_lower:
                score += 2.0  # 매우 높은 가중치
        
        # 채팅 히스토리 컨텍스트 고려
        if chat_history and len(chat_history) > 0:
            try:
                # 이전 대화에서 언급된 키워드들
                context_keywords = []
                for msg in chat_history[-3:]:  # 최근 3개 메시지만 고려
                    if isinstance(msg, dict):
                        content_msg = msg.get('content', '')
                        # 주요 도메인 키워드 추출
                        domain_keywords = ['계약', '연체료', '안전관리', '공사비', '분담금', '요금', '신청', '승인', '통지', '책임', '준수']
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
        # 도시가스 관련 주요 키워드들
        gas_keywords = [
            "도시가스", "가스공급", "가스요금", "가스사용", "가스계량", 
            "가스설비", "가스안전", "가스공급계약", "공급규정", "서울특별시",
            "기준열량", "최고열량", "계량기", "공급관", "안전관리", "안전", "책임", "관리",
            "열량변경", "가스사고", "사고", "열량", "변경작업", "연소기", "제조사"
        ]
        
        # 질문에서 키워드 매칭
        found_keywords = []
        for keyword in gas_keywords:
            if keyword in query:
                found_keywords.append(keyword)
        
        # 지역명 추가
        regions = ["강원", "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                  "경기", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
        for region in regions:
            if region in query:
                found_keywords.append(region)
        
        # 키워드가 없으면 질문을 단어로 분할 (더 관대하게)
        if not found_keywords:
            words = query.split()
            # 2글자 이상의 단어만 키워드로 사용
            found_keywords = [word for word in words if len(word) >= 2][:5]
        
        # 중복 제거
        found_keywords = list(set(found_keywords))
        
        logger.info(f"Extracted keywords from query '{query}': {found_keywords}")
        return found_keywords
    
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
        
        # 검색된 문서의 메타데이터 로깅 추가
        logger.info(f"Found {len(docs)} documents:")
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            region = extract_region_from_filename(source)
            logger.info(f"  Doc {i+1}: source={source}, region={region}")
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            formatted_doc = f"<document id='{i}'>\n"
            formatted_doc += f"<content>{doc.page_content}</content>\n"
            if 'source' in doc.metadata:
                formatted_doc += f"<source>{doc.metadata['source']}</source>\n"
            if 'version' in doc.metadata and doc.metadata['version']:
                formatted_doc += f"<version>{doc.metadata['version']}</version>\n"
            if 'region' in doc.metadata and doc.metadata['region']:
                formatted_doc += f"<region>{doc.metadata['region']}</region>\n"
            if 'organization' in doc.metadata and doc.metadata['organization']:
                formatted_doc += f"<organization>{doc.metadata['organization']}</organization>\n"
            if 'page' in doc.metadata:
                formatted_doc += f"<page>{doc.metadata['page']}</page>\n"
            if 'chunk_id' in doc.metadata:
                formatted_doc += f"<chunk_id>{doc.metadata['chunk_id']}</chunk_id>\n"
            if 'title' in doc.metadata:
                formatted_doc += f"<title>{doc.metadata['title']}</title>\n"
            if 'date' in doc.metadata:
                formatted_doc += f"<date>{doc.metadata['date']}</date>\n"
            formatted_doc += "</document>"
            formatted_docs.append(formatted_doc)
        
        logger.info(f"Formatted {len(formatted_docs)} documents")
        return "\n".join(formatted_docs) 