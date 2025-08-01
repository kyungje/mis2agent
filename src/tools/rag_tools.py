from typing import List, Dict, Any, Optional, ClassVar, Type, Union
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr
import os
import logging
import traceback

logger = logging.getLogger(__name__)

class LegalRAGTool(BaseTool):
    name: ClassVar[str] = "gas_supply_regulation_search"
    description: ClassVar[str] = "서울특별시 도시가스회사 공급규정에서 관련 정보를 검색합니다."
    
    # 프라이빗 속성으로 정의 (Pydantic 모델에 포함되지 않음)
    _embeddings: Any = PrivateAttr(default=None)
    _vectorstore: Any = PrivateAttr(default=None)
    _retriever: BaseRetriever = PrivateAttr(default=None)
    
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
                    return
                    
            except Exception as e:
                logger.warning(f"Error reading directory contents: {e}")
                self._vectorstore = None
                self._retriever = None
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
            base_retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # 검색 결과 수를 3개로 증가
            )
            
            # MultiQueryRetriever로 래핑하여 다양한 쿼리 생성
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            self._retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm
            )
            logger.info("MultiQueryRetriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LegalRAGTool: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("LegalRAGTool will be initialized without vectorstore")
            self._vectorstore = None
            self._retriever = None
    
    def _run(self, query: str, search_strategy: str = "default") -> str:
        """서울특별시 도시가스회사 공급규정 관련 질문에 대한 문서를 검색합니다."""
        logger.info(f"Running search query: {query} with strategy: {search_strategy}")
        
        try:
            if not self._retriever:
                logger.warning("Retriever is not initialized - no legal documents index available")
                return "법률 문서 인덱스가 없습니다. 일반 대화로 응답합니다."
            
            # 검색 전략에 따른 검색 수행
            if search_strategy == "expanded":
                # 확장된 검색: 더 많은 결과와 더 넓은 범위
                docs = self._search_with_expanded_strategy(query)
            elif search_strategy == "keyword":
                # 키워드 기반 검색
                docs = self._search_with_keyword_strategy(query)
            elif search_strategy == "comparison":
                # 출처별 비교를 위한 검색
                docs = self._search_with_comparison_strategy(query)
            else:
                # 기본 검색 - MultiQueryRetriever로 쿼리 다양화만 사용
                docs = self._retriever.get_relevant_documents(query)
            
            logger.info(f"Found {len(docs)} relevant documents")
            
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
        
        # 더 많은 결과를 가져와서 출처별로 그룹화
        comparison_retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}  # 더 많은 결과
        )
        
        docs = comparison_retriever.get_relevant_documents(query)
        
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
        # 도시가스 관련 주요 키워드들
        gas_keywords = [
            "도시가스", "가스공급", "가스요금", "가스사용", "가스계량", 
            "가스설비", "가스안전", "가스공급계약", "공급규정", "서울특별시",
            "기준열량", "최고열량", "계량기", "공급관", "안전관리"
        ]
        
        # 질문에서 키워드 매칭
        found_keywords = []
        for keyword in gas_keywords:
            if keyword in query:
                found_keywords.append(keyword)
        
        # 키워드가 없으면 질문을 단어로 분할
        if not found_keywords:
            words = query.split()
            # 2글자 이상의 단어만 키워드로 사용
            found_keywords = [word for word in words if len(word) >= 2][:3]
        
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
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            formatted_doc = f"<document id='{i}'>\n"
            formatted_doc += f"<content>{doc.page_content}</content>\n"
            if 'source' in doc.metadata:
                formatted_doc += f"<source>{doc.metadata['source']}</source>\n"
            if 'page' in doc.metadata:
                formatted_doc += f"<page>{doc.metadata['page']}</page>\n"
            if 'chunk_id' in doc.metadata:
                formatted_doc += f"<chunk_id>{doc.metadata['chunk_id']}</chunk_id>\n"
            if 'title' in doc.metadata:
                formatted_doc += f"<title>{doc.metadata['title']}</title>\n"
            if 'date' in doc.metadata:
                formatted_doc += f"<date>{doc.metadata['date']}</date>\n"
            if 'region' in doc.metadata:
                formatted_doc += f"<region>{doc.metadata['region']}</region>\n"
            formatted_doc += "</document>"
            formatted_docs.append(formatted_doc)
        
        logger.info(f"Formatted {len(formatted_docs)} documents")
        return "\n".join(formatted_docs) 