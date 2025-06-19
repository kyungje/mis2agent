from typing import List, Dict, Any, Optional, ClassVar, Type, Union
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
import os
import logging
import traceback

logger = logging.getLogger(__name__)

class OthersRAGTool(BaseTool):
    name: ClassVar[str] = "others_document_search"
    description: ClassVar[str] = "기타 문서들에서 관련 정보를 검색합니다."
    
    # 프라이빗 속성으로 정의 (Pydantic 모델에 포함되지 않음)
    _embeddings: Any = PrivateAttr(default=None)
    _vectorstore: Any = PrivateAttr(default=None)
    _retriever: BaseRetriever = PrivateAttr(default=None)
    
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
                logger.error(f"Index path does not exist: {index_path}")
                raise FileNotFoundError(f"기타 문서 FAISS 인덱스 디렉토리를 찾을 수 없습니다: {index_path}")
            
            # 디렉토리 내용 확인
            logger.info(f"Contents of index_path: {os.listdir(index_path)}")
            
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
            self._retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # 검색 결과 수를 3개로 제한
            )
            logger.info("Retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OthersRAGTool: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"기타 문서 FAISS 인덱스 로드 중 오류 발생: {str(e)}")
    
    def _run(self, query: str, search_strategy: str = "default") -> str:
        """기타 문서들에서 관련 정보를 검색합니다."""
        logger.info(f"Running search query: {query} with strategy: {search_strategy}")
        
        try:
            if not self._retriever:
                logger.error("Retriever is not initialized")
                return "죄송합니다. 기타 문서 RAG 검색 시스템이 초기화되지 않았습니다."
            
            # 검색 전략에 따른 검색 수행
            if search_strategy == "expanded":
                # 확장된 검색: 더 많은 결과와 더 넓은 범위
                docs = self._search_with_expanded_strategy(query)
            elif search_strategy == "keyword":
                # 키워드 기반 검색
                docs = self._search_with_keyword_strategy(query)
            else:
                # 기본 검색
                docs = self._retriever.get_relevant_documents(query)
            
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
            
            formatted_doc += "</document>"
            formatted_docs.append(formatted_doc)
        
        logger.info(f"Formatted {len(formatted_docs)} documents")
        return "\n".join(formatted_docs) 