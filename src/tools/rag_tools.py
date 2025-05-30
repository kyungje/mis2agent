from typing import List, Dict, Any, Optional, ClassVar, Type, Union
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
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
            # HuggingFace 임베딩 모델 초기화
            logger.info("Initializing HuggingFaceEmbeddings model")
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=os.path.join(os.path.dirname(__file__), "../.cache")
            )
            logger.info("HuggingFaceEmbeddings initialized successfully")
            
            # 인덱스 파일 경로 확인
            logger.info(f"Checking if index_path exists: {os.path.exists(index_path)}")
            if not os.path.exists(index_path):
                logger.error(f"Index path does not exist: {index_path}")
                raise FileNotFoundError(f"FAISS 인덱스 디렉토리를 찾을 수 없습니다: {index_path}")
            
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
            logger.error(f"Error initializing LegalRAGTool: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"FAISS 인덱스 로드 중 오류 발생: {str(e)}")
    
    def _run(self, query: str) -> str:
        """서울특별시 도시가스회사 공급규정 관련 질문에 대한 문서를 검색합니다."""
        logger.info(f"Running search query: {query}")
        
        
        try:
            if not self._retriever:
                logger.error("Retriever is not initialized")
                return "죄송합니다. RAG 검색 시스템이 초기화되지 않았습니다."
                
            logger.info("Retrieving relevant documents")
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
    
    def _format_docs(self, docs: List[Document]) -> str:
        """검색된 문서들을 포맷팅합니다."""
        logger.info("Formatting documents")
        
        if not docs:
            logger.warning("No documents to format")
            return "관련 문서를 찾을 수 없습니다."
        
        formatted_docs = []
        for doc in docs:
            formatted_doc = f"<document>\n"
            formatted_doc += f"<content>{doc.page_content}</content>\n"
            if 'source' in doc.metadata:
                formatted_doc += f"<source>{doc.metadata['source']}</source>\n"
            if 'page' in doc.metadata:
                formatted_doc += f"<page>{doc.metadata['page']}</page>\n"
            formatted_doc += "</document>"
            formatted_docs.append(formatted_doc)
        
        logger.info(f"Formatted {len(formatted_docs)} documents")
        return "\n".join(formatted_docs) 