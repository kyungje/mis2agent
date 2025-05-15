from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pydantic import Field
import os

class LegalRAGTool(BaseTool):
    name: str = "gas_supply_regulation_search"
    description: str = "서울특별시 도시가스회사 공급규정에 대한 질문에 답하기 위해 관련 문서를 검색합니다."
    embeddings: Optional[OpenAIEmbeddings] = Field(default=None)
    vectorstore: Optional[FAISS] = Field(default=None)
    retriever: Optional[Any] = Field(default=None)
    
    def __init__(self, index_path: str):
        super().__init__()
        self.embeddings = OpenAIEmbeddings()
        
        # 인덱스 파일 경로 확인
        index_dir = os.path.dirname(index_path)
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"FAISS 인덱스 디렉토리를 찾을 수 없습니다: {index_dir}")
            
        try:
            # FAISS는 index.faiss 파일을 찾으므로, 디렉토리 경로만 전달
            self.vectorstore = FAISS.load_local(
                index_dir, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        except Exception as e:
            raise Exception(f"FAISS 인덱스 로드 중 오류 발생: {str(e)}")
    
    def _run(self, query: str) -> str:
        """서울특별시 도시가스회사 공급규정 관련 질문에 대한 문서를 검색합니다."""
        # 검색어에 "서울특별시 도시가스회사 공급규정" 관련 키워드가 있는지 확인
        keywords = ["서울특별시", "도시가스", "공급규정", "가스공급", "가스요금"]
        if not any(keyword in query for keyword in keywords):
            return "죄송합니다. 이 도구는 서울특별시 도시가스회사 공급규정에 대한 질문만 처리할 수 있습니다."
            
        docs = self.retriever.get_relevant_documents(query)
        return self._format_docs(docs)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """검색된 문서들을 포맷팅합니다."""
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
        return "\n".join(formatted_docs) 