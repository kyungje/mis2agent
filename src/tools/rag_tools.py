from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class LegalRAGTool(BaseTool):
    name: str = "legal_rag_search"
    description: str = "법조문이나 법령에 대한 질문에 답하기 위해 관련 문서를 검색합니다."
    
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def _run(self, query: str) -> str:
        """질문에 관련된 법조문/법령을 검색합니다."""
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