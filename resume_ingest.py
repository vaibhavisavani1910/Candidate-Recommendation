import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50

def load_and_split_resumes(resume_files: List[str]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    all_chunks = []
    
    for path in resume_files:
        loader = PyPDFLoader(path)
        pages = loader.load()
        full_text = " ".join([p.page_content for p in pages])
        doc = Document(
            page_content=full_text,
            metadata={
                "source": os.path.basename(path),
                "ID": os.path.splitext(os.path.basename(path))[0]
            }
        )
        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)
    
    return all_chunks
