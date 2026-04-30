from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf(file_path, original_name):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = original_name
        
        doc.metadata["page"] = doc.metadata.get("page", "?")
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "- ", "• ", ". "]
    )

    return splitter.split_documents(docs)