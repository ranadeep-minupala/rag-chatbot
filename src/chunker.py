from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk(documents):
    all_chunks = []
    for doc in documents:
        # 1. Split this document's text into a list of string pieces.
        pieces = splitter.split_text(doc["text"])     

        # 2. Wrap each piece with its source filename.
        for piece in pieces:
            all_chunks.append({"text": piece, "source": doc["source"]})               

    return all_chunks