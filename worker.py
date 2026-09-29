from celery import Celery
from pypdf import PdfReader
from rag import embed, chunk_text
from db import insert_chunk

celery = Celery(
    "worker",
    broker="redis://localhost:6379/0"
)

@celery.task
def index_document(path, tenant):
    reader = PdfReader(path)

    text = "" #extract
    for page in reader.pages:
        text += page.extract_text() or ""

    #clean
    text = text.replace("\x00", "")
    text = " ".join(text.split())

    #chunk
    chunks = chunk_text(text)

    #embed and store
    for chunk in chunks:
        vector = embed(chunk)

        insert_chunk(
            tenant =tenant,
            source = path,
            title = path,
            content = chunk,
            embedding = vector
        )
    return len(chunks)