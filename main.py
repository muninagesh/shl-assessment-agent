import json
import faiss
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from sentence_transformers import SentenceTransformer


with open("shl_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")


def build_document(item):
    return f"""
    Name: {item.get('name', '')}
    Description: {item.get('description', '')}
    Job Levels: {', '.join(item.get('job_levels', []))}
    Categories: {', '.join(item.get('keys', []))}
    Duration: {item.get('duration', '')}
    """


documents = [build_document(item) for item in catalog]

embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


def retrieve_assessments(query, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        item = catalog[idx]

        results.append(
            {
                "name": item.get("name", ""),
                "url": item.get("link", ""),
                "test_type": item.get("keys", ["General"])[0],
            }
        )

    return results


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    messages = [m.dict() for m in request.messages]

    full_context = "\n".join(
        [f"{m['role']}: {m['content']}" for m in messages]
    )

    recommendations = retrieve_assessments(full_context, top_k=5)

    return {
        "reply": "Here are recommended SHL assessments based on your hiring requirements.",
        "recommendations": recommendations,
        "end_of_conversation": True,
    }
