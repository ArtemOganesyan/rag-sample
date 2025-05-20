from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(request: TextRequest):
    text = request.text
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}

if __name__ == "__main__":
    uvicorn.run("embedder_service:app", host="0.0.0.0", port=8002)

