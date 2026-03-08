import os
from typing import Union, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("UYARI: HF_TOKEN bulunamadı! Model indirme/yükleme başarısız olabilir.")
else:
    login(token=hf_token)

app = FastAPI(title="Magibu Embedding API")

model_name = "magibu/embeddingmagibu-200m"
print(f"--- {model_name} yükleniyor... ---")
model = SentenceTransformer(model_name)
print("--- Model başarıyla yüklendi ve API hazır! ---")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = model_name

@app.get("/")
async def health_check():
    return {"status": "online", "model": model_name, "docs": "/docs"}

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    try:
        
        texts = request.input if isinstance(request.input, list) else [request.input]

        embeddings = model.encode(texts)
        
        data = []
        for i, vec in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": vec.tolist()
            })
            
        total_tokens = sum(len(t.split()) for t in texts) 

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))