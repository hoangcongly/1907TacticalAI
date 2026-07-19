from fastapi import FastAPI
app = FastAPI(title="Aegis Sentiment Service v11.8")
@app.get("/health")
def health(): return {"status": "ok"}
