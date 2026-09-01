from fastapi import FastAPI

from app.api.documents import router as documents_router

app = FastAPI(
    title="Atlas — Enterprise Intelligence Platform",
    description="Production-oriented enterprise document intelligence platform.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return "healthy"


app.include_router(documents_router)