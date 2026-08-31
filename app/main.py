from fastapi import FastAPI

app = FastAPI(
    title="Atlas — Enterprise Intelligence Platform",
    description="Production-oriented enterprise document intelligence platform.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}