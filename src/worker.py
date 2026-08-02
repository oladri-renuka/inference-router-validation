import asyncio
import random
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()


class InferenceInput(BaseModel):
    prompt: str


class InferenceOutput(BaseModel):
    output: str
    tokens_generated: int
    processing_time_ms: float


# Worker state
WORKER_ID = "worker-default"
PROCESSING_TIME_PER_TOKEN = 10  # milliseconds


def set_worker_id(worker_id: str):
    global WORKER_ID
    WORKER_ID = worker_id


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "worker_id": WORKER_ID,
        "timestamp": time.time(),
    }


@app.post("/infer")
async def infer(request: InferenceInput) -> InferenceOutput:
    """Simulate inference request."""
    prompt = request.prompt

    # Simulate output length based on prompt
    # Add randomness to simulate variance
    base_length = max(50, len(prompt) // 5)
    output_tokens = int(base_length * random.uniform(0.8, 1.2))

    # Simulate processing time
    processing_time_ms = output_tokens * PROCESSING_TIME_PER_TOKEN + random.gauss(
        0, 50
    )
    processing_time_ms = max(100, processing_time_ms)

    # Actually sleep to simulate processing
    await asyncio.sleep(processing_time_ms / 1000.0)

    output_text = f"Generated response based on: {prompt[:50]}..." * (
        output_tokens // 20 + 1
    )

    return InferenceOutput(
        output=output_text[:output_tokens],
        tokens_generated=output_tokens,
        processing_time_ms=processing_time_ms,
    )


def run_worker(worker_id: str, port: int):
    """Run a worker server."""
    import uvicorn

    set_worker_id(worker_id)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
