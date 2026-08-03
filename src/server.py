import asyncio
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

from predictor import OutputLengthPredictor
from load_balancer import SmartLoadBalancer, InferenceRequest, RoutingStrategy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class InferenceRequestInput(BaseModel):
    prompt: str


class InferenceResponse(BaseModel):
    request_id: str
    output: str
    worker_id: str
    predicted_length: float
    actual_length: int
    latency_ms: float


# Global state
load_balancer: SmartLoadBalancer = None
predictor: OutputLengthPredictor = None
inference_tasks = {}


@app.on_event("startup")
async def startup():
    global load_balancer, predictor

    # Initialize predictor
    predictor = OutputLengthPredictor()

    # Train on synthetic data
    training_prompts = [
        "What is machine learning?",
        "Explain the concept of neural networks in detail",
        "Write a Python function to sort a list",
        "Describe the history of artificial intelligence",
        "How does quantum computing work?",
    ] * 20

    training_lengths = [
        100,
        500,
        150,
        800,
        400,
    ] * 20

    predictor.train(training_prompts, training_lengths)

    # Initialize load balancer with worker addresses
    worker_addresses = [
        "127.0.0.1:8001",
        "127.0.0.1:8002",
        "127.0.0.1:8003",
    ]

    load_balancer = SmartLoadBalancer(worker_addresses, predictor)

    # Start health checks for each worker
    for worker_id in worker_addresses:
        asyncio.create_task(load_balancer.health_check_worker(worker_id))

    logger.info(f"Load balancer initialized with workers: {worker_addresses}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "component": "load-balancer",
        "timestamp": time.time(),
    }


@app.post("/infer")
async def infer(request: InferenceRequestInput) -> InferenceResponse:
    """Submit inference request and route to worker."""
    if not load_balancer or not predictor:
        raise HTTPException(status_code=500, detail="Server not initialized")

    # Predict output length
    predicted_length = predictor.predict(request.prompt)

    # Create request object
    inference_req = InferenceRequest(
        request_id=str(uuid.uuid4()),
        prompt=request.prompt,
        predicted_output_length=predicted_length,
    )

    try:
        # Route request
        worker_id = load_balancer.route_request(
            inference_req, strategy=RoutingStrategy.PREDICTED_COST
        )

        # Process request
        start_time = time.time()
        result = await load_balancer.process_request(inference_req, worker_id)
        latency_ms = (time.time() - start_time) * 1000

        return InferenceResponse(
            request_id=inference_req.request_id,
            output=result["output"],
            worker_id=worker_id,
            predicted_length=predicted_length,
            actual_length=result["tokens_generated"],
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Error processing inference: {e}")
        raise HTTPException(status_code=503, detail="No healthy workers available")


@app.get("/metrics")
async def get_metrics():
    """Get load balancer metrics."""
    if not load_balancer:
        raise HTTPException(status_code=500, detail="Server not initialized")

    return load_balancer.get_metrics()


@app.get("/workers")
async def get_workers():
    """Get worker status."""
    if not load_balancer:
        raise HTTPException(status_code=500, detail="Server not initialized")

    return {
        "workers": [
            {
                "id": worker_id,
                "health": load_balancer.health[worker_id].healthy,
                "current_load": load_balancer.health[worker_id].current_requests,
                "total_requests": load_balancer.health[worker_id].total_requests,
            }
            for worker_id in load_balancer.workers
        ]
    }
