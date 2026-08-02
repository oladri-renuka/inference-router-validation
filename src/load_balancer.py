import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import httpx


class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PREDICTED_COST = "predicted_cost"


@dataclass
class WorkerHealth:
    """Track health metrics for a worker."""

    worker_id: str
    healthy: bool = True
    current_requests: int = 0
    total_requests: int = 0
    total_latency: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    error_count: int = 0

    @property
    def avg_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests

    @property
    def load_score(self) -> float:
        """Estimate worker load (0-1 scale)."""
        # Combine current requests and recent latency
        return min(
            (self.current_requests / 10.0) * 0.7 + (self.avg_latency / 5000.0) * 0.3,
            1.0,
        )


@dataclass
class InferenceRequest:
    """Incoming inference request."""

    request_id: str
    prompt: str
    predicted_output_length: float
    submitted_at: float = field(default_factory=time.time)
    routed_to_worker: Optional[str] = None
    completed_at: Optional[float] = None

    @property
    def latency(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return self.completed_at - self.submitted_at


class SmartLoadBalancer:
    """Intelligent load balancer that routes based on predicted output length."""

    def __init__(self, workers: list[str], predictor=None):
        self.workers = workers
        self.predictor = predictor
        self.health = {worker_id: WorkerHealth(worker_id) for worker_id in workers}
        self.request_queue = asyncio.Queue()
        self.active_requests = {}
        self.completed_requests = []
        self.round_robin_index = 0

    async def health_check_worker(self, worker_id: str, timeout: float = 2.0):
        """Periodically check worker health."""
        while True:
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(f"http://{worker_id}/health")
                    if response.status_code == 200:
                        self.health[worker_id].healthy = True
                        self.health[worker_id].last_heartbeat = time.time()
                    else:
                        self.health[worker_id].healthy = False
            except Exception as e:
                self.health[worker_id].healthy = False
                self.health[worker_id].error_count += 1

            await asyncio.sleep(5)

    def get_healthy_workers(self) -> list[str]:
        """Get list of healthy workers."""
        return [
            worker_id
            for worker_id in self.workers
            if self.health[worker_id].healthy
        ]

    def route_request(
        self, request: InferenceRequest, strategy: RoutingStrategy = RoutingStrategy.PREDICTED_COST
    ) -> str:
        """Route request to a worker based on predicted cost."""
        healthy_workers = self.get_healthy_workers()

        if not healthy_workers:
            raise RuntimeError("No healthy workers available")

        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(healthy_workers)

        elif strategy == RoutingStrategy.LEAST_LOADED:
            return self._route_least_loaded(healthy_workers)

        elif strategy == RoutingStrategy.PREDICTED_COST:
            return self._route_predicted_cost(
                request, healthy_workers
            )

        return healthy_workers[0]

    def _route_round_robin(self, workers: list[str]) -> str:
        """Simple round-robin routing."""
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker

    def _route_least_loaded(self, workers: list[str]) -> str:
        """Route to least loaded worker."""
        least_loaded = min(
            workers, key=lambda w: self.health[w].load_score
        )
        return least_loaded

    def _route_predicted_cost(
        self, request: InferenceRequest, workers: list[str]
    ) -> str:
        """Route based on predicted output cost."""
        # High-cost requests go to least-loaded workers
        if request.predicted_output_length > 500:
            return self._route_least_loaded(workers)
        # Low-cost requests use round-robin
        else:
            return self._route_round_robin(workers)

    async def submit_request(self, request: InferenceRequest):
        """Submit request to queue."""
        await self.request_queue.put(request)

    async def process_request(
        self, request: InferenceRequest, worker_url: str, timeout: float = 30.0
    ):
        """Send request to worker and track metrics."""
        request.routed_to_worker = worker_url
        worker_health = self.health[worker_url]

        try:
            worker_health.current_requests += 1
            worker_health.total_requests += 1

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"http://{worker_url}/infer",
                    json={"prompt": request.prompt},
                )

            request.completed_at = time.time()
            latency = request.latency

            if latency:
                worker_health.total_latency += latency

            self.completed_requests.append(request)
            return response.json()

        except Exception as e:
            worker_health.error_count += 1
            raise

        finally:
            worker_health.current_requests -= 1

    def get_metrics(self) -> dict:
        """Get current load balancer metrics."""
        completed = self.completed_requests
        latencies = [r.latency for r in completed if r.latency]

        throughput_rps = 0
        if len(completed) > 1:
            duration = completed[-1].completed_at - completed[0].submitted_at
            throughput_rps = len(completed) / max(duration, 1)

        return {
            "total_requests": len(completed),
            "p50_latency": np.percentile(latencies, 50) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency": np.percentile(latencies, 99) if latencies else 0,
            "avg_latency": np.mean(latencies) if latencies else 0,
            "throughput_rps": throughput_rps,
            "worker_health": {
                worker_id: {
                    "healthy": h.healthy,
                    "current_load": h.current_requests,
                    "avg_latency": h.avg_latency,
                    "error_count": h.error_count,
                }
                for worker_id, h in self.health.items()
            },
        }


import numpy as np
