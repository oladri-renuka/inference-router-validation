import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from load_balancer import (
    SmartLoadBalancer,
    InferenceRequest,
    RoutingStrategy,
    WorkerHealth,
)


def test_worker_health():
    """Test worker health tracking."""
    health = WorkerHealth("worker-1")

    assert health.healthy
    assert health.current_requests == 0
    assert health.total_requests == 0
    assert health.avg_latency == 0.0
    assert health.load_score >= 0


def test_load_balancer_initialization():
    """Test load balancer initialization."""
    workers = ["worker-1", "worker-2", "worker-3"]
    lb = SmartLoadBalancer(workers)

    assert len(lb.workers) == 3
    assert len(lb.health) == 3
    assert all(lb.health[w].healthy for w in workers)


def test_get_healthy_workers():
    """Test getting healthy workers."""
    workers = ["worker-1", "worker-2", "worker-3"]
    lb = SmartLoadBalancer(workers)

    healthy = lb.get_healthy_workers()
    assert len(healthy) == 3

    # Mark one as unhealthy
    lb.health["worker-1"].healthy = False
    healthy = lb.get_healthy_workers()
    assert len(healthy) == 2
    assert "worker-1" not in healthy


def test_round_robin_routing():
    """Test round-robin routing."""
    workers = ["worker-1", "worker-2", "worker-3"]
    lb = SmartLoadBalancer(workers)

    req = InferenceRequest(
        request_id="test-1",
        prompt="What is ML?",
        predicted_output_length=100,
    )

    # Route multiple requests
    routes = []
    for _ in range(9):
        route = lb.route_request(req, strategy=RoutingStrategy.ROUND_ROBIN)
        routes.append(route)

    # Should distribute evenly
    assert routes.count("worker-1") == 3
    assert routes.count("worker-2") == 3
    assert routes.count("worker-3") == 3


def test_least_loaded_routing():
    """Test least-loaded routing."""
    workers = ["worker-1", "worker-2", "worker-3"]
    lb = SmartLoadBalancer(workers)

    # Set different loads
    lb.health["worker-1"].current_requests = 5
    lb.health["worker-2"].current_requests = 1  # Least loaded
    lb.health["worker-3"].current_requests = 3

    req = InferenceRequest(
        request_id="test-1",
        prompt="What is ML?",
        predicted_output_length=100,
    )

    route = lb.route_request(req, strategy=RoutingStrategy.LEAST_LOADED)
    assert route == "worker-2"


def test_predicted_cost_routing():
    """Test predicted cost routing."""
    workers = ["worker-1", "worker-2", "worker-3"]
    lb = SmartLoadBalancer(workers)

    # Set different loads
    lb.health["worker-1"].current_requests = 2  # Least loaded
    lb.health["worker-2"].current_requests = 5
    lb.health["worker-3"].current_requests = 8

    # High-cost request should go to least loaded
    high_cost_req = InferenceRequest(
        request_id="test-1",
        prompt="Long prompt" * 50,
        predicted_output_length=600,
    )

    route = lb.route_request(high_cost_req, strategy=RoutingStrategy.PREDICTED_COST)
    assert route == "worker-1"

    # Low-cost request uses round-robin
    low_cost_req = InferenceRequest(
        request_id="test-2",
        prompt="Short?",
        predicted_output_length=50,
    )

    lb.round_robin_index = 0
    route = lb.route_request(low_cost_req, strategy=RoutingStrategy.PREDICTED_COST)
    assert route in workers


def test_metrics():
    """Test metrics collection."""
    workers = ["worker-1", "worker-2"]
    lb = SmartLoadBalancer(workers)

    metrics = lb.get_metrics()

    assert "total_requests" in metrics
    assert "p50_latency" in metrics
    assert "p95_latency" in metrics
    assert "p99_latency" in metrics
    assert "worker_health" in metrics


if __name__ == "__main__":
    test_worker_health()
    print("✓ Worker health test passed")

    test_load_balancer_initialization()
    print("✓ Load balancer initialization test passed")

    test_get_healthy_workers()
    print("✓ Healthy workers test passed")

    test_round_robin_routing()
    print("✓ Round-robin routing test passed")

    test_least_loaded_routing()
    print("✓ Least-loaded routing test passed")

    test_predicted_cost_routing()
    print("✓ Predicted cost routing test passed")

    test_metrics()
    print("✓ Metrics test passed")

    print("\nAll load balancer tests passed!")
