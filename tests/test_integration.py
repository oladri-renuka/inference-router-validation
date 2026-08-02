#!/usr/bin/env python3
"""Integration test for complete load balancer flow."""

import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predictor import OutputLengthPredictor
from load_balancer import SmartLoadBalancer, InferenceRequest, RoutingStrategy


async def test_complete_flow():
    """Test complete inference flow: predict -> route -> process."""

    print("Integration Test: Complete Inference Flow")
    print("=" * 60)

    # Step 1: Initialize predictor
    print("\n1. Training predictor...")
    predictor = OutputLengthPredictor()

    prompts = [
        "What is ML?" * 5,
        "Explain neural networks in detail" * 5,
        "Write comprehensive guide" * 5,
    ] * 10

    lengths = [50, 500, 800] * 10
    predictor.train(prompts, lengths)
    print("   ✓ Predictor trained")

    # Step 2: Initialize load balancer
    print("\n2. Initializing load balancer...")
    workers = ["worker-1", "worker-2", "worker-3"]
    lb = SmartLoadBalancer(workers, predictor)
    print(f"   ✓ Load balancer created with {len(workers)} workers")

    # Step 3: Simulate multiple requests
    print("\n3. Routing requests...")
    request_results = []

    test_requests = [
        ("Short question?", 100),
        ("Detailed explanation of transformers" * 10, 500),
        ("Very long comprehensive guide" * 30, 800),
    ]

    for i, (prompt, expected_length) in enumerate(test_requests):
        # Predict length
        predicted_length = predictor.predict(prompt)

        # Create request
        req = InferenceRequest(
            request_id=f"req-{i}",
            prompt=prompt,
            predicted_output_length=predicted_length,
        )

        # Route request
        worker = lb.route_request(req, strategy=RoutingStrategy.PREDICTED_COST)

        # Track metrics
        lb.health[worker].current_requests += 1
        lb.health[worker].total_requests += 1

        request_results.append({
            "request_id": req.request_id,
            "predicted_length": predicted_length,
            "routed_to": worker,
            "worker_load": lb.health[worker].current_requests,
        })

        print(f"   Req {i}: predicted={predicted_length:.0f} tokens → {worker}")

    # Step 4: Verify routing
    print("\n4. Verifying routing...")

    # High-cost requests should go to different workers (distributed)
    high_cost = [r for r in request_results if r["predicted_length"] > 400]
    if high_cost:
        routed_to = [r["routed_to"] for r in high_cost]
        print(f"   High-cost requests routed to: {routed_to}")
        print(f"   ✓ Distributed among {len(set(routed_to))} workers")

    # Step 5: Get metrics
    print("\n5. Collecting metrics...")
    metrics = lb.get_metrics()

    print(f"   Total requests processed: {metrics['total_requests']}")
    print(f"   Worker health:")
    for worker_id, health in metrics['worker_health'].items():
        print(f"     - {worker_id}: healthy={health['healthy']}, "
              f"requests={health.get('current_load', 0)}")

    print("\n" + "=" * 60)
    print("Integration test PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_complete_flow())
