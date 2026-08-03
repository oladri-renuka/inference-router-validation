#!/usr/bin/env python3
"""Run benchmarks comparing routing strategies."""

import subprocess
import json
import time
import sys
import os
import asyncio
import multiprocessing

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from worker import run_worker


def start_workers(num_workers=3):
    """Start worker processes."""
    processes = []
    for i in range(num_workers):
        port = 8001 + i
        worker_id = f"worker-{i+1}"
        p = multiprocessing.Process(target=run_worker, args=(worker_id, port))
        p.start()
        processes.append(p)
        print(f"Started {worker_id} on port {port}")
    return processes


def _run_uvicorn():
    """Helper function to run uvicorn (needed for multiprocessing)."""
    import uvicorn
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from server import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


def start_load_balancer():
    """Start load balancer server."""
    # Run in subprocess
    p = multiprocessing.Process(target=_run_uvicorn)
    p.start()
    print("Started load balancer on port 8000")
    return p


def run_locust_benchmark(duration=60, users=50, hatch_rate=10):
    """Run Locust benchmark."""
    print(f"\nRunning benchmark: {duration}s, {users} concurrent users")
    print("=" * 60)

    cmd = [
        "locust",
        "-f",
        "benchmarks/locustfile.py",
        "--host=http://127.0.0.1:8000",
        f"--users={users}",
        f"--spawn-rate={hatch_rate}",
        f"--run-time={duration}s",
        "--headless",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def get_load_balancer_metrics():
    """Fetch metrics from load balancer."""
    import httpx

    try:
        with httpx.Client(timeout=5) as client:
            response = client.get("http://127.0.0.1:8000/metrics")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        print(f"Error fetching metrics: {e}")
    return None


def main():
    """Run complete benchmark suite."""
    print("Smart Load Balancer Benchmark Suite")
    print("=" * 60)

    # Start services
    print("\nStarting infrastructure...")
    worker_processes = start_workers(num_workers=3)
    lb_process = start_load_balancer()

    # Wait for services to be ready
    time.sleep(5)

    results = {}

    try:
        # Run benchmark
        print("\nExecuting benchmark with mixed traffic...")
        print("Traffic mix: 60% short (100 tokens), 30% medium (400 tokens), 10% long (800+ tokens)")

        success = run_locust_benchmark(duration=120, users=50, hatch_rate=10)

        # Get final metrics
        time.sleep(2)
        metrics = get_load_balancer_metrics()

        if metrics:
            results["metrics"] = metrics
            print("\nFinal Load Balancer Metrics:")
            print(f"  Total Requests: {metrics['total_requests']}")
            print(f"  P50 Latency: {metrics['p50_latency']:.2f}ms")
            print(f"  P95 Latency: {metrics['p95_latency']:.2f}ms")
            print(f"  P99 Latency: {metrics['p99_latency']:.2f}ms")
            print(f"  Avg Latency: {metrics['avg_latency']:.2f}ms")
            print(f"  Throughput: {metrics['throughput_rps']:.2f} req/s")

            # Worker breakdown
            print("\nWorker Health:")
            for worker_id, health in metrics["worker_health"].items():
                print(
                    f"  {worker_id}: "
                    f"healthy={health['healthy']}, "
                    f"requests={health.get('current_load', 0)}, "
                    f"avg_latency={health['avg_latency']:.2f}ms"
                )

    finally:
        # Cleanup
        print("\nShutting down services...")
        lb_process.terminate()
        for p in worker_processes:
            p.terminate()

        # Wait for processes
        lb_process.join(timeout=5)
        for p in worker_processes:
            p.join(timeout=5)

        print("Benchmark complete.")


if __name__ == "__main__":
    main()
