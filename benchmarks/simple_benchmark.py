#!/usr/bin/env python3
"""Simple benchmark script that runs load tests directly."""

import subprocess
import time
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics


async def run_inference_load(duration=60, num_workers=20):
    """Run inference load test."""

    prompts = [
        "What is machine learning?",
        "Explain neural networks" * 5,
        "Write comprehensive guide" * 20,
    ]

    latencies = []
    request_count = 0
    errors = 0

    start_time = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() - start_time < duration:
            prompt = prompts[request_count % len(prompts)]

            try:
                req_start = time.time()
                response = await client.post(
                    "http://127.0.0.1:8000/infer",
                    json={"prompt": prompt}
                )
                latency = (time.time() - req_start) * 1000

                if response.status_code == 200:
                    latencies.append(latency)
                else:
                    errors += 1

            except Exception as e:
                errors += 1

            request_count += 1

    return latencies, request_count, errors


async def main():
    """Run complete benchmark."""
    print("\n" + "="*70)
    print("SMART LOAD BALANCER — SIMPLIFIED BENCHMARK")
    print("="*70)

    # Start workers
    print("\n[1/3] Starting workers...")
    worker_procs = []
    for i in range(3):
        port = 8001 + i
        cmd = f"import sys; sys.path.insert(0, 'src'); from worker import run_worker; run_worker('worker-{i+1}', {port})"
        proc = subprocess.Popen(
            ["python", "-c", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/Users/renukaoladri/Claude/Projects/knowledge_agent/smart-load-balancer"
        )
        worker_procs.append(proc)
    print("      ✓ Started 3 workers (ports 8001-8003)")
    time.sleep(2)

    # Start load balancer
    print("\n[2/3] Starting load balancer...")
    cmd = "import sys; sys.path.insert(0, 'src'); from server import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000, log_level='error')"
    lb_proc = subprocess.Popen(
        ["python", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd="/Users/renukaoladri/Claude/Projects/knowledge_agent/smart-load-balancer"
    )
    print("      ✓ Started load balancer (port 8000)")
    time.sleep(5)

    # Verify connectivity
    max_retries = 10
    response = None
    for i in range(max_retries):
        try:
            response = httpx.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                break
            time.sleep(0.5)
        except Exception as e:
            time.sleep(0.5)
            if i == max_retries - 1:
                print(f"      ✗ Cannot connect to load balancer: {e}")
                return

    if not response or response.status_code != 200:
        print("      ✗ Load balancer not responding")
        return

    print("      ✓ Load balancer responding")

    # Run benchmark
    print("\n[3/3] Running benchmark (60 seconds)...")
    print("      Sending mixed traffic pattern...")

    try:
        latencies, total_requests, errors = await run_inference_load(duration=60, num_workers=20)

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        if latencies:
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

            print(f"\nTotal Requests:     {total_requests}")
            print(f"Successful:         {len(latencies)}")
            print(f"Errors:             {errors}")
            print(f"\nLatency Metrics:")
            print(f"  P50:              {p50:.2f}ms")
            print(f"  P95:              {p95:.2f}ms")
            print(f"  P99:              {p99:.2f}ms")
            print(f"  Average:          {statistics.mean(latencies):.2f}ms")
            print(f"  Median:           {statistics.median(latencies):.2f}ms")
            if len(latencies) > 1:
                print(f"  Stdev:            {statistics.stdev(latencies):.2f}ms")

            duration = 60
            throughput = len(latencies) / duration
            print(f"\nThroughput:         {throughput:.2f} req/s")
            print(f"Success Rate:       {100 * len(latencies) / total_requests:.1f}%")

        # Get load balancer metrics
        try:
            response = httpx.get("http://127.0.0.1:8000/metrics", timeout=5)
            if response.status_code == 200:
                metrics = response.json()
                print(f"\nLoad Balancer Metrics:")
                print(f"  Total Requests (LB): {metrics.get('total_requests', 'N/A')}")
                print(f"  Worker Health:")
                for worker_id, health in metrics.get("worker_health", {}).items():
                    print(f"    {worker_id}: healthy={health['healthy']}, requests={health.get('current_load', 0)}, errors={health['error_count']}")
        except:
            pass

        print("\n" + "="*70)

    finally:
        print("\nShutting down...")
        lb_proc.terminate()
        for proc in worker_procs:
            proc.terminate()

        time.sleep(1)
        lb_proc.kill()
        for proc in worker_procs:
            proc.kill()

        print("✓ All services stopped")


if __name__ == "__main__":
    asyncio.run(main())
