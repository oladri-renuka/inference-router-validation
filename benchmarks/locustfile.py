import random
import time
from locust import HttpUser, task, between, events
from statistics import mean, median, stdev


class InferenceUser(HttpUser):
    """Simulated inference client."""

    wait_time = between(0.5, 2)

    # Traffic distribution
    SHORT_PROMPTS = [
        "What is ML?",
        "Define AI",
        "Explain backprop",
        "What is a neural net?",
        "Define deep learning",
    ]

    MEDIUM_PROMPTS = [
        "Explain how neural networks work in detail",
        "What are the main types of machine learning?",
        "How does gradient descent optimize weights?",
        "Describe convolutional neural networks",
        "What is the role of activation functions?",
    ]

    LONG_PROMPTS = [
        "Write a comprehensive guide to transformer architectures including attention mechanisms, multi-head attention, positional encoding, and modern variations like Vision Transformers",
        "Explain the complete training pipeline for large language models including data preprocessing, tokenization, distributed training, fine-tuning, and evaluation metrics",
        "Describe advanced optimization techniques in deep learning including Adam, RMSprop, learning rate scheduling, batch normalization, layer normalization, and their interactions",
        "Provide a detailed analysis of transfer learning approaches, domain adaptation, meta-learning, and few-shot learning techniques",
        "Explain the mathematical foundations and practical implementations of reinforcement learning algorithms from policy gradient to actor-critic methods",
    ]

    @task(60)
    def short_request(self):
        """60% short requests (low cost)."""
        prompt = random.choice(self.SHORT_PROMPTS)
        self._send_request(prompt)

    @task(30)
    def medium_request(self):
        """30% medium requests (medium cost)."""
        prompt = random.choice(self.MEDIUM_PROMPTS)
        self._send_request(prompt)

    @task(10)
    def long_request(self):
        """10% long requests (high cost)."""
        prompt = random.choice(self.LONG_PROMPTS)
        self._send_request(prompt)

    def _send_request(self, prompt: str):
        """Send inference request."""
        with self.client.post(
            "/infer",
            json={"prompt": prompt},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


# Metrics collection
latencies = []
request_count = 0


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Track latency metrics."""
    global latencies, request_count

    if exception is None:
        latencies.append(response_time)
        request_count += 1


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Report metrics when test stops."""
    global latencies, request_count

    if latencies:
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total Requests: {request_count}")
        print(f"Average Latency: {mean(latencies):.2f}ms")
        print(f"Median Latency: {median(latencies):.2f}ms")
        print(f"P50 Latency: {p50:.2f}ms")
        print(f"P95 Latency: {p95:.2f}ms")
        print(f"P99 Latency: {p99:.2f}ms")
        if len(latencies) > 1:
            print(f"Std Dev: {stdev(latencies):.2f}ms")
        print(f"Throughput: {request_count / kwargs.get('duration', 1):.2f} req/s")
        print("=" * 60)
