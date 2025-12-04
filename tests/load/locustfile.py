"""
Load testing scenarios for RAG System using Locust.

Usage:
    locust -f locustfile.py --host=http://localhost:8000

Web UI will be available at http://localhost:8089
"""

from locust import HttpUser, task, between, events
import random
import json


# Sample queries for load testing
SAMPLE_QUERIES = [
    "What is machine learning?",
    "Explain neural networks",
    "How does deep learning work?",
    "What is the difference between AI and ML?",
    "Explain backpropagation algorithm",
    "What are convolutional neural networks?",
    "How does gradient descent work?",
    "What is overfitting in machine learning?",
    "Explain the attention mechanism",
    "What is transfer learning?",
    "How do transformers work?",
    "What is reinforcement learning?",
    "Explain the $E=mc^2$ formula",
    "What is the Pythagorean theorem?",
    "How does logistic regression work?",
]


class RAGSystemUser(HttpUser):
    """
    Simulates a user interacting with the RAG system.
    """

    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)

    def on_start(self):
        """Called when a simulated user starts."""
        self.correlation_ids = []

    @task(10)
    def query_simple(self):
        """
        Simple query without optional parameters.
        Weight: 10 (most common)
        """
        query = random.choice(SAMPLE_QUERIES)

        with self.client.post(
            "/api/v1/query",
            json={
                "query": query,
                "output_format": "markdown"
            },
            catch_response=True,
            name="/api/v1/query [simple]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.correlation_ids.append(data.get("correlation_id"))
                    response.success()
                else:
                    response.failure(f"Query failed: {data}")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def query_with_options(self):
        """
        Query with custom options.
        Weight: 5 (less common)
        """
        query = random.choice(SAMPLE_QUERIES)
        output_format = random.choice(["markdown", "html", "json"])

        with self.client.post(
            "/api/v1/query",
            json={
                "query": query,
                "output_format": output_format,
                "retrieval_top_k": random.randint(10, 30),
                "ranking_top_k": random.randint(5, 15),
                "enable_query_analysis": True,
                "enable_ranking": True
            },
            catch_response=True,
            name="/api/v1/query [with-options]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Query failed: {data}")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def query_with_latex(self):
        """
        Query containing LaTeX formulas.
        Weight: 2 (rare)
        """
        latex_queries = [q for q in SAMPLE_QUERIES if "$" in q]
        if latex_queries:
            query = random.choice(latex_queries)
        else:
            query = "Explain the formula $E=mc^2$"

        with self.client.post(
            "/api/v1/query",
            json={
                "query": query,
                "output_format": "html",
                "enable_latex_parsing": True
            },
            catch_response=True,
            name="/api/v1/query [latex]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """
        Health check endpoint.
        Weight: 1 (occasional monitoring)
        """
        with self.client.get(
            "/api/v1/health",
            catch_response=True,
            name="/api/v1/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class RAGSystemStressUser(HttpUser):
    """
    Aggressive user for stress testing.
    No wait time between requests.
    """

    wait_time = between(0.1, 0.5)  # Very short wait

    @task
    def rapid_fire_queries(self):
        """Fire queries rapidly."""
        query = random.choice(SAMPLE_QUERIES)

        self.client.post(
            "/api/v1/query",
            json={
                "query": query,
                "output_format": "markdown"
            },
            name="/api/v1/query [stress]"
        )


class RAGSystemSoakUser(HttpUser):
    """
    Slow, steady user for soak testing.
    Longer wait times, simulates real user behavior.
    """

    wait_time = between(5, 15)  # 5-15 seconds between requests

    @task
    def realistic_query(self):
        """Realistic query with reading time."""
        query = random.choice(SAMPLE_QUERIES)

        self.client.post(
            "/api/v1/query",
            json={
                "query": query,
                "output_format": "markdown"
            },
            name="/api/v1/query [soak]"
        )


# Event hooks for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the load test starts."""
    print("🚀 Load test starting...")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the load test stops."""
    print("\n📊 Load test completed!")
    print("\nSummary:")

    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Success rate: {(1 - stats.total.fail_ratio) * 100:.2f}%")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Median response time: {stats.total.median_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    print(f"Requests per second: {stats.total.total_rps:.2f}")


# Custom scenarios for different test types
class QuickLoadTest(HttpUser):
    """Quick load test - 30 seconds."""
    wait_time = between(1, 2)
    tasks = [RAGSystemUser.query_simple]


class SustainedLoadTest(HttpUser):
    """Sustained load test - simulate normal usage."""
    wait_time = between(2, 5)
    tasks = [RAGSystemUser.query_simple, RAGSystemUser.query_with_options]


class PeakLoadTest(HttpUser):
    """Peak load test - simulate high traffic."""
    wait_time = between(0.5, 1)
    tasks = [RAGSystemUser.query_simple, RAGSystemUser.query_with_options]
