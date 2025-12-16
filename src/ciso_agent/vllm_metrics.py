# Copyright contributors to the ITBench project. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
vLLM Prometheus Metrics Collector

Collects vLLM metrics per test case using Prometheus time-range queries.
This ensures metrics are isolated to each test case and unaffected by previous runs.
"""

import json
import os
import time
import requests
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def is_vllm_metrics_enabled() -> bool:
    """
    Check if vLLM metrics collection is enabled.
    
    Returns:
        True if VLLM_PROMETHEUS_URL environment variable is set
    """
    return bool(os.getenv("VLLM_PROMETHEUS_URL"))


class VLLMMetricsCollector:
    """
    Collects vLLM Prometheus metrics for a specific time window.
    
    Usage:
        if is_vllm_metrics_enabled():
            collector = VLLMMetricsCollector()
            collector.start_collection("test_case_1")
            # ... run agent test case ...
            metrics = collector.end_collection()
            collector.save_metrics("/path/to/output")
    """

    # Metrics to collect with their PromQL query templates
    # {duration} will be replaced with actual duration in seconds
    METRIC_QUERIES = {
        "prompt_tokens": "increase(vllm:prompt_tokens_total[{duration}s])",
        "generation_tokens": "increase(vllm:generation_tokens_total[{duration}s])",
        "requests_completed": "increase(vllm:request_success_total[{duration}s])",
        "avg_ttft_seconds": (
            "rate(vllm:time_to_first_token_seconds_sum[{duration}s]) / "
            "rate(vllm:time_to_first_token_seconds_count[{duration}s])"
        ),
        "avg_e2e_latency_seconds": (
            "rate(vllm:e2e_request_latency_seconds_sum[{duration}s]) / "
            "rate(vllm:e2e_request_latency_seconds_count[{duration}s])"
        ),
        "avg_prefill_time_seconds": (
            "rate(vllm:request_prefill_time_seconds_sum[{duration}s]) / "
            "rate(vllm:request_prefill_time_seconds_count[{duration}s])"
        ),
        "avg_decode_time_seconds": (
            "rate(vllm:request_decode_time_seconds_sum[{duration}s]) / "
            "rate(vllm:request_decode_time_seconds_count[{duration}s])"
        ),
        "peak_gpu_cache_usage": "max_over_time(vllm:gpu_cache_usage_perc[{duration}s])",
        "peak_cpu_cache_usage": "max_over_time(vllm:cpu_cache_usage_perc[{duration}s])",
        "max_requests_running": "max_over_time(vllm:num_requests_running[{duration}s])",
        "max_requests_waiting": "max_over_time(vllm:num_requests_waiting[{duration}s])",
    }

    def __init__(
        self,
        prometheus_url: Optional[str] = None,
        vllm_metrics_url: Optional[str] = None,
        idle_timeout: int = 60,
        idle_poll_interval: float = 0.5,
    ):
        """
        Initialize the metrics collector.

        Args:
            prometheus_url: URL of Prometheus server (default: http://localhost:8000)
            vllm_metrics_url: URL of vLLM metrics endpoint (default: http://localhost:8000/metrics)
            idle_timeout: Max seconds to wait for vLLM to become idle
            idle_poll_interval: Seconds between idle state checks
        """
        self.prometheus_url = prometheus_url or os.getenv("VLLM_PROMETHEUS_URL")
        if not self.prometheus_url:
            raise RuntimeError(
                "VLLM_PROMETHEUS_URL environment variable must be set to use vLLM metrics collection"
            )

        # vLLM metrics endpoint is the prometheus_url + /metrics
        self.vllm_metrics_url = vllm_metrics_url or f"{self.prometheus_url}/metrics"
        self.idle_timeout = idle_timeout
        self.idle_poll_interval = idle_poll_interval

        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.test_case_id: Optional[str] = None
        self._metrics_cache: Optional[Dict[str, Any]] = None

    def start_collection(self, test_case_id: Optional[str] = None) -> None:
        """
        Start metrics collection for a test case.
        Waits for vLLM to be idle before recording the start timestamp.

        Args:
            test_case_id: Unique identifier for this test case (auto-generated if not provided)
        """
        self._metrics_cache = None
        self.test_case_id = test_case_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Wait for vLLM to be idle before starting
        if not self.wait_for_idle():
            print(f"[WARN] vLLM not idle after {self.idle_timeout}s, proceeding anyway")

        self.start_time = datetime.now(timezone.utc)
        print(f"[INFO] vLLM metrics collection started for test case: {self.test_case_id}")

    def end_collection(self) -> Dict[str, Any]:
        """
        End metrics collection and query Prometheus for metrics in the time window.
        Waits for vLLM to complete all pending requests before recording end timestamp.

        Returns:
            Dictionary containing all collected metrics
        """
        if self.start_time is None:
            raise RuntimeError("start_collection() must be called before end_collection()")

        # Wait for all requests to complete
        if not self.wait_for_idle():
            print(f"[WARN] vLLM not idle after {self.idle_timeout}s, metrics may be incomplete")

        self.end_time = datetime.now(timezone.utc)
        self._metrics_cache = self._query_metrics()

        print(f"[INFO] vLLM metrics collection ended for test case: {self.test_case_id}")
        return self._metrics_cache

    def wait_for_idle(self, timeout: Optional[int] = None) -> bool:
        """
        Wait until vLLM has no running or waiting requests.

        Args:
            timeout: Max seconds to wait (uses instance default if not provided)

        Returns:
            True if vLLM became idle, False if timeout occurred
        """
        timeout = timeout if timeout is not None else self.idle_timeout
        start = time.time()

        while time.time() - start < timeout:
            try:
                running, waiting = self._get_request_counts()
                if running == 0 and waiting == 0:
                    return True
            except Exception as e:
                print(f"[WARN] Failed to check vLLM idle state: {e}")
                # If we can't connect, assume it's idle or not running
                return True

            time.sleep(self.idle_poll_interval)

        return False

    def _get_request_counts(self) -> tuple:
        """
        Get current running and waiting request counts from vLLM metrics endpoint.

        Returns:
            Tuple of (running_count, waiting_count)
        """
        response = requests.get(self.vllm_metrics_url, timeout=5)
        response.raise_for_status()

        running = 0
        waiting = 0

        for line in response.text.split("\n"):
            if line.startswith("vllm:num_requests_running"):
                # Parse: vllm:num_requests_running{...} value
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        running = int(float(parts[-1]))
                    except ValueError:
                        pass
            elif line.startswith("vllm:num_requests_waiting"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        waiting = int(float(parts[-1]))
                    except ValueError:
                        pass

        return running, waiting

    def _query_metrics(self) -> Dict[str, Any]:
        """
        Execute PromQL queries for all metrics in the time range.

        Returns:
            Dictionary with test case info and all metric values
        """
        duration = (self.end_time - self.start_time).total_seconds()
        # Ensure minimum duration for Prometheus queries
        duration = max(duration, 1)

        metrics = {
            "test_case_id": self.test_case_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": duration,
        }

        for metric_name, query_template in self.METRIC_QUERIES.items():
            query = query_template.format(duration=int(duration) + 5)  # Add buffer for scrape interval
            try:
                value = self._prometheus_query(query, self.end_time)
                metrics[metric_name] = value
            except Exception as e:
                print(f"[WARN] Failed to query metric '{metric_name}': {e}")
                metrics[metric_name] = None

        return metrics

    def _prometheus_query(self, query: str, time: datetime) -> Optional[float]:
        """
        Execute an instant query against Prometheus.

        Args:
            query: PromQL query string
            time: Timestamp for the query

        Returns:
            Query result as float, or None if no data
        """
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={
                "query": query,
                "time": time.timestamp(),
            },
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {data.get('error', 'unknown error')}")

        result = data.get("data", {}).get("result", [])
        if not result:
            return None

        # Return the first result value
        try:
            value = float(result[0]["value"][1])
            # Handle NaN from division by zero in rate queries
            if value != value:  # NaN check
                return None
            return value
        except (IndexError, KeyError, ValueError):
            return None

    def save_metrics(self, output_dir: str) -> str:
        """
        Save collected metrics to a JSON file.

        Args:
            output_dir: Directory to save the metrics file

        Returns:
            Path to the saved metrics file
        """
        if self._metrics_cache is None:
            raise RuntimeError("No metrics to save. Call end_collection() first.")

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"vllm_metrics_{self.test_case_id}.json")

        with open(filepath, "w") as f:
            json.dump(self._metrics_cache, f, indent=2)

        print(f"[INFO] vLLM metrics saved to: {filepath}")
        return filepath

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the collected metrics without saving.

        Returns:
            Collected metrics dictionary or None if not yet collected
        """
        return self._metrics_cache


def collect_vllm_metrics(
    test_case_id: str,
    output_dir: str,
    prometheus_url: Optional[str] = None,
    vllm_metrics_url: Optional[str] = None,
):
    """
    Context manager for collecting vLLM metrics around a test case.
    
    Only collects metrics if VLLM_PROMETHEUS_URL environment variable is set.

    Usage:
        with collect_vllm_metrics("test_1", "/output/dir") as collector:
            # ... run test case ...
        # Metrics are automatically saved when exiting the context (if enabled)

    Args:
        test_case_id: Unique identifier for the test case
        output_dir: Directory to save metrics JSON
        prometheus_url: Optional Prometheus server URL
        vllm_metrics_url: Optional vLLM metrics endpoint URL

    Yields:
        VLLMMetricsCollector instance or None if not enabled
    """
    class MetricsContext:
        def __init__(self):
            self.collector = None
            if is_vllm_metrics_enabled():
                try:
                    self.collector = VLLMMetricsCollector(
                        prometheus_url=prometheus_url,
                        vllm_metrics_url=vllm_metrics_url,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to initialize vLLM metrics collector: {e}")

        def __enter__(self):
            if self.collector is not None:
                self.collector.start_collection(test_case_id)
            return self.collector

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.collector is not None:
                try:
                    self.collector.end_collection()
                    self.collector.save_metrics(output_dir)
                except Exception as e:
                    print(f"[ERROR] Failed to save vLLM metrics: {e}")
            return False  # Don't suppress exceptions

    return MetricsContext()
