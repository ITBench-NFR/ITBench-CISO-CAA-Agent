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

import json
import os
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
    except ImportError:
        from langchain.callbacks import BaseCallbackHandler

from ciso_agent.streaming_llm import init_llm, get_llm_params

load_dotenv()


class FirstTokenCallback(BaseCallbackHandler):
    """Callback handler to capture the time when the first token is received."""
    
    def __init__(self):
        super().__init__()
        self.first_token_time = None
        self.request_start_time = None
        self.first_token_received = False
        self.token_count = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when the LLM starts running."""
        self.request_start_time = time.time()
        self.first_token_received = False
        self.token_count = 0
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated."""
        if not self.first_token_received:
            self.first_token_time = time.time()
            self.first_token_received = True
        self.token_count += 1
    
    def get_ttft(self) -> float:
        """Calculate Time to First Token in seconds."""
        if self.first_token_time is None or self.request_start_time is None:
            return None
        return self.first_token_time - self.request_start_time


class TTFTBenchmarkAgent(object):
    """
    Agent for benchmarking Time to First Token (TTFT) metric.
    
    TTFT measures the latency from when a request is sent to when the first token
    is generated. This is critical for interactive IT bots.
    """
    
    agent_goal: str = """Benchmark Time to First Token (TTFT) for LLM interactions.
    This agent measures the latency from request submission to first token generation."""
    
    tool_description: str = """This agent performs TTFT benchmarking by:
    - Making streaming LLM calls
    - Measuring time from request start to first token
    - Running multiple iterations for statistical accuracy
    - Reporting metrics (mean, median, min, max, stddev)"""
    
    input_description: dict = {
        "prompt": "The prompt to send to the LLM for benchmarking",
        "num_iterations": "Number of iterations to run (default: 10)",
        "workdir": "Working directory to save results",
    }
    
    output_description: dict = {
        "ttft_results": "Dictionary containing TTFT metrics and statistics",
        "path_to_benchmark_results": "Path to JSON file with detailed results",
    }
    
    workdir_root: str = "/tmp/agent/"
    
    def kickoff(self, inputs: dict):
        """Entry point for the TTFT benchmark agent."""
        # Extract parameters from inputs
        goal = inputs.get("goal", "")
        workdir = inputs.get("workdir")
        
        # Try to extract prompt and iterations from goal if provided
        prompt = inputs.get("prompt")
        num_iterations = inputs.get("num_iterations", 10)
        
        # If prompt not provided, try to extract from goal
        if not prompt and goal:
            # Look for prompt in goal text
            if "prompt:" in goal.lower():
                # Try to extract prompt from goal
                lines = goal.split("\n")
                for line in lines:
                    if "prompt:" in line.lower():
                        prompt = line.split(":", 1)[1].strip()
                        break
            
            # If still no prompt, use a default based on goal
            if not prompt:
                prompt = goal if goal and len(goal) < 200 else "What is the capital of France?"
        
        # Extract iterations from goal if specified
        if "iterations:" in goal.lower() or "num_iterations:" in goal.lower():
            import re
            match = re.search(r'(?:iterations|num_iterations):\s*(\d+)', goal, re.IGNORECASE)
            if match:
                num_iterations = int(match.group(1))
        
        return self.run_benchmark(
            prompt=prompt or "What is the capital of France?",
            num_iterations=num_iterations,
            workdir=workdir,
        )
    
    def run_benchmark(
        self,
        prompt: str = "What is the capital of France?",
        num_iterations: int = 10,
        workdir: str = None,
        **kwargs
    ) -> dict:
        """
        Run TTFT benchmark with multiple iterations.
        
        Args:
            prompt: The prompt to send to the LLM
            num_iterations: Number of benchmark iterations to run
            workdir: Working directory to save results
            **kwargs: Additional arguments (e.g., goal, kubeconfig, etc.)
        
        Returns:
            Dictionary containing benchmark results
        """
        if workdir is None:
            workdir = os.path.join(
                self.workdir_root,
                datetime.now().strftime("%Y%m%d%H%M%S_ttft_benchmark"),
                "workspace"
            )
        
        if not os.path.exists(workdir):
            os.makedirs(workdir, exist_ok=True)
        
        
        # Initialize LLM
        model, api_url, api_key = get_llm_params()
        llm = init_llm(model=model, api_url=api_url, api_key=api_key)
        
        if not llm:
            raise ValueError("Failed to initialize LLM. Check your environment variables.")
        
        # Run benchmark iterations
        ttft_values: List[float] = []
        detailed_results: List[Dict[str, Any]] = []
        
        for iteration in range(1, num_iterations + 1):
            print(f"Running iteration {iteration}/{num_iterations}...", end=" ", flush=True)
            
            try:
                # Prepare messages
                messages = [HumanMessage(content=prompt)]
                
                # Make streaming call
                start_time = time.time()
                try:
                    # Don't pass callbacks to stream() - manually track timing instead
                    response = llm.stream(messages)
                    
                    # Consume the stream and track first token time
                    full_response = ""
                    first_chunk_time = None
                    token_count = 0
                    for chunk in response:
                        if first_chunk_time is None:
                            # This is the first chunk - record the time
                            first_chunk_time = time.time()
                        if hasattr(chunk, 'content') and chunk.content:
                            full_response += chunk.content
                            # Estimate tokens (rough approximation: ~4 chars per token)
                            token_count += len(chunk.content) // 4
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    # TTFT is the time from start to first chunk
                    if first_chunk_time is not None:
                        ttft = first_chunk_time - start_time
                    else:
                        ttft = None
                
                except (AttributeError, TypeError) as e:
                    # Streaming not supported, fall back to non-streaming with timing
                    print(f"  (Streaming not supported: {type(e).__name__}: {str(e)}, using fallback method)", end=" ", flush=True)
                    response = llm.invoke(messages)
                    end_time = time.time()
                    total_time = end_time - start_time
                    # For non-streaming, we can't measure true TTFT, so we use total time as approximation
                    # This is not ideal but provides some measurement
                    ttft = total_time
                    full_response = response.content if hasattr(response, 'content') else str(response)
                    # Estimate token count for fallback
                    token_count = len(full_response) // 4
                
                if ttft is not None:
                    ttft_values.append(ttft)
                    detailed_results.append({
                        "iteration": iteration,
                        "ttft_seconds": ttft,
                        "total_time_seconds": total_time,
                        "tokens_generated": token_count,
                        "timestamp": datetime.now().isoformat(),
                    })
                    print(f"✓ TTFT: {ttft:.4f}s")
                else:
                    print("✗ Failed to capture TTFT")
                    detailed_results.append({
                        "iteration": iteration,
                        "ttft_seconds": None,
                        "error": "Failed to capture first token",
                        "timestamp": datetime.now().isoformat(),
                    })
            
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                detailed_results.append({
                    "iteration": iteration,
                    "ttft_seconds": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
            
            # Small delay between iterations to avoid rate limiting
            if iteration < num_iterations:
                time.sleep(0.5)
        
        # Calculate statistics
        if ttft_values:
            stats = {
                "mean": statistics.mean(ttft_values),
                "median": statistics.median(ttft_values),
                "min": min(ttft_values),
                "max": max(ttft_values),
                "stddev": statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0.0,
                "count": len(ttft_values),
            }
        else:
            stats = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "stddev": None,
                "count": 0,
            }
        
        # Prepare results
        results = {
            "metric": "Time to First Token (TTFT)",
            "model": model,
            "api_url": api_url if api_url else "default",
            "prompt": prompt,
            "num_iterations": num_iterations,
            "successful_iterations": len(ttft_values),
            "statistics": stats,
            "detailed_results": detailed_results,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results to file
        results_file = os.path.join(workdir, "ttft_benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("TTFT BENCHMARK RESULTS")
        print("=" * 80)
        
        if stats["count"] > 0:
            print(f"{'Successful Iterations':<25} {stats['count']}/{num_iterations}")
            print(f"{'Mean TTFT':<25} {stats['mean']:.4f} seconds")
            print(f"{'Median TTFT':<25} {stats['median']:.4f} seconds")
            print(f"{'Min TTFT':<25} {stats['min']:.4f} seconds")
            print(f"{'Max TTFT':<25} {stats['max']:.4f} seconds")
            print(f"{'Std Dev':<25} {stats['stddev']:.4f} seconds")
        else:
            print("No successful iterations completed.")
        
        print(f"\n{'='*80}")
        print(f"Detailed results saved to: {results_file}")
        print(f"{'='*80}\n")
        
        return {
            "result": {
                "ttft_results": results,
                "path_to_benchmark_results": results_file,
            }
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TTFT Benchmark Agent")
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="Prompt to use for benchmarking"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory for results"
    )
    
    args = parser.parse_args()
    
    agent = TTFTBenchmarkAgent()
    result = agent.run_benchmark(
        prompt=args.prompt,
        num_iterations=args.iterations,
        workdir=args.workdir,
    )
    
    print("\nFinal Result:")
    print(json.dumps(result, indent=2))
