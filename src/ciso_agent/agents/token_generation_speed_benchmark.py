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


class TokenGenerationSpeedCallback(BaseCallbackHandler):
    """Callback handler to measure token generation speed (tokens/sec)."""
    
    def __init__(self):
        super().__init__()
        self.first_token_time = None
        self.last_token_time = None
        self.request_start_time = None
        self.token_count = 0
        self.tokens_received = False
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when the LLM starts running."""
        self.request_start_time = time.time()
        self.first_token_time = None
        self.last_token_time = None
        self.token_count = 0
        self.tokens_received = False
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated."""
        current_time = time.time()
        
        if not self.tokens_received:
            # First token received
            self.first_token_time = current_time
            self.tokens_received = True
        
        # Update last token time for each token
        self.last_token_time = current_time
        self.token_count += 1
    
    def get_token_generation_speed(self) -> float:
        """
        Calculate Token Generation Speed in tokens per second.
        
        This measures pure inference speed by calculating:
        (total_tokens - 1) / (time_from_first_to_last_token)
        
        We subtract 1 from token count because the first token marks the start
        of generation, so we measure speed of the remaining tokens.
        """
        if self.first_token_time is None or self.last_token_time is None:
            return None
        
        if self.token_count < 2:
            # Need at least 2 tokens to calculate speed
            return None
        
        # Time spent generating tokens (from first to last)
        generation_time = self.last_token_time - self.first_token_time
        
        if generation_time <= 0:
            return None
        
        # Tokens generated after the first one
        tokens_generated = self.token_count - 1
        
        # Calculate tokens per second
        tokens_per_second = tokens_generated / generation_time
        
        return tokens_per_second
    
    def get_total_tokens(self) -> int:
        """Get total number of tokens generated."""
        return self.token_count


class TokenGenerationSpeedBenchmarkAgent(object):
    """
    Agent for benchmarking Token Generation Speed (Tokens/Sec) metric.
    
    Token Generation Speed measures the pure inference speed, independent of 
    network overhead. It calculates tokens per second from the first token to 
    the last token, excluding the initial latency (TTFT).
    """
    
    agent_goal: str = """Benchmark Token Generation Speed (Tokens/Sec) for LLM interactions.
    This agent measures the pure inference speed, independent of network overhead."""
    
    tool_description: str = """This agent performs Token Generation Speed benchmarking by:
    - Making streaming LLM calls
    - Measuring tokens generated per second after first token
    - Running multiple iterations for statistical accuracy
    - Reporting metrics (mean, median, min, max, stddev)"""
    
    input_description: dict = {
        "prompt": "The prompt to send to the LLM for benchmarking",
        "num_iterations": "Number of iterations to run (default: 10)",
        "workdir": "Working directory to save results",
    }
    
    output_description: dict = {
        "token_speed_results": "Dictionary containing token generation speed metrics and statistics",
        "path_to_benchmark_results": "Path to JSON file with detailed results",
    }
    
    workdir_root: str = "/tmp/agent/"
    
    def kickoff(self, inputs: dict):
        """Entry point for the Token Generation Speed benchmark agent."""
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
                prompt = goal if goal and len(goal) < 200 else "Write a detailed explanation of how machine learning works."
        
        # Extract iterations from goal if specified
        if "iterations:" in goal.lower() or "num_iterations:" in goal.lower():
            import re
            match = re.search(r'(?:iterations|num_iterations):\s*(\d+)', goal, re.IGNORECASE)
            if match:
                num_iterations = int(match.group(1))
        
        return self.run_benchmark(
            prompt=prompt or "Write a detailed explanation of how machine learning works.",
            num_iterations=num_iterations,
            workdir=workdir,
        )
    
    def run_benchmark(
        self,
        prompt: str = "Write a detailed explanation of how machine learning works.",
        num_iterations: int = 10,
        workdir: str = None,
        **kwargs
    ) -> dict:
        """
        Run Token Generation Speed benchmark with multiple iterations.
        
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
                datetime.now().strftime("%Y%m%d%H%M%S_token_speed_benchmark"),
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
        speed_values: List[float] = []
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
                    
                    # Consume the stream and track timing
                    full_response = ""
                    first_chunk_time = None
                    last_chunk_time = None
                    token_count = 0
                    for chunk in response:
                        current_time = time.time()
                        if first_chunk_time is None:
                            first_chunk_time = current_time
                        last_chunk_time = current_time
                        if hasattr(chunk, 'content') and chunk.content:
                            full_response += chunk.content
                            # Estimate tokens (rough approximation: ~4 chars per token)
                            token_count += len(chunk.content) // 4
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Calculate token generation speed
                    if first_chunk_time is not None and last_chunk_time is not None and token_count > 1:
                        # Time from first token to last token
                        generation_time = last_chunk_time - first_chunk_time
                        if generation_time > 0:
                            # Tokens generated after the first one
                            tokens_generated = token_count - 1
                            tokens_per_second = tokens_generated / generation_time
                        else:
                            tokens_per_second = None
                    else:
                        tokens_per_second = None
                    total_tokens = token_count
                
                except (AttributeError, TypeError) as e:
                    # Streaming not supported, fall back to non-streaming
                    print(f"  (Streaming not supported: {type(e).__name__}: {str(e)}, using fallback method)", end=" ", flush=True)
                    response = llm.invoke(messages)
                    end_time = time.time()
                    total_time = end_time - start_time
                    full_response = response.content if hasattr(response, 'content') else str(response)
                    
                    # For non-streaming, we can't measure true token generation speed
                    # Estimate based on response length (rough approximation)
                    # This is not ideal but provides some measurement
                    if hasattr(response, 'response_metadata') and response.response_metadata:
                        usage = response.response_metadata.get('token_usage', {})
                        total_tokens = usage.get('completion_tokens', 0) or usage.get('total_tokens', 0)
                    else:
                        # Fallback: estimate tokens (rough: ~4 chars per token)
                        total_tokens = len(full_response) // 4
                    
                    if total_tokens > 1 and total_time > 0:
                        # Approximate: assume first token takes longer, use 80% of time for generation
                        generation_time = total_time * 0.8
                        tokens_per_second = (total_tokens - 1) / generation_time if generation_time > 0 else None
                    else:
                        tokens_per_second = None
                
                if tokens_per_second is not None and tokens_per_second > 0:
                    speed_values.append(tokens_per_second)
                    detailed_results.append({
                        "iteration": iteration,
                        "tokens_per_second": tokens_per_second,
                        "total_tokens": total_tokens,
                        "total_time_seconds": total_time,
                        "timestamp": datetime.now().isoformat(),
                    })
                    print(f"✓ {tokens_per_second:.2f} tokens/sec ({total_tokens} tokens)")
                else:
                    print("✗ Failed to capture token generation speed")
                    detailed_results.append({
                        "iteration": iteration,
                        "tokens_per_second": None,
                        "error": "Failed to capture token generation speed",
                        "timestamp": datetime.now().isoformat(),
                    })
            
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                detailed_results.append({
                    "iteration": iteration,
                    "tokens_per_second": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
            
            # Small delay between iterations to avoid rate limiting
            if iteration < num_iterations:
                time.sleep(0.5)
        
        # Calculate statistics
        if speed_values:
            stats = {
                "mean": statistics.mean(speed_values),
                "median": statistics.median(speed_values),
                "min": min(speed_values),
                "max": max(speed_values),
                "stddev": statistics.stdev(speed_values) if len(speed_values) > 1 else 0.0,
                "count": len(speed_values),
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
            "metric": "Token Generation Speed (Tokens/Sec)",
            "model": model,
            "api_url": api_url if api_url else "default",
            "prompt": prompt,
            "num_iterations": num_iterations,
            "successful_iterations": len(speed_values),
            "statistics": stats,
            "detailed_results": detailed_results,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results to file
        results_file = os.path.join(workdir, "token_generation_speed_benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("TOKEN GENERATION SPEED BENCHMARK RESULTS")
        print("=" * 80)
        
        if stats["count"] > 0:
            print(f"{'Successful Iterations':<25} {stats['count']}/{num_iterations}")
            print(f"{'Mean Tokens/Sec':<25} {stats['mean']:.2f} tokens/sec")
            print(f"{'Median Tokens/Sec':<25} {stats['median']:.2f} tokens/sec")
            print(f"{'Min Tokens/Sec':<25} {stats['min']:.2f} tokens/sec")
            print(f"{'Max Tokens/Sec':<25} {stats['max']:.2f} tokens/sec")
            print(f"{'Std Dev':<25} {stats['stddev']:.2f} tokens/sec")
        else:
            print("No successful iterations completed.")
        
        print(f"\n{'='*80}")
        print(f"Detailed results saved to: {results_file}")
        print(f"{'='*80}\n")
        
        return {
            "result": {
                "token_speed_results": results,
                "path_to_benchmark_results": results_file,
            }
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Token Generation Speed Benchmark Agent")
    parser.add_argument(
        "--prompt",
        default="Write a detailed explanation of how machine learning works.",
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
    
    agent = TokenGenerationSpeedBenchmarkAgent()
    result = agent.run_benchmark(
        prompt=args.prompt,
        num_iterations=args.iterations,
        workdir=args.workdir,
    )
    
    print("\nFinal Result:")
    print(json.dumps(result, indent=2))
