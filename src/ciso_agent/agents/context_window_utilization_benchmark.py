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
from typing import List, Dict, Any, Optional
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

from ciso_agent.llm import init_llm, get_llm_params

load_dotenv()


class TokenUsageCallback(BaseCallbackHandler):
    """Callback handler to track token usage during LLM interactions."""
    
    def __init__(self):
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.token_count = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when the LLM starts running."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.token_count = 0
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when the LLM finishes running."""
        # Try to extract token usage from response metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('token_usage', {})
            if usage:
                self.input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                self.output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                self.total_tokens = usage.get('total_tokens', 0) or (self.input_tokens + self.output_tokens)
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated (for streaming)."""
        self.token_count += 1
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "streamed_token_count": self.token_count,
        }


def get_model_context_window(model: str) -> int:
    """
    Get the maximum context window length for a given model.
    
    This function contains a mapping of common models to their context window sizes.
    If the model is not found, it returns a default value based on model characteristics.
    """
    model_lower = model.lower()
    
    # Context window sizes for common models (in tokens)
    context_windows = {
        # OpenAI models
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        
        # Anthropic models
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-2": 200000,
        
        # Llama models
        "llama-3-8b": 8192,
        "llama-3-70b": 8192,
        "llama-3-405b": 128000,
        "llama-3-8b-instruct": 8192,
        "llama-3-70b-instruct": 8192,
        "llama-3-405b-instruct": 128000,
        "llama-3.1-8b": 131072,
        "llama-3.1-70b": 131072,
        "llama-3.1-405b": 131072,
        
        # Mistral models
        "mixtral-8x7b": 32768,
        "mistral-7b": 8192,
        "mistral-large": 32768,
        
        # Google models
        "gemini-pro": 32768,
        "gemini-1.5-pro": 2097152,  # 2M tokens
        "gemini-1.5-flash": 1048576,  # 1M tokens
        
        # IBM models
        "granite": 8192,
        "granite-3": 8192,
        
        # Qwen models
        "qwen": 32768,
        "qwen2": 131072,
    }
    
    # Check for exact match
    for key, window in context_windows.items():
        if key in model_lower:
            return window
    
    # Check for model size patterns
    if "70b" in model_lower or "70-b" in model_lower:
        if "llama-3.1" in model_lower:
            return 131072
        return 8192
    elif "8b" in model_lower or "8-b" in model_lower:
        if "llama-3.1" in model_lower:
            return 131072
        return 8192
    elif "405b" in model_lower or "405-b" in model_lower:
        if "llama-3.1" in model_lower:
            return 131072
        return 128000
    
    # Default fallback: check environment variable or use conservative default
    default_window = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
    return default_window


class ContextWindowUtilizationBenchmarkAgent(object):
    """
    Agent for benchmarking Context Window Utilization (CWU) metric.
    
    Context Window Utilization measures the percentage of an LLM's available context
    capacity actually used during inference. CWU = U/L, where U is the number of tokens
    utilized (input + output) and L is the total context window length.
    """
    
    agent_goal: str = """Benchmark Context Window Utilization (CWU) for LLM interactions.
    This agent measures the percentage of available context capacity used during inference.
    CWU = (Input Tokens + Output Tokens) / Context Window Length."""
    
    tool_description: str = """This agent performs Context Window Utilization benchmarking by:
    - Making LLM calls with varying prompt sizes
    - Tracking input and output token usage
    - Calculating utilization percentage (used tokens / context window length)
    - Running multiple iterations for statistical accuracy
    - Reporting metrics (mean, median, min, max, stddev)"""
    
    input_description: dict = {
        "prompt": "The prompt to send to the LLM for benchmarking",
        "num_iterations": "Number of iterations to run (default: 10)",
        "workdir": "Working directory to save results",
        "context_window_length": "Optional: Override the model's context window length (in tokens)",
    }
    
    output_description: dict = {
        "cwu_results": "Dictionary containing CWU metrics and statistics",
        "path_to_benchmark_results": "Path to JSON file with detailed results",
    }
    
    workdir_root: str = "/tmp/agent/"
    
    def kickoff(self, inputs: dict):
        """Entry point for the Context Window Utilization benchmark agent."""
        # Extract parameters from inputs
        goal = inputs.get("goal", "")
        workdir = inputs.get("workdir")
        
        # Try to extract prompt and iterations from goal if provided
        prompt = inputs.get("prompt")
        num_iterations = inputs.get("num_iterations", 10)
        context_window_length = inputs.get("context_window_length")
        
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
                prompt = goal if goal and len(goal) < 500 else "Explain the concept of machine learning in detail, including supervised learning, unsupervised learning, and reinforcement learning."
        
        # Extract iterations from goal if specified
        if "iterations:" in goal.lower() or "num_iterations:" in goal.lower():
            import re
            match = re.search(r'(?:iterations|num_iterations):\s*(\d+)', goal, re.IGNORECASE)
            if match:
                num_iterations = int(match.group(1))
        
        # Extract context window length from goal if specified
        if "context_window" in goal.lower() and context_window_length is None:
            import re
            match = re.search(r'context[_\s-]?window[:\s]+(\d+)', goal, re.IGNORECASE)
            if match:
                context_window_length = int(match.group(1))
        
        return self.run_benchmark(
            prompt=prompt or "Explain the concept of machine learning in detail, including supervised learning, unsupervised learning, and reinforcement learning.",
            num_iterations=num_iterations,
            workdir=workdir,
            context_window_length=context_window_length,
        )
    
    def run_benchmark(
        self,
        prompt: str = "Explain the concept of machine learning in detail, including supervised learning, unsupervised learning, and reinforcement learning.",
        num_iterations: int = 10,
        workdir: str = None,
        context_window_length: Optional[int] = None,
        **kwargs
    ) -> dict:
        """
        Run Context Window Utilization benchmark with multiple iterations.
        
        Args:
            prompt: The prompt to send to the LLM
            num_iterations: Number of benchmark iterations to run
            workdir: Working directory to save results
            context_window_length: Optional override for context window length (in tokens)
            **kwargs: Additional arguments (e.g., goal, kubeconfig, etc.)
        
        Returns:
            Dictionary containing benchmark results
        """
        if workdir is None:
            workdir = os.path.join(
                self.workdir_root,
                datetime.now().strftime("%Y%m%d%H%M%S_cwu_benchmark"),
                "workspace"
            )
        
        if not os.path.exists(workdir):
            os.makedirs(workdir, exist_ok=True)
        
        # Initialize LLM
        model, api_url, api_key = get_llm_params()
        llm = init_llm(model=model, api_url=api_url, api_key=api_key)
        
        if not llm:
            raise ValueError("Failed to initialize LLM. Check your environment variables.")
        
        # Get context window length
        if context_window_length is None:
            context_window_length = get_model_context_window(model)
        
        # Run benchmark iterations
        cwu_values: List[float] = []
        detailed_results: List[Dict[str, Any]] = []
        
        for iteration in range(1, num_iterations + 1):
            print(f"Running iteration {iteration}/{num_iterations}...", end=" ", flush=True)
            
            callback = TokenUsageCallback()
            
            try:
                # Prepare messages
                messages = [HumanMessage(content=prompt)]
                
                # Make LLM call (try streaming first, fallback to non-streaming)
                start_time = time.time()
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0
                
                try:
                    # Try streaming first
                    response = llm.stream(messages, callbacks=[callback])
                    
                    # Consume the stream to trigger callbacks
                    full_response = ""
                    for chunk in response:
                        if hasattr(chunk, 'content') and chunk.content:
                            full_response += chunk.content
                    
                    # After streaming, try to get token usage from metadata
                    # Some providers expose this differently
                    if hasattr(response, 'response_metadata'):
                        usage = response.response_metadata.get('token_usage', {}) if hasattr(response, 'response_metadata') else {}
                        if usage:
                            input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                            output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                            total_tokens = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
                    
                    # If we got token usage from callback, use it
                    callback_usage = callback.get_token_usage()
                    if callback_usage['input_tokens'] > 0 or callback_usage['output_tokens'] > 0:
                        input_tokens = callback_usage['input_tokens']
                        output_tokens = callback_usage['output_tokens']
                        total_tokens = callback_usage['total_tokens']
                    
                except (AttributeError, TypeError) as e:
                    # Streaming not supported, fall back to non-streaming
                    print(f"  (Streaming not supported, using non-streaming)", end=" ", flush=True)
                    response = llm.invoke(messages, callbacks=[callback])
                    full_response = response.content if hasattr(response, 'content') else str(response)
                    
                    # Extract token usage from response metadata
                    if hasattr(response, 'response_metadata') and response.response_metadata:
                        usage = response.response_metadata.get('token_usage', {})
                        if usage:
                            input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                            output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                            total_tokens = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
                    
                    # Check callback as fallback
                    callback_usage = callback.get_token_usage()
                    if callback_usage['input_tokens'] > 0 or callback_usage['output_tokens'] > 0:
                        input_tokens = callback_usage['input_tokens']
                        output_tokens = callback_usage['output_tokens']
                        total_tokens = callback_usage['total_tokens']
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # If we still don't have token counts, estimate them
                # Rough estimation: ~4 characters per token for English text
                if input_tokens == 0:
                    # Estimate input tokens from prompt
                    input_tokens = len(prompt) // 4
                    # Add overhead for message formatting
                    input_tokens += 10
                
                if output_tokens == 0:
                    # Estimate output tokens from response
                    output_tokens = len(full_response) // 4
                
                if total_tokens == 0:
                    total_tokens = input_tokens + output_tokens
                
                # Calculate Context Window Utilization
                # CWU = (Input Tokens + Output Tokens) / Context Window Length
                cwu_percentage = (total_tokens / context_window_length) * 100 if context_window_length > 0 else 0
                
                cwu_values.append(cwu_percentage)
                detailed_results.append({
                    "iteration": iteration,
                    "cwu_percentage": cwu_percentage,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "context_window_length": context_window_length,
                    "total_time_seconds": total_time,
                    "timestamp": datetime.now().isoformat(),
                })
                print(f"✓ CWU: {cwu_percentage:.2f}% ({total_tokens}/{context_window_length} tokens)")
            
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                detailed_results.append({
                    "iteration": iteration,
                    "cwu_percentage": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
            
            # Small delay between iterations to avoid rate limiting
            if iteration < num_iterations:
                time.sleep(0.5)
        
        # Calculate statistics
        if cwu_values:
            stats = {
                "mean": statistics.mean(cwu_values),
                "median": statistics.median(cwu_values),
                "min": min(cwu_values),
                "max": max(cwu_values),
                "stddev": statistics.stdev(cwu_values) if len(cwu_values) > 1 else 0.0,
                "count": len(cwu_values),
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
        
        # Calculate average token usage
        avg_input_tokens = statistics.mean([r.get("input_tokens", 0) for r in detailed_results if r.get("input_tokens")])
        avg_output_tokens = statistics.mean([r.get("output_tokens", 0) for r in detailed_results if r.get("output_tokens")])
        avg_total_tokens = statistics.mean([r.get("total_tokens", 0) for r in detailed_results if r.get("total_tokens")])
        
        # Prepare results
        results = {
            "metric": "Context Window Utilization (CWU)",
            "model": model,
            "api_url": api_url if api_url else "default",
            "context_window_length": context_window_length,
            "prompt": prompt,
            "num_iterations": num_iterations,
            "successful_iterations": len(cwu_values),
            "statistics": stats,
            "average_token_usage": {
                "input_tokens": avg_input_tokens,
                "output_tokens": avg_output_tokens,
                "total_tokens": avg_total_tokens,
            },
            "detailed_results": detailed_results,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results to file
        results_file = os.path.join(workdir, "context_window_utilization_benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("CONTEXT WINDOW UTILIZATION (CWU) BENCHMARK RESULTS")
        print("=" * 80)
        
        if stats["count"] > 0:
            print(f"{'Model':<25} {model}")
            print(f"{'Context Window Length':<25} {context_window_length:,} tokens")
            print(f"{'Successful Iterations':<25} {stats['count']}/{num_iterations}")
            print(f"{'Mean CWU':<25} {stats['mean']:.2f}%")
            print(f"{'Median CWU':<25} {stats['median']:.2f}%")
            print(f"{'Min CWU':<25} {stats['min']:.2f}%")
            print(f"{'Max CWU':<25} {stats['max']:.2f}%")
            print(f"{'Std Dev':<25} {stats['stddev']:.2f}%")
            print(f"{'Avg Input Tokens':<25} {avg_input_tokens:.1f}")
            print(f"{'Avg Output Tokens':<25} {avg_output_tokens:.1f}")
            print(f"{'Avg Total Tokens':<25} {avg_total_tokens:.1f}")
        else:
            print("No successful iterations completed.")
        
        print(f"\n{'='*80}")
        print(f"Detailed results saved to: {results_file}")
        print(f"{'='*80}\n")
        
        return {
            "result": {
                "cwu_results": results,
                "path_to_benchmark_results": results_file,
            }
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Context Window Utilization Benchmark Agent")
    parser.add_argument(
        "--prompt",
        default="Explain the concept of machine learning in detail, including supervised learning, unsupervised learning, and reinforcement learning.",
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
    parser.add_argument(
        "--context-window-length",
        type=int,
        default=None,
        help="Override context window length (in tokens)"
    )
    
    args = parser.parse_args()
    
    agent = ContextWindowUtilizationBenchmarkAgent()
    result = agent.run_benchmark(
        prompt=args.prompt,
        num_iterations=args.iterations,
        workdir=args.workdir,
        context_window_length=getattr(args, 'context_window_length', None),
    )
    
    print("\nFinal Result:")
    print(json.dumps(result, indent=2))

