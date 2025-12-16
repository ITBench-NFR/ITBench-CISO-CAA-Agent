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


class ToolRoundTripTimeCallback(BaseCallbackHandler):
    """Callback handler to measure tool round-trip time and processing overhead time."""
    
    def __init__(self):
        super().__init__()
        self.tool_calls: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.current_tool_start: Optional[float] = None
        self.current_tool_name: Optional[str] = None
        self.current_llm_start: Optional[float] = None
        self.request_start_time: Optional[float] = None
        self.last_llm_end_time: Optional[float] = None
        self.last_tool_end_time: Optional[float] = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when the LLM starts running."""
        current_time = time.time()
        
        if self.request_start_time is None:
            self.request_start_time = current_time
        
        # Record processing time from last tool end (or request start) to LLM start
        # Processing time is the gap between tool completion and LLM processing start
        # This represents the time the system spends processing results, serialization, and overhead
        if self.last_tool_end_time is not None:
            # Time from last tool completion to this LLM start
            processing_time = current_time - self.last_tool_end_time
        elif self.last_llm_end_time is not None:
            # If no tool was called, measure from last LLM end (for first LLM call after another)
            processing_time = current_time - self.last_llm_end_time
        elif self.request_start_time is not None:
            # First LLM call - time from request start
            processing_time = current_time - self.request_start_time
        else:
            processing_time = 0.0
        
        self.current_llm_start = current_time
        
        self.llm_calls.append({
            "llm_start_time": current_time,
            "processing_time_before": processing_time,
        })
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when the LLM finishes running."""
        current_time = time.time()
        self.last_llm_end_time = current_time
        
        if self.llm_calls and self.current_llm_start:
            self.llm_calls[-1]["llm_end_time"] = current_time
            self.llm_calls[-1]["llm_duration"] = current_time - self.current_llm_start
            self.current_llm_start = None
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Called when a tool starts executing."""
        current_time = time.time()
        tool_name = serialized.get("name", "unknown_tool")
        
        self.current_tool_start = current_time
        self.current_tool_name = tool_name
        
        self.tool_calls.append({
            "tool_name": tool_name,
            "tool_start_time": current_time,
            "input": input_str[:200] if input_str else "",  # Truncate for storage
        })
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool finishes executing."""
        current_time = time.time()
        
        if self.tool_calls and self.current_tool_start:
            self.tool_calls[-1]["tool_end_time"] = current_time
            self.tool_calls[-1]["tool_round_trip_time"] = current_time - self.current_tool_start
            self.tool_calls[-1]["output_length"] = len(output) if output else 0
            self.last_tool_end_time = current_time  # Track when tool ended for processing time calculation
            self.current_tool_start = None
            self.current_tool_name = None
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when a tool execution encounters an error."""
        current_time = time.time()
        
        if self.tool_calls and self.current_tool_start:
            self.tool_calls[-1]["tool_end_time"] = current_time
            self.tool_calls[-1]["tool_round_trip_time"] = current_time - self.current_tool_start
            self.tool_calls[-1]["error"] = str(error)
            self.last_tool_end_time = current_time  # Track when tool ended for processing time calculation
            self.current_tool_start = None
            self.current_tool_name = None
    
    def get_tool_round_trip_times(self) -> List[float]:
        """Get list of all tool round-trip times."""
        return [
            call["tool_round_trip_time"]
            for call in self.tool_calls
            if "tool_round_trip_time" in call
        ]
    
    def get_llm_processing_times(self) -> List[float]:
        """Get list of all processing times (time before LLM calls, after tool completion)."""
        return [
            call["processing_time_before"]
            for call in self.llm_calls
            if "processing_time_before" in call
        ]
    
    def get_total_tool_time(self) -> float:
        """Get total time spent in tool execution."""
        return sum(self.get_tool_round_trip_times())
    
    def get_total_processing_time(self) -> float:
        """Get total time spent in processing overhead (between tool completion and LLM calls)."""
        return sum(self.get_llm_processing_times())
    
    def get_total_llm_time(self) -> float:
        """Get total time spent in LLM calls."""
        return sum(
            call.get("llm_duration", 0.0)
            for call in self.llm_calls
            if "llm_duration" in call
        )


class ToolRoundTripTimeBenchmarkAgent(object):
    """
    Agent for benchmarking Tool Round-Trip Time (TRTT) metric.
    
    Tool Round-Trip Time measures the latency specifically attributed to tool execution
    vs. the LLM "thinking" time. This is critical for understanding performance in
    ITBench scenarios where agents use tools (scripts, CLIs).
    """
    
    agent_goal: str = """Benchmark Tool Round-Trip Time (TRTT) for agent interactions with tools.
    This agent measures the latency specifically attributed to tool execution vs. processing overhead time."""
    
    tool_description: str = """This agent performs Tool Round-Trip Time benchmarking by:
    - Making agent calls that use tools
    - Measuring time from tool call start to tool return
    - Measuring processing overhead time (time between tool completion and LLM calls)
    - Running multiple iterations for statistical accuracy
    - Reporting metrics (mean, median, min, max, stddev) for both tool execution and processing time"""
    
    input_description: dict = {
        "prompt": "The prompt to send to the agent that will trigger tool usage",
        "num_iterations": "Number of iterations to run (default: 10)",
        "workdir": "Working directory to save results",
    }
    
    output_description: dict = {
        "trtt_results": "Dictionary containing Tool Round-Trip Time metrics and statistics",
        "path_to_benchmark_results": "Path to JSON file with detailed results",
    }
    
    workdir_root: str = "/tmp/agent/"
    
    def kickoff(self, inputs: dict):
        """Entry point for the Tool Round-Trip Time benchmark agent."""
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
                prompt = goal if goal and len(goal) < 200 else "List all pods in the default namespace using kubectl."
        
        # Extract iterations from goal if specified
        if "iterations:" in goal.lower() or "num_iterations:" in goal.lower():
            import re
            match = re.search(r'(?:iterations|num_iterations):\s*(\d+)', goal, re.IGNORECASE)
            if match:
                num_iterations = int(match.group(1))
        
        return self.run_benchmark(
            prompt=prompt or "List all pods in the default namespace using kubectl.",
            num_iterations=num_iterations,
            workdir=workdir,
        )
    
    def run_benchmark(
        self,
        prompt: str = "List all pods in the default namespace using kubectl.",
        num_iterations: int = 10,
        workdir: str = None,
        **kwargs
    ) -> dict:
        """
        Run Tool Round-Trip Time benchmark with multiple iterations.
        
        Args:
            prompt: The prompt to send to the agent that will trigger tool usage
            num_iterations: Number of benchmark iterations to run
            workdir: Working directory to save results
            **kwargs: Additional arguments (e.g., goal, kubeconfig, etc.)
        
        Returns:
            Dictionary containing benchmark results
        """
        if workdir is None:
            workdir = os.path.join(
                self.workdir_root,
                datetime.now().strftime("%Y%m%d%H%M%S_trtt_benchmark"),
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
        all_tool_times: List[float] = []
        all_processing_times: List[float] = []
        all_llm_times: List[float] = []
        detailed_results: List[Dict[str, Any]] = []
        
        for iteration in range(1, num_iterations + 1):
            print(f"Running iteration {iteration}/{num_iterations}...", end=" ", flush=True)
            
            callback = ToolRoundTripTimeCallback()
            
            try:
                # Prepare messages - use a prompt that will trigger tool usage
                # For this benchmark, we'll use a simple prompt that asks the LLM
                # to use a tool. Since we don't have direct tool access in this benchmark,
                # we'll simulate by making the LLM call and measuring what we can.
                # In a real scenario, this would be integrated with an agent framework.
                
                messages = [HumanMessage(content=prompt)]
                
                # Make call with callback to track tool usage
                start_time = time.time()
                try:
                    # Note: This is a simplified version. In a real agent framework,
                    # tools would be called through the agent, and we'd measure actual tool execution.
                    # For now, we'll measure LLM response time and note that tool calls
                    # would be measured separately in an integrated system.
                    response = llm.invoke(messages, callbacks=[callback])
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Get metrics from callback
                    tool_times = callback.get_tool_round_trip_times()
                    processing_times = callback.get_llm_processing_times()
                    llm_times = callback.get_total_llm_time()
                    
                    # If no tool calls were made, this iteration didn't use tools
                    # We'll still record it but note that it's a baseline measurement
                    if tool_times:
                        all_tool_times.extend(tool_times)
                        all_processing_times.extend(processing_times)
                        if llm_times > 0:
                            all_llm_times.append(llm_times)
                        
                        detailed_results.append({
                            "iteration": iteration,
                            "total_time_seconds": total_time,
                            "tool_round_trip_times": tool_times,
                            "num_tool_calls": len(tool_times),
                            "processing_times": processing_times,
                            "llm_time_seconds": llm_times,
                            "total_tool_time": callback.get_total_tool_time(),
                            "total_processing_time": callback.get_total_processing_time(),
                            "tool_calls_detail": callback.tool_calls,
                            "llm_calls_detail": callback.llm_calls,
                            "timestamp": datetime.now().isoformat(),
                        })
                        avg_trtt = statistics.mean(tool_times) if tool_times else 0
                        print(f"✓ Avg TRTT: {avg_trtt:.4f}s ({len(tool_times)} tool calls)")
                    else:
                        # No tools were called - this is expected in a simple LLM call
                        # In a real agent scenario, tools would be invoked
                        detailed_results.append({
                            "iteration": iteration,
                            "total_time_seconds": total_time,
                            "tool_round_trip_times": [],
                            "num_tool_calls": 0,
                            "note": "No tool calls detected. This may be expected for simple LLM calls.",
                            "timestamp": datetime.now().isoformat(),
                        })
                        print(f"⚠ No tool calls detected")
                
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    detailed_results.append({
                        "iteration": iteration,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    })
            
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                detailed_results.append({
                    "iteration": iteration,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
            
            # Small delay between iterations to avoid rate limiting
            if iteration < num_iterations:
                time.sleep(0.5)
        
        # Calculate statistics for tool round-trip times
        if all_tool_times:
            tool_stats = {
                "mean": statistics.mean(all_tool_times),
                "median": statistics.median(all_tool_times),
                "min": min(all_tool_times),
                "max": max(all_tool_times),
                "stddev": statistics.stdev(all_tool_times) if len(all_tool_times) > 1 else 0.0,
                "count": len(all_tool_times),
            }
        else:
            tool_stats = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "stddev": None,
                "count": 0,
            }
        
        # Calculate statistics for processing times
        if all_processing_times:
            processing_stats = {
                "mean": statistics.mean(all_processing_times),
                "median": statistics.median(all_processing_times),
                "min": min(all_processing_times),
                "max": max(all_processing_times),
                "stddev": statistics.stdev(all_processing_times) if len(all_processing_times) > 1 else 0.0,
                "count": len(all_processing_times),
            }
        else:
            processing_stats = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "stddev": None,
                "count": 0,
            }
        
        # Calculate statistics for LLM times
        if all_llm_times:
            llm_stats = {
                "mean": statistics.mean(all_llm_times),
                "median": statistics.median(all_llm_times),
                "min": min(all_llm_times),
                "max": max(all_llm_times),
                "stddev": statistics.stdev(all_llm_times) if len(all_llm_times) > 1 else 0.0,
                "count": len(all_llm_times),
            }
        else:
            llm_stats = {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "stddev": None,
                "count": 0,
            }
        
        # Prepare results
        results = {
            "metric": "Tool Round-Trip Time (TRTT)",
            "model": model,
            "api_url": api_url if api_url else "default",
            "prompt": prompt,
            "num_iterations": num_iterations,
            "successful_iterations": len([r for r in detailed_results if "error" not in r]),
            "iterations_with_tool_calls": len([r for r in detailed_results if r.get("num_tool_calls", 0) > 0]),
            "statistics": {
                "tool_round_trip_time": tool_stats,
                "processing_time": processing_stats,
                "llm_execution_time": llm_stats,
            },
            "detailed_results": detailed_results,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results to file
        results_file = os.path.join(workdir, "tool_round_trip_time_benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("TOOL ROUND-TRIP TIME BENCHMARK RESULTS")
        print("=" * 80)
        
        if tool_stats["count"] > 0:
            print(f"\n{'Tool Round-Trip Time Statistics:'}")
            print(f"{'  Successful Iterations':<30} {results['iterations_with_tool_calls']}/{num_iterations}")
            print(f"{'  Total Tool Calls':<30} {tool_stats['count']}")
            print(f"{'  Mean TRTT':<30} {tool_stats['mean']:.4f} seconds")
            print(f"{'  Median TRTT':<30} {tool_stats['median']:.4f} seconds")
            print(f"{'  Min TRTT':<30} {tool_stats['min']:.4f} seconds")
            print(f"{'  Max TRTT':<30} {tool_stats['max']:.4f} seconds")
            print(f"{'  Std Dev':<30} {tool_stats['stddev']:.4f} seconds")
            
            if processing_stats["count"] > 0:
                print(f"\n{'Processing Time Statistics:'}")
                print(f"{'  Mean Processing Time':<30} {processing_stats['mean']:.4f} seconds")
                print(f"{'  Median Processing Time':<30} {processing_stats['median']:.4f} seconds")
                print(f"{'  Min Processing Time':<30} {processing_stats['min']:.4f} seconds")
                print(f"{'  Max Processing Time':<30} {processing_stats['max']:.4f} seconds")
                print(f"{'  Std Dev':<30} {processing_stats['stddev']:.4f} seconds")
            
            if llm_stats["count"] > 0:
                print(f"\n{'LLM Execution Time Statistics:'}")
                print(f"{'  Mean LLM Time':<30} {llm_stats['mean']:.4f} seconds")
                print(f"{'  Median LLM Time':<30} {llm_stats['median']:.4f} seconds")
        else:
            print("No tool calls detected in any iteration.")
            print("Note: This benchmark requires an agent framework that uses tools.")
            print("      For a simple LLM call, no tools are invoked.")
        
        print(f"\n{'='*80}")
        print(f"Detailed results saved to: {results_file}")
        print(f"{'='*80}\n")
        
        return {
            "result": {
                "trtt_results": results,
                "path_to_benchmark_results": results_file,
            }
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tool Round-Trip Time Benchmark Agent")
    parser.add_argument(
        "--prompt",
        default="List all pods in the default namespace using kubectl.",
        help="Prompt to use for benchmarking (should trigger tool usage)"
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
    
    agent = ToolRoundTripTimeBenchmarkAgent()
    result = agent.run_benchmark(
        prompt=args.prompt,
        num_iterations=args.iterations,
        workdir=args.workdir,
    )
    
    print("\nFinal Result:")
    print(json.dumps(result, indent=2))

