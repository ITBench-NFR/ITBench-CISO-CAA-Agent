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

import argparse
import datetime
import json
import os
import shutil   
import sys
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from langfuse import get_client

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
    except ImportError:
        from langchain.callbacks import BaseCallbackHandler

from ciso_agent.tracing import extract_metrics_from_trace
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

from ciso_agent.streaming_llm import init_streaming_llm, get_llm_params
from ciso_agent.llm import extract_code
from ciso_agent.tools.generate_kyverno import GenerateKyvernoTool
from ciso_agent.tools.run_kubectl import RunKubectlTool
from ciso_agent.tracing import extract_metrics_from_trace
from ciso_agent.vllm_metrics import VLLMMetricsCollector, is_vllm_metrics_enabled

# Load .env file - check Docker mount path first, then current directory
docker_env_path = "/etc/ciso-agent/.env"
if os.path.exists(docker_env_path):
    load_dotenv(docker_env_path)
else:
    load_dotenv()
langfuse = get_client()


class StreamingMetricsCallback(BaseCallbackHandler):
    """Callback handler to track TTFT and Token Generation Speed during streaming."""
    
    def __init__(self, debug: bool = True):
        super().__init__()
        self.all_ttft_values: List[float] = []
        self.all_tgs_values: List[float] = []
        self.current_request_start_time = None
        self.current_first_token_time = None
        self.current_last_token_time = None
        self.current_token_count = 0
        self.current_tokens_received = False
        self.debug = debug
        self.callback_invocations = {
            "on_llm_start": 0,
            "on_llm_new_token": 0,
            "on_llm_end": 0,
        }
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when the LLM starts running."""
        self.callback_invocations["on_llm_start"] += 1
        if self.debug:
            print(f"[DEBUG] StreamingMetricsCallback.on_llm_start called (count: {self.callback_invocations['on_llm_start']})")
        
        self.current_request_start_time = time.time()
        self.current_first_token_time = None
        self.current_last_token_time = None
        self.current_token_count = 0
        self.current_tokens_received = False
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated."""
        self.callback_invocations["on_llm_new_token"] += 1
        current_time = time.time()
        
        if not self.current_tokens_received:
            # First token received - calculate TTFT
            self.current_first_token_time = current_time
            if self.current_request_start_time is not None:
                ttft = current_time - self.current_request_start_time
                self.all_ttft_values.append(ttft)
                if self.debug:
                    print(f"[DEBUG] First token received! TTFT: {ttft:.4f}s")
            self.current_tokens_received = True
        
        # Update last token time for each token
        self.current_last_token_time = current_time
        self.current_token_count += 1
        
        if self.debug and self.current_token_count <= 3:
            print(f"[DEBUG] Token {self.current_token_count} received: {repr(token[:20])}")
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Called when the LLM finishes running."""
        self.callback_invocations["on_llm_end"] += 1
        if self.debug:
            print(f"[DEBUG] StreamingMetricsCallback.on_llm_end called (count: {self.callback_invocations['on_llm_end']}, tokens: {self.current_token_count})")
        
        # Calculate Token Generation Speed
        if (self.current_first_token_time is not None and 
            self.current_last_token_time is not None and 
            self.current_token_count >= 2):
            
            generation_time = self.current_last_token_time - self.current_first_token_time
            if generation_time > 0:
                # Tokens generated after the first one
                tokens_generated = self.current_token_count - 1
                tokens_per_second = tokens_generated / generation_time
                self.all_tgs_values.append(tokens_per_second)
                if self.debug:
                    print(f"[DEBUG] TGS calculated: {tokens_per_second:.2f} tokens/sec ({tokens_generated} tokens in {generation_time:.4f}s)")
    
    def get_ttft_stats(self) -> Dict[str, Any]:
        """Get TTFT statistics."""
        if not self.all_ttft_values:
            return {"mean": None, "min": None, "max": None, "count": 0}
        
        import statistics
        return {
            "mean": statistics.mean(self.all_ttft_values) if len(self.all_ttft_values) > 0 else None,
            "min": min(self.all_ttft_values),
            "max": max(self.all_ttft_values),
            "count": len(self.all_ttft_values),
            "values": self.all_ttft_values,
        }
    
    def get_tgs_stats(self) -> Dict[str, Any]:
        """Get Token Generation Speed statistics."""
        if not self.all_tgs_values:
            return {"mean": None, "min": None, "max": None, "count": 0}
        
        import statistics
        return {
            "mean": statistics.mean(self.all_tgs_values) if len(self.all_tgs_values) > 0 else None,
            "min": min(self.all_tgs_values),
            "max": max(self.all_tgs_values),
            "count": len(self.all_tgs_values),
            "values": self.all_tgs_values,
        }


class StreamingLLMWrapper:
    """
    Wrapper around LangChain LLM to track streaming metrics by intercepting invoke/stream calls.
    
    This works around the issue where CrewAI/LiteLLM doesn't forward LangChain callbacks.
    We intercept the LLM calls directly to track TTFT and TGS metrics.
    """
    
    def __init__(self, llm, metrics_callback: StreamingMetricsCallback):
        self._llm = llm
        self.metrics_callback = metrics_callback
        # Preserve important attributes that CrewAI might check
        self.streaming = getattr(llm, 'streaming', False)
        self.model_name = getattr(llm, 'model_name', None)
        self.temperature = getattr(llm, 'temperature', None)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped LLM."""
        return getattr(self._llm, name)
    
    def invoke(self, input, config=None, **kwargs):
        """Intercept invoke calls to track metrics."""
        request_start = time.time()
        first_token_time = None
        last_token_time = None
        token_count = 0
        
        # Try to use streaming if available
        if hasattr(self._llm, 'stream') and self.metrics_callback.debug:
            print(f"[DEBUG] StreamingLLMWrapper: Using stream() to track metrics")
            try:
                # Use stream to get tokens one by one
                chunks = []
                for chunk in self._llm.stream(input, config=config, **kwargs):
                    chunks.append(chunk)
                    current_time = time.time()
                    
                    if first_token_time is None:
                        first_token_time = current_time
                        ttft = first_token_time - request_start
                        self.metrics_callback.all_ttft_values.append(ttft)
                        if self.metrics_callback.debug:
                            print(f"[DEBUG] First token received via stream! TTFT: {ttft:.4f}s")
                    
                    last_token_time = current_time
                    token_count += 1
                
                # Calculate TGS
                if first_token_time and last_token_time and token_count >= 2:
                    generation_time = last_token_time - first_token_time
                    if generation_time > 0:
                        tokens_generated = token_count - 1
                        tokens_per_second = tokens_generated / generation_time
                        self.metrics_callback.all_tgs_values.append(tokens_per_second)
                        if self.metrics_callback.debug:
                            print(f"[DEBUG] TGS calculated via stream: {tokens_per_second:.2f} tokens/sec")
                
                # Combine chunks into final response
                if chunks:
                    # Try to combine chunks - this depends on the chunk type
                    if hasattr(chunks[0], 'content'):
                        combined_content = ''.join(chunk.content for chunk in chunks if hasattr(chunk, 'content'))
                        # Create a response object similar to what invoke would return
                        final_chunk = chunks[-1]
                        if hasattr(final_chunk, 'content'):
                            final_chunk.content = combined_content
                        return final_chunk
                    return chunks[-1] if chunks else None
            except Exception as e:
                if self.metrics_callback.debug:
                    print(f"[DEBUG] Streaming failed, falling back to invoke: {e}")
        
        # Fallback to regular invoke
        response = self._llm.invoke(input, config=config, **kwargs)
        
        # If we couldn't track via streaming, we can't measure TTFT/TGS accurately
        # But we can at least record that a call was made
        if self.metrics_callback.debug:
            print(f"[DEBUG] StreamingLLMWrapper: Used invoke() (non-streaming), cannot track TTFT/TGS")
        
        return response
    
    def stream(self, input, config=None, **kwargs):
        """Intercept stream calls to track metrics."""
        request_start = time.time()
        first_token_time = None
        last_token_time = None
        token_count = 0
        
        if self.metrics_callback.debug:
            print(f"[DEBUG] StreamingLLMWrapper: Intercepting stream() call")
        
        for chunk in self._llm.stream(input, config=config, **kwargs):
            current_time = time.time()
            
            if first_token_time is None:
                first_token_time = current_time
                ttft = first_token_time - request_start
                self.metrics_callback.all_ttft_values.append(ttft)
                if self.metrics_callback.debug:
                    print(f"[DEBUG] First token received via stream! TTFT: {ttft:.4f}s")
            
            last_token_time = current_time
            token_count += 1
            
            yield chunk
        
        # Calculate TGS after streaming completes
        if first_token_time and last_token_time and token_count >= 2:
            generation_time = last_token_time - first_token_time
            if generation_time > 0:
                tokens_generated = token_count - 1
                tokens_per_second = tokens_generated / generation_time
                self.metrics_callback.all_tgs_values.append(tokens_per_second)
                if self.metrics_callback.debug:
                    print(f"[DEBUG] TGS calculated via stream: {tokens_per_second:.2f} tokens/sec ({tokens_generated} tokens in {generation_time:.4f}s)")
    
    def get_ttft_stats(self) -> Dict[str, Any]:
        """Get TTFT statistics."""
        if not self.all_ttft_values:
            return {"mean": None, "min": None, "max": None, "count": 0}
        
        import statistics
        return {
            "mean": statistics.mean(self.all_ttft_values) if len(self.all_ttft_values) > 0 else None,
            "min": min(self.all_ttft_values),
            "max": max(self.all_ttft_values),
            "count": len(self.all_ttft_values),
            "values": self.all_ttft_values,
        }
    
    def get_tgs_stats(self) -> Dict[str, Any]:
        """Get Token Generation Speed statistics."""
        if not self.all_tgs_values:
            return {"mean": None, "min": None, "max": None, "count": 0}
        
        import statistics
        return {
            "mean": statistics.mean(self.all_tgs_values) if len(self.all_tgs_values) > 0 else None,
            "min": min(self.all_tgs_values),
            "max": max(self.all_tgs_values),
            "count": len(self.all_tgs_values),
            "values": self.all_tgs_values,
        }

class KubernetesKyvernoUpdateCrew(object):
    agent_goal: str = """Currently, the following Kyverno policies are deployed in the Kubernetes cluster.
Review these existing policies and edit them to meet additional security requirements.
Please avoid creating new policy resources; instead, make the necessary changes directly to the existing resources.

${current_compliance}

However, the new internal security policy now requires the following additional conditions:

${updated_compliance}

Your task is to review the existing Kyverno policies and directly edit the existing policies to meet these new requirements.
Do not change the resource names. Once you have completed the edits, update the existing resources in the Kubernetes cluster.
Steps
- Get Kyverno policies and review them.
- Generate an updated Kyverno policy to meet the new requirements based on the current one. Ensure that you do not change the names of the resources.
- Apply the updated one to the cluster.

Once you get a final answer, you can quit the work.
"""
    tool_description: str = """This agent has the following tools to use:
- RunKubectlTool
- GenerateKyvernoTool
"""

    input_description: dict = {
        "current_compliance": "a short string of current compliance requirement",
        "updated_compliance": "a short string of updated compliance requirement in addition to the current one",
        "workdir": "a working directory to save temporary files",
    }

    output_description: dict = {
        "updated_resource": "a dict of Kubernetes metadata for the updated Kyverno policy",
        "path_to_generated_kyverno_policy": "a string of the filepath to the generated Kyverno policy YAML",
        "streaming_metrics": "a dict containing TTFT and TGS statistics from streaming LLM calls",
        "path_to_streaming_metrics": "a string of the filepath to the streaming metrics JSON file",
    }

    workdir_root: str = "/tmp/agent/"

    def kickoff(self, inputs: dict):
        CrewAIInstrumentor().instrument(skip_dep_check=True)
        LangChainInstrumentor().instrument(skip_dep_check=True)
        LiteLLMInstrumentor().instrument(skip_dep_check=True)

        # Initialize vLLM metrics collector (only if VLLM_PROMETHEUS_URL is set)
        workdir = inputs.get("workdir", self.workdir_root)
        metrics_collector = None
        
        if is_vllm_metrics_enabled():
            test_case_id = f"kyverno_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                metrics_collector = VLLMMetricsCollector()
                metrics_collector.start_collection(test_case_id)
            except Exception as e:
                print(f"[WARN] Failed to start vLLM metrics collection: {e}")
                metrics_collector = None

        # Store streaming_metrics_callback as instance variable so we can access it later
        return_value, streaming_metrics_callback = self.run_scenario(**inputs)

        # End vLLM metrics collection and save (only if enabled and started successfully)
        if metrics_collector is not None:
            try:
                metrics_collector.end_collection()
                traces_dir = os.path.join(workdir, "ciso_traces")
                metrics_collector.save_metrics(traces_dir)
            except Exception as e:
                print(f"[WARN] Failed to collect/save vLLM metrics: {e}")

        langfuse.flush()
        time.sleep(20)  # wait for trace to be available
        traces = langfuse.api.trace.list()
        if traces.data and len(traces.data) > 0:
            trace_detail = traces.data[0]  # Most recent trace
            all_observations = []
            page = 1
            while True:
                observations = langfuse.api.observations.get_many(trace_id=trace_detail.id, page=page, limit=50)
                if not observations.data:
                    break
                all_observations.extend(observations.data)
                if page >= observations.meta.total_pages:
                    break
                page += 1
                
            print(f"Total observations fetched: {len(all_observations)}")
            # Ensure directory exists before writing
            traces_dir = os.path.join(workdir, "ciso_traces")
            os.makedirs(traces_dir, exist_ok=True)
            extract_metrics_from_trace(all_observations)
            
            # Extract streaming metrics from Langfuse observations
            self._extract_streaming_metrics_from_langfuse(all_observations, streaming_metrics_callback, workdir)
        return return_value
    
    def _extract_streaming_metrics_from_langfuse(self, observations, metrics_callback, workdir):
        """
        Extract TTFT and TGS metrics from Langfuse observations.
        
        Langfuse tracks LLM calls and may have timing information we can use.
        """
        import statistics
        
        ttft_values = []
        tgs_values = []
        
        for obs in observations:
            try:
                # Check if this is an LLM generation observation
                obs_dict = obs.dict() if hasattr(obs, 'dict') else {}
                
                # Look for timing information in the observation
                # Langfuse may store this in different fields depending on the observation type
                if hasattr(obs, 'type') and obs.type == 'GENERATION':
                    # Try to extract timing from model_parameters, usage, or metadata
                    if hasattr(obs, 'model_parameters'):
                        # Some models provide timing info in parameters
                        pass
                    
                    # Check for start_time and end_time
                    start_time = None
                    end_time = None
                    first_token_time = None
                    
                    if hasattr(obs, 'start_time') and obs.start_time:
                        start_time = obs.start_time
                    if hasattr(obs, 'end_time') and obs.end_time:
                        end_time = obs.end_time
                    
                    # Try to get first token time from metadata or other fields
                    if hasattr(obs, 'metadata') and obs.metadata:
                        if isinstance(obs.metadata, dict):
                            first_token_time = obs.metadata.get('first_token_time')
                            if first_token_time and start_time:
                                # Calculate TTFT
                                if isinstance(start_time, str):
                                    from dateutil import parser
                                    start_time = parser.parse(start_time)
                                if isinstance(first_token_time, str):
                                    from dateutil import parser
                                    first_token_time = parser.parse(first_token_time)
                                
                                if hasattr(start_time, 'timestamp') and hasattr(first_token_time, 'timestamp'):
                                    ttft = (first_token_time.timestamp() - start_time.timestamp())
                                    ttft_values.append(ttft)
                                    metrics_callback.all_ttft_values.append(ttft)
                    
                    # Try to calculate TGS from token counts and timing
                    if hasattr(obs, 'usage') and obs.usage:
                        usage = obs.usage
                        if hasattr(usage, 'output_tokens') and usage.output_tokens:
                            output_tokens = usage.output_tokens
                            if start_time and end_time and first_token_time and output_tokens > 1:
                                # Calculate generation time (from first token to end)
                                if isinstance(start_time, str):
                                    from dateutil import parser
                                    start_time = parser.parse(start_time)
                                if isinstance(end_time, str):
                                    from dateutil import parser
                                    end_time = parser.parse(end_time)
                                if isinstance(first_token_time, str):
                                    from dateutil import parser
                                    first_token_time = parser.parse(first_token_time)
                                
                                if hasattr(first_token_time, 'timestamp') and hasattr(end_time, 'timestamp'):
                                    generation_time = end_time.timestamp() - first_token_time.timestamp()
                                    if generation_time > 0:
                                        # Tokens after the first one
                                        tokens_generated = output_tokens - 1
                                        tgs = tokens_generated / generation_time
                                        tgs_values.append(tgs)
                                        metrics_callback.all_tgs_values.append(tgs)
                
                # Alternative: Check for timing in the observation's data/response
                if hasattr(obs, 'output') and obs.output:
                    # Some observations might have timing in the output metadata
                    pass
                    
            except Exception as e:
                if metrics_callback.debug:
                    print(f"[DEBUG] Error extracting metrics from observation: {e}")
                continue
        
        if metrics_callback.debug:
            print(f"[DEBUG] Extracted {len(ttft_values)} TTFT values and {len(tgs_values)} TGS values from Langfuse observations")
        
        # Update the metrics callback with extracted values
        if ttft_values:
            metrics_callback.all_ttft_values.extend(ttft_values)
        if tgs_values:
            metrics_callback.all_tgs_values.extend(tgs_values)

    def run_scenario(self, goal: str, **kwargs):
        workdir = kwargs.get("workdir")
        if not workdir:
            workdir = os.path.join(self.workdir_root, datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S_"), "workspace")

        if not os.path.exists(workdir):
            os.makedirs(workdir, exist_ok=True)

        if "kubeconfig" in kwargs and kwargs["kubeconfig"]:
            kubeconfig = kwargs["kubeconfig"]
            dest = os.path.join(workdir, "kubeconfig.yaml")
            if kubeconfig != dest:
                shutil.copy(kubeconfig, dest)

        # Initialize streaming LLM
        model, api_url, api_key = get_llm_params()
        
        # CrewAI/LiteLLM requires OPENAI_API_KEY environment variable when using ChatOpenAI
        # Set it if api_key is available and OPENAI_API_KEY is not already set
        if api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Create callback handler for tracking streaming metrics
        streaming_metrics_callback = StreamingMetricsCallback(debug=True)
        
        llm = init_streaming_llm(model=model, api_url=api_url, api_key=api_key)
        
        print(f"[DEBUG] Initialized LLM: {type(llm).__name__}")
        print(f"[DEBUG] LLM streaming attribute: {getattr(llm, 'streaming', 'N/A')}")
        
        # Monkey-patch the LLM's stream and invoke methods to track metrics
        # This ensures we intercept calls even if CrewAI/LiteLLM wraps the LLM
        original_stream = llm.stream
        original_invoke = llm.invoke
        
        def patched_stream(input, config=None, **kwargs):
            """Patched stream method that tracks metrics."""
            if streaming_metrics_callback.debug:
                print(f"[DEBUG] Patched stream() called on {type(llm).__name__}")
            request_start = time.time()
            first_token_time = None
            last_token_time = None
            token_count = 0
            
            for chunk in original_stream(input, config=config, **kwargs):
                current_time = time.time()
                
                if first_token_time is None:
                    first_token_time = current_time
                    ttft = first_token_time - request_start
                    streaming_metrics_callback.all_ttft_values.append(ttft)
                    if streaming_metrics_callback.debug:
                        print(f"[DEBUG] First token received via patched stream! TTFT: {ttft:.4f}s")
                
                last_token_time = current_time
                token_count += 1
                yield chunk
            
            # Calculate TGS after streaming completes
            if first_token_time and last_token_time and token_count >= 2:
                generation_time = last_token_time - first_token_time
                if generation_time > 0:
                    tokens_generated = token_count - 1
                    tokens_per_second = tokens_generated / generation_time
                    streaming_metrics_callback.all_tgs_values.append(tokens_per_second)
                    if streaming_metrics_callback.debug:
                        print(f"[DEBUG] TGS calculated via patched stream: {tokens_per_second:.2f} tokens/sec")
        
        def patched_invoke(input, config=None, **kwargs):
            """Patched invoke method that tries to use streaming for metrics."""
            if streaming_metrics_callback.debug:
                print(f"[DEBUG] Patched invoke() called on {type(llm).__name__}")
            
            # Try to use streaming if available to track metrics
            if hasattr(llm, 'stream') and original_stream:
                try:
                    request_start = time.time()
                    first_token_time = None
                    last_token_time = None
                    token_count = 0
                    chunks = []
                    
                    for chunk in original_stream(input, config=config, **kwargs):
                        chunks.append(chunk)
                        current_time = time.time()
                        
                        if first_token_time is None:
                            first_token_time = current_time
                            ttft = first_token_time - request_start
                            streaming_metrics_callback.all_ttft_values.append(ttft)
                            if streaming_metrics_callback.debug:
                                print(f"[DEBUG] First token received via patched invoke->stream! TTFT: {ttft:.4f}s")
                        
                        last_token_time = current_time
                        token_count += 1
                    
                    # Calculate TGS
                    if first_token_time and last_token_time and token_count >= 2:
                        generation_time = last_token_time - first_token_time
                        if generation_time > 0:
                            tokens_generated = token_count - 1
                            tokens_per_second = tokens_generated / generation_time
                            streaming_metrics_callback.all_tgs_values.append(tokens_per_second)
                            if streaming_metrics_callback.debug:
                                print(f"[DEBUG] TGS calculated via patched invoke->stream: {tokens_per_second:.2f} tokens/sec")
                    
                    # Combine chunks into response
                    if chunks:
                        if hasattr(chunks[0], 'content'):
                            combined_content = ''.join(chunk.content for chunk in chunks if hasattr(chunk, 'content'))
                            final_chunk = chunks[-1]
                            if hasattr(final_chunk, 'content'):
                                final_chunk.content = combined_content
                            return final_chunk
                        return chunks[-1] if chunks else None
                except Exception as e:
                    if streaming_metrics_callback.debug:
                        print(f"[DEBUG] Streaming in invoke failed, falling back: {e}")
            
            # Fallback to regular invoke
            return original_invoke(input, config=config, **kwargs)
        
        # Apply the monkey patches using object.__setattr__ to bypass Pydantic validation
        # Pydantic models don't allow setting arbitrary attributes, so we need to bypass it
        object.__setattr__(llm, 'stream', patched_stream)
        object.__setattr__(llm, 'invoke', patched_invoke)
        print(f"[DEBUG] Monkey-patched LLM stream() and invoke() methods to track metrics")
        
        # Also try to set callbacks directly (in case they work)
        try:
            from langchain_core.callbacks import CallbackManager
            
            callback_manager = CallbackManager([streaming_metrics_callback])
            
            if hasattr(llm, 'callbacks'):
                llm.callbacks = callback_manager
                print(f"[DEBUG] Set callbacks directly on LLM via callbacks attribute")
            elif hasattr(llm, 'callback_manager'):
                llm.callback_manager = callback_manager
                print(f"[DEBUG] Set callbacks directly on LLM via callback_manager attribute")
        except Exception as e:
            print(f"[DEBUG] Failed to set callbacks on LLM: {e}")
        
        test_agent = Agent(
            role="Test",
            goal=goal,
            backstory="",
            llm=llm,
            verbose=True,
            callbacks=[streaming_metrics_callback],  # Still pass callbacks in case they work
        )
        
        print(f"[DEBUG] Agent created with callbacks: {len(test_agent.callbacks) if hasattr(test_agent, 'callbacks') else 'N/A'}")

        target_task = Task(
            name="target_task",
            description="""Get the Kyverno policy and generate the updated one based on it. Then deploy it on the cluster.""",
            expected_output="""A boolean which indicates if the result is OK or not""",
            agent=test_agent,
            tools=[
                RunKubectlTool(workdir=workdir, read_only=False),
                GenerateKyvernoTool(workdir=workdir),
            ],
        )
        report_task = Task(
            name="report_task",
            description="""Report a filepath that was created in the previous task.
You must not replay the steps in the privious task such as generating code / running something.
Just to report the result.
""",
            expected_output="""A JSON string with the following info:
```json
{
    "updated_resource": {
        "namespace": <PLACEHOLDER>,
        "kind": <PLACEHOLDER>,
        "name": <PLACEHOLDER>
    },
    "path_to_generated_kyverno_policy": <PLACEHOLDER>,
}
```
You can omit `namespace` in `updated_resource` if the policy is a cluster-scope resource.
""",
            context=[target_task],
            agent=test_agent,
        )

        crew = Crew(
            name="CISOCrew",
            tasks=[
                target_task,
                report_task,
            ],
            agents=[
                test_agent,
            ],
            process=Process.sequential,
            verbose=True,
            cache=False,
        )
        inputs = {}
        output = crew.kickoff(inputs=inputs)
        result_str = output.raw.strip()
        if not result_str:
            raise ValueError("crew agent returned an empty string.")

        if "```" in result_str:
            result_str = extract_code(result_str, code_type="json")
        result_str = result_str.strip()

        if not result_str:
            raise ValueError(f"crew agent returned an invalid string. This is the actual output: {output.raw}")

        result = {}
        try:
            result = json.loads(result_str)
        except Exception:
            print(f"Failed to parse this as JSON: {result_str}", file=sys.stderr)

        # add workdir prefix here because agent does not know it
        for key, val in result.items():
            if val and key.startswith("path_to_") and "/" not in val:
                result[key] = os.path.join(workdir, val)

        # Collect streaming metrics
        ttft_stats = streaming_metrics_callback.get_ttft_stats()
        tgs_stats = streaming_metrics_callback.get_tgs_stats()
        
        # Print callback invocation debug info
        print("\n" + "=" * 80)
        print("CALLBACK INVOCATION DEBUG")
        print("=" * 80)
        print(f"on_llm_start called: {streaming_metrics_callback.callback_invocations['on_llm_start']} times")
        print(f"on_llm_new_token called: {streaming_metrics_callback.callback_invocations['on_llm_new_token']} times")
        print(f"on_llm_end called: {streaming_metrics_callback.callback_invocations['on_llm_end']} times")
        print("=" * 80 + "\n")
        
        # Save streaming metrics to file
        metrics_file = os.path.join(workdir, "streaming_metrics.json")
        metrics_data = {
            "ttft_stats": ttft_stats,
            "tgs_stats": tgs_stats,
            "callback_invocations": streaming_metrics_callback.callback_invocations,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        # Print streaming metrics summary
        print("\n" + "=" * 80)
        print("STREAMING METRICS SUMMARY")
        print("=" * 80)
        
        if ttft_stats["count"] > 0:
            print(f"{'TTFT Mean':<25} {ttft_stats['mean']:.4f} seconds")
            print(f"{'TTFT Min':<25} {ttft_stats['min']:.4f} seconds")
            print(f"{'TTFT Max':<25} {ttft_stats['max']:.4f} seconds")
            print(f"{'TTFT Count':<25} {ttft_stats['count']}")
        else:
            print("No TTFT metrics collected")
        
        if tgs_stats["count"] > 0:
            print(f"{'TGS Mean':<25} {tgs_stats['mean']:.2f} tokens/sec")
            print(f"{'TGS Min':<25} {tgs_stats['min']:.2f} tokens/sec")
            print(f"{'TGS Max':<25} {tgs_stats['max']:.2f} tokens/sec")
            print(f"{'TGS Count':<25} {tgs_stats['count']}")
        else:
            print("No TGS metrics collected")
        
        print(f"\n{'='*80}")
        print(f"Streaming metrics saved to: {metrics_file}")
        print(f"{'='*80}\n")
        
        # Add metrics to result
        result["streaming_metrics"] = metrics_data
        result["path_to_streaming_metrics"] = metrics_file

        return {"result": result}, streaming_metrics_callback


if __name__ == "__main__":
    default_compliance = "Ensure that the cluster-admin role is only used where required"
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("-c", "--current-compliance", default=default_compliance, help="The compliance description for the agent to do something for")
    parser.add_argument(
        "-u", "--updated-compliance", default=default_compliance, help="The additional compliance description for the agent to do something for"
    )
    parser.add_argument("-k", "--kubeconfig", required=True, help="The path to the kubeconfig file")
    parser.add_argument("-w", "--workdir", default="", help="The path to the work dir which the agent will use")
    parser.add_argument("-o", "--output", help="The path to the output JSON file")
    args = parser.parse_args()

    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)

    if args.kubeconfig:
        dest_path = os.path.join(args.workdir, "kubeconfig.yaml")
        shutil.copyfile(args.kubeconfig, dest_path)

    inputs = dict(
        current_compliance=args.current_compliance,
        updated_compliance=args.updated_compliance,
        workdir=args.workdir,
    )
    _result = KubernetesKyvernoUpdateCrew().kickoff(inputs=inputs)
    result = _result.get("result")

    result_json_str = json.dumps(result, indent=2)

    print("---- Result ----")
    print(result_json_str)
    print("----------------")

    if args.output:
        with open(args.output, "w") as f:
            f.write(result_json_str)
