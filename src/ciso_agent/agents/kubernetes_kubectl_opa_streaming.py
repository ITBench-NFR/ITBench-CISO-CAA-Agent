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
import string
import time

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langfuse import get_client
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

from ciso_agent.tracing import extract_metrics_from_trace
from ciso_agent.streaming_llm import get_llm_params
from ciso_agent.streaming_utils import StreamingMetricsCallback, StreamingLLMWrapper
from ciso_agent.llm import extract_code
from ciso_agent.tools.generate_opa_rego import GenerateOPARegoTool
from ciso_agent.tools.run_opa_rego import RunOPARegoTool
from ciso_agent.tools.run_kubectl import RunKubectlTool
from ciso_agent.vllm_metrics import VLLMMetricsCollector, is_vllm_metrics_enabled

# Load .env file - check Docker mount path first, then current directory
docker_env_path = "/etc/ciso-agent/.env"
if os.path.exists(docker_env_path):
    load_dotenv(docker_env_path)
else:
    load_dotenv()

langfuse = get_client()


class KubernetesKubectlOPACrew(object):
    agent_goal: str = """I would like to check if the following condition is satisfiled, given a Kubernetes cluster with `kubeconfig.yaml`
    ${compliance}

To check the condition, do the following steps.
- get related resource(s) on the cluster by kubectl command and save it as `collected_data.json`
- chcek if the conditions are satisfied by OPA Rego policy

for those steps, you need to create an OPA Rego policy `policy.rego`.
Also, save the kubectl command as `script.sh`.

If running the policy failed and if you can fix the issue, you will do it and try running again.

Once you get a final answer, you can quit the work.
"""

    tool_description: str = """This agent has the following tools to use:
- RunOPARegoTool
- GenerateOPARegoTool
- RunKubectlTool
"""

    input_description: dict = {
        "compliance": "a short string of compliance requirement",
        "workdir": "a working directory to save temporary files",
    }

    output_description: dict = {
        "path_to_generated_shell_script": "a string of the filepath to the generated shell script",
        "path_to_generated_rego_policy": "a string of the filepath to the generated rego policy",
        "path_to_collected_data_by_script": "a string of the filepath to the collected data",
        "streaming_metrics": "a dict containing TTFT and TGS statistics from streaming LLM calls",
        "path_to_streaming_metrics": "a string of the filepath to the streaming metrics JSON file",
    }

    workdir_root: str = "/tmp/agent/"

    def kickoff(self, inputs: dict):
        print("\n" + "="*80)
        print("KUBERNETES_KUBECTL_OPA_STREAMING AGENT - STREAMING METRICS ENABLED")
        print("="*80 + "\n")
        CrewAIInstrumentor().instrument(skip_dep_check=True)
        LangChainInstrumentor().instrument(skip_dep_check=True)
        LiteLLMInstrumentor().instrument(skip_dep_check=True)

        # Initialize vLLM metrics collector (only if VLLM_PROMETHEUS_URL is set)
        workdir = inputs.get("workdir", self.workdir_root)
        metrics_collector = None
        
        if is_vllm_metrics_enabled():
            test_case_id = f"kubectl_opa_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        print("\n" + "="*80)
        print("KUBERNETES_KUBECTL_OPA_STREAMING.run_scenario() CALLED")
        print(f"Goal: {goal[:100]}...")
        print("="*80 + "\n")
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
        
        # #region agent log
        import json
        # Use workdir for debug logs so they're accessible in Docker
        # Also try a fallback location in case workdir isn't accessible
        log_path = os.path.join(workdir, "debug.log")
        fallback_log_path = "/tmp/agent/debug.log"  # Fallback for Docker
        print(f"[DEBUG] Log path will be: {log_path}")
        print(f"[DEBUG] Workdir: {workdir}")
        print(f"[DEBUG] Workdir exists: {os.path.exists(workdir)}")
        print(f"[DEBUG] This is kubernetes_kubectl_opa_streaming agent - streaming metrics collection enabled")
        try:
            # Ensure directory exists
            os.makedirs(workdir, exist_ok=True)
            log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "kubernetes_kubectl_opa_streaming.py:337", "message": "About to initialize streaming LLM", "data": {"model": model, "api_url": api_url, "has_api_key": bool(api_key), "workdir": workdir}, "timestamp": int(time.time() * 1000)}
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_data) + "\n")
                print(f"[DEBUG] Successfully wrote to log: {log_path}")
            except Exception as e1:
                print(f"[DEBUG] Failed to write to primary log path: {e1}")
                # Try fallback
                try:
                    os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)
                    with open(fallback_log_path, "a") as f:
                        f.write(json.dumps(log_data) + "\n")
                    print(f"[DEBUG] Wrote to fallback log: {fallback_log_path}")
                    log_path = fallback_log_path  # Update log_path for rest of function
                except Exception as e2:
                    print(f"[DEBUG] Failed to write to fallback log: {e2}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[DEBUG] Failed to write log: {e}")
            import traceback
            traceback.print_exc()
        # #endregion
        
        # CrewAI/LiteLLM requires OPENAI_API_KEY environment variable when using ChatOpenAI
        # Set it if api_key is available and OPENAI_API_KEY is not already set
        if api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Create callback handler for tracking streaming metrics
        streaming_metrics_callback = StreamingMetricsCallback(debug=True)
        
        # Use CrewAI's LLM wrapper (like init_agent_llm does) but intercept LiteLLM calls
        # CrewAI wraps LangChain models in its own LLM class which uses LiteLLM
        # We need to intercept at the LiteLLM level to enable streaming
        from crewai import LLM
        
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        
        # Create CrewAI LLM wrapper - this is what CrewAI expects
        if "qwen" in model.lower():
            crewai_llm = LLM(
                model="hosted_vllm/" + model,
                base_url=api_url,
                api_key=api_key,
                temperature=temperature,
            )
        else:
            crewai_llm = LLM(
                model=model,
                base_url=api_url,
                api_key=api_key,
                temperature=temperature,
            )
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "kubernetes_kubectl_opa_streaming.py:365", "message": "CrewAI LLM initialized", "data": {"llm_type": type(crewai_llm).__name__, "llm_module": type(crewai_llm).__module__}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception as e:
            print(f"[DEBUG] Failed to write log: {e}")
        # #endregion
        
        print(f"[DEBUG] Initialized CrewAI LLM: {type(crewai_llm).__name__}")
        
        # Intercept LiteLLM's completion() method to enable streaming and track metrics
        # CrewAI's LLM uses LiteLLM internally, so we need to patch LiteLLM
        try:
            import litellm
            print(f"[DEBUG] LiteLLM imported successfully: {litellm.__version__ if hasattr(litellm, '__version__') else 'unknown'}")
            original_completion = litellm.completion
            print(f"[DEBUG] Original litellm.completion: {original_completion}")
            
            def patched_completion(*args, **kwargs):
                """Patch LiteLLM completion to enable streaming, track metrics, and return dict format."""
                # #region agent log
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "kubernetes_kubectl_opa_streaming.py:397", "message": "litellm.completion() called", "data": {"stream_param": kwargs.get("stream", False), "model": kwargs.get("model"), "args_count": len(args)}, "timestamp": int(time.time() * 1000)}) + "\n")
                except Exception as e:
                    print(f"[DEBUG] Failed to log litellm.completion call: {e}")
                # #endregion
                
                print(f"[DEBUG] litellm.completion() called with stream={kwargs.get('stream', False)}")
                
                # Check if caller wants streaming - if not, we'll stream internally but return dict
                original_stream_param = kwargs.get("stream", False)
                
                # Force streaming to be enabled for metrics tracking
                kwargs["stream"] = True
                print(f"[DEBUG] Forced stream=True (was {original_stream_param})")
                
                # Call original completion with streaming enabled
                try:
                    response_stream = original_completion(*args, **kwargs)
                    print(f"[DEBUG] Original completion returned: {type(response_stream)}")
                except Exception as e:
                    print(f"[DEBUG] Error calling original_completion: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Track metrics from the stream and accumulate response
                request_start = time.time()
                first_token_time = None
                last_token_time = None
                token_count = 0
                accumulated_content = ""
                accumulated_chunks = []
                
                # Consume the stream and track metrics
                print(f"[DEBUG] Starting to consume stream for metrics tracking")
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "kubernetes_kubectl_opa_streaming.py:432", "message": "Starting to consume stream", "data": {}, "timestamp": int(time.time() * 1000)}) + "\n")
                except Exception:
                    pass
                
                for chunk in response_stream:
                    current_time = time.time()
                    
                    # Debug chunk structure (only first few chunks to avoid spam)
                    if token_count < 3:
                        chunk_attrs = [attr for attr in dir(chunk) if not attr.startswith('_')]
                        chunk_dict = {}
                        for attr in ['choices', 'content', 'delta', 'message']:
                            if hasattr(chunk, attr):
                                try:
                                    val = getattr(chunk, attr)
                                    chunk_dict[attr] = str(type(val)) if val is not None else None
                                except:
                                    chunk_dict[attr] = "error_accessing"
                        print(f"[DEBUG] Chunk {token_count} structure: {chunk_dict}")
                        # #region agent log
                        try:
                            with open(log_path, "a") as f:
                                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "kubernetes_kubectl_opa_streaming.py:470", "message": "Chunk structure inspection", "data": {"chunk_num": token_count, "chunk_attrs": chunk_attrs[:10], "chunk_dict": chunk_dict}, "timestamp": int(time.time() * 1000)}) + "\n")
                        except Exception:
                            pass
                        # #endregion
                    
                    # Track first token (TTFT)
                    if first_token_time is None:
                        first_token_time = current_time
                        ttft = first_token_time - request_start
                        streaming_metrics_callback.all_ttft_values.append(ttft)
                        # #region agent log
                        try:
                            with open(log_path, "a") as f:
                                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "kubernetes_kubectl_opa_streaming.py:490", "message": "First token received via litellm stream", "data": {"ttft": ttft, "token_count": token_count}, "timestamp": int(time.time() * 1000)}) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        if streaming_metrics_callback.debug:
                            print(f"[DEBUG] First token received via litellm stream! TTFT: {ttft:.4f}s")
                    
                    last_token_time = current_time
                    token_count += 1
                    
                    # Accumulate chunks for final response
                    accumulated_chunks.append(chunk)
                    
                    # Extract content from chunk - LiteLLM ModelResponseStream structure
                    chunk_content = None
                    try:
                        # Try multiple extraction methods based on LiteLLM's structure
                        if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                            choice = chunk.choices[0]
                            # Check for delta.content (streaming format)
                            if hasattr(choice, 'delta') and choice.delta:
                                if hasattr(choice.delta, 'content') and choice.delta.content:
                                    chunk_content = choice.delta.content
                            # Check for message.content (final format)
                            elif hasattr(choice, 'message') and choice.message:
                                if hasattr(choice.message, 'content') and choice.message.content:
                                    chunk_content = choice.message.content
                            # Check for direct content
                            elif hasattr(choice, 'content') and choice.content:
                                chunk_content = choice.content
                        # Direct content attribute
                        elif hasattr(chunk, 'content') and chunk.content:
                            chunk_content = chunk.content
                        
                        if chunk_content:
                            accumulated_content += chunk_content
                    except Exception as e:
                        # Log extraction errors but continue
                        if token_count <= 3:
                            print(f"[DEBUG] Content extraction error: {e}")
                            # #region agent log
                            try:
                                with open(log_path, "a") as f:
                                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "kubernetes_kubectl_opa_streaming.py:530", "message": "Content extraction error", "data": {"error": str(e), "chunk_type": str(type(chunk))}, "timestamp": int(time.time() * 1000)}) + "\n")
                            except Exception:
                                pass
                            # #endregion
                
                # Calculate TGS after streaming completes
                if first_token_time and last_token_time and token_count >= 2:
                    generation_time = last_token_time - first_token_time
                    if generation_time > 0:
                        tokens_generated = token_count - 1
                        tokens_per_second = tokens_generated / generation_time
                        streaming_metrics_callback.all_tgs_values.append(tokens_per_second)
                        # #region agent log
                        try:
                            with open(log_path, "a") as f:
                                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "kubernetes_kubectl_opa_streaming.py:575", "message": "TGS calculated via litellm stream", "data": {"tokens_per_second": tokens_per_second, "token_count": token_count}, "timestamp": int(time.time() * 1000)}) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        if streaming_metrics_callback.debug:
                            print(f"[DEBUG] TGS calculated via litellm stream: {tokens_per_second:.2f} tokens/sec ({token_count} chunks)")
                
                # CrewAI expects a dictionary response, not a generator
                # Build response in the format CrewAI expects: {"choices": [{"message": {"content": "..."}}]}
                print(f"[DEBUG] Building dict response from {len(accumulated_chunks)} chunks, content length: {len(accumulated_content)}")
                response_dict = {
                    "choices": [{
                        "message": {
                            "content": accumulated_content
                        }
                    }]
                }
                
                # Also preserve other fields from the last chunk if available
                if accumulated_chunks:
                    last_chunk = accumulated_chunks[-1]
                    if hasattr(last_chunk, 'model'):
                        response_dict["model"] = last_chunk.model
                    if hasattr(last_chunk, 'usage'):
                        response_dict["usage"] = last_chunk.usage
                    if hasattr(last_chunk, 'id'):
                        response_dict["id"] = last_chunk.id
                
                print(f"[DEBUG] Returning dict response with content length: {len(response_dict.get('choices', [{}])[0].get('message', {}).get('content', ''))}")
                return response_dict
            
            # Patch litellm.completion
            litellm.completion = patched_completion
            
            # Verify patch was applied
            print(f"[DEBUG] Patched litellm.completion: {litellm.completion}")
            print(f"[DEBUG] Patch is same function: {litellm.completion == patched_completion}")
            
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "kubernetes_kubectl_opa_streaming.py:575", "message": "Patched litellm.completion to enable streaming", "data": {"patch_applied": litellm.completion == patched_completion}, "timestamp": int(time.time() * 1000)}) + "\n")
            except Exception as e:
                print(f"[DEBUG] Failed to log patch confirmation: {e}")
            # #endregion
            
            print(f"[DEBUG] Patched litellm.completion to enable streaming and track metrics")
            
        except ImportError:
            print(f"[WARN] litellm not available, cannot patch for streaming")
        except Exception as e:
            print(f"[WARN] Failed to patch litellm.completion: {e}")
        
        # Use CrewAI's LLM wrapper directly - LiteLLM patching will handle streaming
        llm = crewai_llm
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "kubernetes_kubectl_opa_streaming.py:680", "message": "Using CrewAI LLM wrapper with LiteLLM patching", "data": {"llm_type": type(llm).__name__}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            pass
        # #endregion
        
        test_agent = Agent(
            role="Test",
            goal=goal,
            backstory="",
            llm=llm,
            verbose=True,
            callbacks=[streaming_metrics_callback],  # Still pass callbacks in case they work
        )
        
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "kubernetes_kubectl_opa_streaming.py:648", "message": "Agent created", "data": {"agent_llm_type": type(test_agent.llm).__name__ if hasattr(test_agent, 'llm') else None, "agent_llm_module": type(test_agent.llm).__module__ if hasattr(test_agent, 'llm') else None, "agent_llm_is_same": test_agent.llm is llm if hasattr(test_agent, 'llm') else None, "has_callbacks": hasattr(test_agent, 'callbacks'), "callback_count": len(test_agent.callbacks) if hasattr(test_agent, 'callbacks') else 0}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        
        print(f"[DEBUG] Agent created with callbacks: {len(test_agent.callbacks) if hasattr(test_agent, 'callbacks') else 'N/A'}")

        target_task = Task(
            name="target_task",
            description="""Check a rego policy for a given input file. If policy or input file are not ready, prepare them first.""",
            expected_output="""A boolean which indicates if the check is passed or not""",
            agent=test_agent,
            tools=[
                RunOPARegoTool(workdir=workdir),
                GenerateOPARegoTool(workdir=workdir),
                RunKubectlTool(workdir=workdir, read_only=True),
            ],
        )
        report_task = Task(
            name="report_task",
            description="""Report filepaths that are created in the previous task.
You must not replay the steps in the privious task such as generating code / running something.
Just to report the result.
""",
            expected_output="""A JSON string with the following info:
```json
{
    "path_to_generated_shell_script": <PLACEHOLDER>,
    "path_to_generated_rego_policy": <PLACEHOLDER>,
    "path_to_collected_data_by_script": <PLACEHOLDER>,
}
```
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
        
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "kubernetes_kubectl_opa_streaming.py:698", "message": "About to call crew.kickoff", "data": {"crew_agents_count": len(crew.agents) if hasattr(crew, 'agents') else 0, "crew_tasks_count": len(crew.tasks) if hasattr(crew, 'tasks') else 0}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        
        output = crew.kickoff(inputs=inputs)
        
        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "kubernetes_kubectl_opa_streaming.py:702", "message": "crew.kickoff completed", "data": {"ttft_count": len(streaming_metrics_callback.all_ttft_values), "tgs_count": len(streaming_metrics_callback.all_tgs_values)}, "timestamp": int(time.time() * 1000)}) + "\n")
        # #endregion
        result_str = output.raw
        if "```" in result_str:
            result_str = extract_code(result_str, code_type="json")

        result = None
        try:
            result = json.loads(result_str)
        except Exception:
            raise ValueError(f"Failed to parse this as JSON: {result_str}")

        # add workdir prefix here because agent does not know it
        for key, val in result.items():
            if val and key.startswith("path_to_") and "/" not in val:
                result[key] = os.path.join(workdir, val)

        # for eval, copy the generated files to some fixed filepath (filename)
        copy_files_for_eval(result)

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


def copy_files_for_eval(result: dict):
    script_path = result.get("path_to_generated_shell_script")
    if script_path and os.path.exists(script_path) and os.path.basename(script_path) != "fetcher.sh":
        dest_path = os.path.join(os.path.dirname(script_path), "fetcher.sh")
        shutil.copyfile(script_path, dest_path)

    policy_path = result.get("path_to_generated_rego_policy")
    if policy_path and os.path.exists(policy_path) and os.path.basename(policy_path) != "policy.rego":
        dest_path = os.path.join(os.path.dirname(policy_path), "policy.rego")
        shutil.copyfile(policy_path, dest_path)

    data_path = result.get("path_to_collected_data_by_script")
    if data_path and os.path.exists(data_path) and os.path.basename(data_path) != "collected_data.json":
        dest_path = os.path.join(os.path.dirname(data_path), "collected_data.json")
        shutil.copyfile(data_path, dest_path)
    return


if __name__ == "__main__":
    default_compliance = "Ensure that the cluster-admin role is only used where required"
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("-c", "--compliance", default=default_compliance, help="The compliance description for the agent to do something for")
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
        compliance=args.compliance,
        workdir=args.workdir,
    )
    _result = KubernetesKubectlOPACrew().kickoff(inputs=inputs)
    result = _result.get("result")

    result_json_str = json.dumps(result, indent=2)

    print("---- Result ----")
    print(result_json_str)
    print("----------------")

    if args.output:
        with open(args.output, "w") as f:
            f.write(result_json_str)

