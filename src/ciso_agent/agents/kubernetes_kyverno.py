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


import os
from dotenv import load_dotenv

from langfuse import Langfuse, observe, get_client
import time
from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails
from langfuse.api.resources.observations.types.observations_views import ObservationsViews
import typing
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from collections import defaultdict

load_dotenv()

# if os.getenv("LANGTRACE_API_KEY"):
# langtrace.init(
#     # api_host=os.getenv("LANGTRACE_API_HOST"),
#     api_key=os.getenv("LANGTRACE_API_KEY"),
#     batch=False
# )
import argparse
import datetime
import json
import shutil
import string
import sys


from crewai import Agent, Crew, Process, Task

from ciso_agent.llm import init_agent_llm, extract_code
from ciso_agent.tools.generate_kyverno import GenerateKyvernoTool
from ciso_agent.tools.run_kubectl import RunKubectlTool

langfuse = get_client()

# langfuse = get_client()
# openlit.init(
#     otlp_endpoint="http://127.0.0.1:4318"
# )


class KubernetesKyvernoCrew(object):
    agent_goal: str = """I would like to check if the following condition is satisfiled, given a Kubernetes cluster with `kubeconfig.yaml`
    ${compliance}

To check the condition, do the following steps.
- deploy a Kyverno policy to the cluster
- chcek if the policy is correctly deployed.

If deploying the policy failed and if you can fix the issue, you will do it and try deploying again.
Once you get a final answer, you can quit the work.
"""
    tool_description: str = """This agent has the following tools to use:
- RunKubectlTool
- GenerateKyvernoTool
"""

    input_description: dict = {
        "compliance": "a short string of compliance requirement",
    }

    output_description: dict = {
        "deployed_resource": "a dict of Kubernetes metadata for the deployed Kyverno policy",
        "path_to_generated_kyverno_policy": "a string of the filepath to the generated Kyverno policy YAML",
    }

    workdir_root: str = "/tmp/agent/"

    # @observe()
    def kickoff(self, inputs: dict):
        # langfuse = get_client()
        # start_time = time.time()
        CrewAIInstrumentor().instrument(skip_dep_check=True)
        LangChainInstrumentor().instrument(skip_dep_check=True)
        LiteLLMInstrumentor().instrument(skip_dep_check=True)
        return_value = self.run_scenario(**inputs)
        # end_time = time.time()
        langfuse.flush()
        time.sleep(20)  # wait for trace to be available
        traces = langfuse.api.trace.list()
        if traces.data and len(traces.data) > 0:
            trace_detail = traces.data[0]  # Most recent trace
            # trace_id = trace.id

            # # Fetch full trace details
            # trace_detail = langfuse.api.trace.get(trace_id)

            # Extract metrics
            observations = langfuse.api.observations.get_many(trace_id=trace_detail.id)
            print("Observations page data:")
            print(observations.meta)
            self._extract_metrics_from_trace(observations)
            # print(metrics)

        return return_value

    # @observe()
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

        llm = init_agent_llm()
        test_agent = Agent(
            role="Test",
            goal=goal,
            backstory="",
            llm=llm,
            verbose=True,
        )

        target_task = Task(
            name="target_task",
            description=(
                "Check a Kyverno policy successfully deployed on the cluster. "
                "If not yet, create it first. You must report the filenames that you generated."
            ),
            expected_output="""All files you generated in your task and those explanations""",
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
    "deployed_resource": {
        "namespace": <PLACEHOLDER>,
        "kind": <PLACEHOLDER>,
        "name": <PLACEHOLDER>
    },
    "path_to_generated_kyverno_policy": <PLACEHOLDER>,
}
```
You can omit `namespace` in `deployed_resource` if the policy is a cluster-scope resource.
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
        
        with langfuse.start_as_current_observation(as_type="span", name="crewai-index-trace"):
            output = crew.kickoff(inputs=inputs)
        # print_attributes(output)
        print("Tokens from crew ai:", output.token_usage)
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

        return {"result": result}

    def _extract_metrics_from_trace(self, observations: ObservationsViews):
        """Extract metrics from Langfuse trace data"""
        print("\n" + "=" * 80)
        print("TRACE OBSERVATIONS")
        print("=" * 80)

        tool_call_latencies = []
        reasoning_token_usages = []
        tasks_token_usages = defaultdict(list)
        tasks = []

        llm_call_count = 0
        for idx, obs in enumerate(observations.data, 1):
            print(f"\n📊 Observation #{idx}")
            print(f"{'─'*80}")

            # Get all attributes (excluding private/magic methods)
            all_attrs = [attr for attr in dir(obs) if not attr.startswith("_")]

            for attr in sorted(all_attrs):
                try:
                    value = getattr(obs, attr)
                    # Skip methods/callables
                    if not callable(value):
                        # Format attribute name with proper spacing
                        attr_display = attr.replace("_", " ").title()
                        if(attr_display == "Usage Details" and isinstance(value, dict)):
                            for k, v in value.items():
                                if("reasoning" in k.lower()):
                                    reasoning_token_usages.append((obs.name, k, v))
                        if(attr_display == "Metadata"):
                            if("attributes" in value and isinstance(value["attributes"], dict)):
                                for k, v in value["attributes"].items():
                                    if("task_id" in k.lower()):
                                        task_id = v
                                        obs_id = obs.id
                                        tasks.append((task_id, obs_id))
                        # Truncate long values
                        value_str = str(value)
                        if(attr_display == "Type" and value_str == "TOOL"):
                            tool_call_latencies.append((obs.name, obs.latency))
                        if len(value_str) > 100:
                            value_str = value_str[:97] + "..."
                        value_str = value_str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                        print(f"  {attr_display:<25} {value_str}")
                except Exception as e:
                    print(f"  {attr:<25} <Error: {str(e)[:50]}>")
            
            if obs.completion_start_time and obs.start_time:
                ttft = (obs.completion_start_time - obs.start_time).total_seconds()
                print(f"  {'Time To First Token':<25} {ttft:.4f} seconds")

            if getattr(obs, 'model', None) is not None:
                llm_call_count += 1

            
        for task_id, obs_id in tasks:
            for obs in observations.data:
                if obs.parent_observation_id == obs_id:
                    if(obs.usage_details):
                        tasks_token_usages[task_id].append(obs.usage_details)
            
        print("\n" + "=" * 80)
        print("PERFORMANCE REPORT & NFRs")
        print("=" * 80)

        # 1. Global Latency
        # Try to find the root span (usually the one with no parent or named 'crewai-index-trace')
        root_span = next((o for o in observations.data if not o.parent_observation_id), None)
        if not root_span:
             # Fallback: check for specific name
             root_span = next((o for o in observations.data if o.name == 'crewai-index-trace'), None)
        
        if root_span:
            latency_sec = 0.0
            if getattr(root_span, 'latency', None):
                 latency_sec = root_span.latency 
            elif root_span.end_time and root_span.start_time:
                 latency_sec = (root_span.end_time - root_span.start_time).total_seconds()
            print(f"{'End to End Latency':<25} {latency_sec:.2f} seconds")

        # 2. Total Cost
        total_cost = sum(getattr(o, 'calculated_total_cost', 0.0) or 0.0 for o in observations.data)
        if total_cost > 0:
            print(f"{'Total Cost':<25} ${total_cost:.4f}")
        
        print(f"{'Total LLM Calls':<25} {llm_call_count}")

        # 3. Planning Overhead (Reasoning Ratio)
        total_reasoning_tokens = 0
        total_output_tokens = 0
        
        for obs in observations.data:
            usage = getattr(obs, 'usage_details', {}) or {}
            # Check for reasoning tokens in various common keys
            r_tokens = usage.get('reasoning', 0)
            # Also check nested structure if valid
            if not r_tokens:
                 for k, v in usage.items():
                     if "reasoning" in k.lower() and isinstance(v, (int, float)):
                         r_tokens += v
            
            total_reasoning_tokens += r_tokens
            
            # Output tokens usually under 'output' or 'completion'
            out_tokens = usage.get('output', 0) or usage.get('completion', 0)
            total_output_tokens += out_tokens

        if total_output_tokens > 0:
            overhead_pct = (total_reasoning_tokens / total_output_tokens) * 100
            print(f"{'Planning Overhead':<25} {overhead_pct:.1f}% ({total_reasoning_tokens}/{total_output_tokens} tokens)")
        
        print("\nTokens Breakdown:")
        for task_id, usages in tasks_token_usages.items():
            average_usage = 0
            count = 0
            for usage in usages:
                if "total" in usage:
                    average_usage += usage["total"]
                    count += 1
            if count > 0:
                average_usage /= count
                print(f"\nAverage token usage for Task ID {task_id}: {average_usage} tokens")

        print(f"\n{'='*80}")
        print(f"Total Observations: {len(observations.data)}")
        print(f"{'='*80}\n")

        print("Tool Call Latencies:")
        for tool_name, latency in tool_call_latencies:
            print(f"  {tool_name}: {latency} ms")
        
        print("\nReasoning Token Usages:")
        for obs_name, usage_type, token_count in reasoning_token_usages:
            print(f"  {obs_name} - {usage_type}: {token_count} tokens")



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
    _result = KubernetesKyvernoCrew().kickoff(inputs=inputs)
    result = _result.get("result")

    result_json_str = json.dumps(result, indent=2)

    print("---- Result ----")
    print(result_json_str)
    print("----------------")

    if args.output:
        with open(args.output, "w") as f:
            f.write(result_json_str)
