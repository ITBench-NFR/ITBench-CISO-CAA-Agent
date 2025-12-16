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
import string
import time

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langtrace_python_sdk import langtrace
from langfuse import get_client

from ciso_agent.llm import init_agent_llm, extract_code
from ciso_agent.tools.generate_opa_rego import GenerateOPARegoTool
from ciso_agent.tools.run_opa_rego import RunOPARegoTool
from ciso_agent.tools.generate_playbook import GeneratePlaybookTool
from ciso_agent.tools.run_playbook import RunPlaybookTool
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from ciso_agent.tracing import extract_metrics_from_trace
from ciso_agent.vllm_metrics import VLLMMetricsCollector, is_vllm_metrics_enabled

load_dotenv()
langfuse = get_client()

class RHELPlaybookOPACrew(object):
    agent_goal: str = """I would like to check if the following condition is satisfiled, given a host name `rhel9_servers`
    ${compliance}

To check the condition, do the following steps.
- collect some required configuration to check the condition from the RHEL host and save it locally. you can use ansible-playbook to do that.
- chcek if the condition is met by using rego policy with the input given by the step above.

for those steps, you need to create ansible playbook `playbook.yml` and OPA rego policy `policy.rego`.
If you can fix the generated code, do it and run the fixed code again.

You can use the inventory file `inventory.ansible.ini` to access `rhel9_servers`.

Once you get a final answer, you can quit the work.
"""

    tool_description: str = """This agent has the following tools to use:
- RunOPARegoTool
- GenerateOPARegoTool
- RunPlaybookTool
- GeneratePlaybookTool
"""

    input_description: dict = {
        "compliance": "a short string of compliance requirement",
        "workdir": "a working directory to save temporary files",
    }

    output_description: dict = {
        "path_to_generated_playbook": "a string of the filepath to the generated playbook",
        "path_to_generated_rego_policy": "a string of the filepath to the generated rego policy",
        "path_to_collected_data_by_playbook": "a string of the filepath to the collected data",
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
            test_case_id = f"rhel_playbook_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                metrics_collector = VLLMMetricsCollector()
                metrics_collector.start_collection(test_case_id)
            except Exception as e:
                print(f"[WARN] Failed to start vLLM metrics collection: {e}")
                metrics_collector = None

        return_value = self.run_scenario(**inputs)

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
            extract_metrics_from_trace(all_observations)
        return return_value

    def run_scenario(self, goal: str, **kwargs):
        workdir = kwargs.get("workdir")
        if not workdir:
            workdir = os.path.join(self.workdir_root, datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S_"), "workspace")

        if not os.path.exists(workdir):
            os.makedirs(workdir, exist_ok=True)

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
            description="""Check a rego policy for a given input file. If policy or input file are not ready, prepare them first.""",
            expected_output="""A boolean which indicates if the check is passed or not""",
            agent=test_agent,
            tools=[
                RunOPARegoTool(workdir=workdir),
                GenerateOPARegoTool(workdir=workdir),
                RunPlaybookTool(workdir=workdir),
                GeneratePlaybookTool(workdir=workdir),
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
    "path_to_generated_playbook": <PLACEHOLDER>,
    "path_to_generated_rego_policy": <PLACEHOLDER>,
    "path_to_collected_data_by_playbook": <PLACEHOLDER>,
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
        )
        inputs = {}
        output = crew.kickoff(inputs=inputs)
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

        return {"result": result}


if __name__ == "__main__":
    default_compliance = "Ensure that the cron daemon is enabled"
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("-c", "--compliance", default=default_compliance, help="The compliance description for the agent to do something for")
    parser.add_argument("-w", "--workdir", default="", help="The path to the work dir which the agent will use")
    parser.add_argument("-o", "--output", help="The path to the output JSON file")
    args = parser.parse_args()

    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)

    inputs = dict(
        compliance=args.compliance,
        workdir=args.workdir,
    )
    _result = RHELPlaybookOPACrew().kickoff(inputs=inputs)
    result = _result.get("result")

    result_json_str = json.dumps(result, indent=2)

    print("---- Result ----")
    print(result_json_str)
    print("----------------")

    if args.output:
        with open(args.output, "w") as f:
            f.write(result_json_str)
