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
import json
import os


def run(inputs: dict):
    from ciso_agent.manager import CISOManager
    manager = CISOManager(
        eval_policy=False,
    )
    state = inputs
    output = manager.invoke(state)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("-g", "--goal", default="", help="The compliance goal for the agent to achieve")
    parser.add_argument("-o", "--output", default="", help="The path to the output JSON file")
    parser.add_argument("-a", "--auto-approve", action="store_true", help="do nothing for now")
    parser.add_argument("--agent-type", default=os.getenv("AGENT_TYPE", "crew"), help="The type of agent to run (crew or plan_execute)")
    args = parser.parse_args()

    if args.agent_type:
        os.environ["AGENT_TYPE"] = args.agent_type

    from ciso_agent.manager import CISOState

    inputs = CISOState(goal=args.goal)
    result = run(inputs=inputs)
    result_json_str = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result_json_str)
