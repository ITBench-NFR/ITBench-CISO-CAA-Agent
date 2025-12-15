from langfuse.api.resources.commons.types.observations_view import ObservationsView
import json
import datetime
from typing import List

def extract_metrics_from_trace(observations_data: List[ObservationsView]):
    """Extract metrics from Langfuse trace data"""

    # List to store all observations for JSON dump
    all_observations_data = []

    for idx, obs in enumerate(observations_data):
        try:
            obs_dict = obs.dict(by_alias=False, exclude_unset=False, exclude_none=False)
        except TypeError:
            # Fallback if by_alias is not supported
            obs_dict = obs.dict()

        all_observations_data.append(obs_dict)

    # Serialize and Save to JSON
    def json_serial(obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return str(obj)

    try:
        with open("observations_dump.json", "w") as f:
            json.dump(all_observations_data, f, default=json_serial, indent=2)
        print(f"\n[INFO] Observations dumped to 'observations_dump.json'")
    except Exception as e:
        print(f"\n[ERROR] Failed to dump observations to JSON: {e}")