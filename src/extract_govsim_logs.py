"""Pull every harvest decision (the agent's private reasoning), every discussion turn and
GovSim's extracted per-month limit out of the fishery log_env.json files, for the groups
the deviant-seat hand coding reads (D-0130). Runs on ada, where the GovSim results live.

Usage (on ada): python src/extract_govsim_logs.py
Writes evals-out-p2/govsim/32b/handcode/log_extract.jsonl (one record per log event).
"""
import glob
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "third_party/GovSim/simulation/results/fishing_v7.0"
COMPS = ["BBBBA", "BBBBa", "BBBBm", "AAAAA", "aaaaa", "mmmmm", "BBBBB"]


def main():
    out = REPO / "evals-out-p2/govsim/32b/handcode/log_extract.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for comp in COMPS:
            for unit in sorted(glob.glob(f"{ROOT}/dw_32b-tf_fish_{comp}_s*")):
                for lf in glob.glob(f"{unit}/*/log_env.json"):
                    for x in json.load(open(lf)):
                        h = x.get("html_interactions")
                        if isinstance(h, list):
                            h = "\n=====\n".join(h)
                        rec = {"unit": os.path.basename(unit), "comp": comp, "agent": x["agent_id"],
                               "round": x["round"], "action": x["action"], "wanted": x.get("wanted_resource"),
                               "collected": x.get("resource_collected"), "limit": x.get("resource_limit"),
                               "utterance": x.get("utterance"),
                               "pool_before": x.get("resource_in_pool_before_harvesting")}
                        if x["action"] == "harvesting" and h:
                            parts = re.split(r"<strong>ASSISTANT</strong>:", h)
                            rec["reasoning"] = parts[-1] if len(parts) > 1 else h[-4000:]
                            rec["n_calls"] = len(parts) - 1
                        f.write(json.dumps(rec) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
