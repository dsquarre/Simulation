"""The environment: reset / step / score, backed by the sandbox service."""
import hashlib
import json

import requests

from .tasks import build_task

PASS_SCORE = 0.95   # a run counts as a success at or above this score


class Env:
    def __init__(self, sandbox_url, tool_timeout=60, verify_timeout=120):
        self.url = sandbox_url.rstrip("/")
        self.tool_timeout = tool_timeout
        self.verify_timeout = verify_timeout
        self.task = None

    def _post(self, path, payload, timeout):
        response = requests.post(self.url + path, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    # ---------------------------------------------------------------- reset
    def reset(self, task_id, seed):
        """Start a clean episode. The same (task_id, seed) always gives the same start."""
        self.task = build_task(task_id, seed)
        self._post("/reset", {"files": self.task.files}, self.tool_timeout)
        return (f"Task: {self.task.instructions}\n\n"
                "The project is in the current directory. Use the tools to look at the files, "
                "make your changes and test them, then call `finish`.")

    # ----------------------------------------------------------------- step
    def step(self, name, args):
        """Run one tool call. Returns (text, is_error)."""
        result = self._post("/tool", {"name": name, "args": args}, self.tool_timeout)
        return result["output"], result["is_error"]

    def state_hash(self):
        files = requests.get(self.url + "/state", timeout=self.tool_timeout).json()["files"]
        return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:8]

    # ---------------------------------------------------------------- score
    def score(self):
        """Programmatic verifier. Returns the reward components and their total.

        reward = tests_passed + no_regressions
          tests_passed   = 0.7 x share of target checks that pass
          no_regressions = 0.3 x share of regression checks that pass, but only
                           in proportion to target progress, so doing nothing earns 0
        """
        cases = self.task.cases
        results = self._post("/verify", {"cases": [c.code for c in cases]}, self.verify_timeout)["results"]
        tally = {"target": [0, 0], "regression": [0, 0]}
        for case, result in zip(cases, results):
            passed = result.get("ok") and _equal(result.get("result"), case.expect)
            tally[case.group][1] += 1
            tally[case.group][0] += 1 if passed else 0
        target = tally["target"][0] / max(1, tally["target"][1])
        regression = tally["regression"][0] / max(1, tally["regression"][1])
        components = {"tests_passed": round(0.7 * target, 4),
                      "no_regressions": round(0.3 * target * regression, 4)}
        components["total"] = round(sum(components.values()), 4)
        return components


def _equal(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) < 1e-9
    return a == b
