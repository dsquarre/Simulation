"""Task registry.

Registered Tasks:
  1. crop_planning: Multi-Season Crop Rotation & Market POMDP with Dynamic Soil Nitrogen (Primary Demo, Hard)
  1b. crop_planning_v2: compounding capital, ruin, crop-specific price shocks (harness/crop_v2.py)
  2. box_packing: 3D Discrete Box Packing with Spatial Incompatibilities (Combinatorial Optimization, Hard)
"""
import textwrap
from dataclasses import dataclass


@dataclass
class Case:
    group: str      # "target": fails before the fix.  "regression": must keep passing.
    code: str       # runs inside the sandbox and must set a variable called `result`
    expect: object  # compared in the harness only; never sent to the sandbox


@dataclass
class TaskInstance:
    task_id: str
    title: str
    difficulty: str
    instructions: str
    files: dict      # starting files the agent can see
    cases: list      # hidden checks (list of Case)
    reference: dict  # files that solve the task; used by the oracle baseline
    system_prompt: str = ""  # replaces runner.SYSTEM_PROMPT when set
    closing: str = ""        # replaces the default last line of the first user message when set


def _case(group, code, expect):
    return Case(group, textwrap.dedent(code).strip(), expect)


def crop_planning_task(seed):
    from .crop_planning import crop_planning_task as _cpt
    return _cpt(seed)


def box_packing_task(seed):
    from .box_packing import box_packing_task as _bpt
    return _bpt(seed)


def crop_planning_v2_task(seed):
    from .crop_v2_task import crop_v2_task
    return crop_v2_task(seed)


TASKS = {
    "crop_planning": crop_planning_task,
    "crop_planning_v2": crop_planning_v2_task,
    "box_packing": box_packing_task,
}
DEMO_TASK = "crop_planning"


def build_task(task_id, seed):
    if task_id not in TASKS:
        raise KeyError(f"unknown task '{task_id}'. Available: {', '.join(TASKS)}")
    return TASKS[task_id](seed)
