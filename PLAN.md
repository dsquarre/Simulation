# PLAN.md — dated log

## 2026-10-31 — Port the crop harness to LangGraph, trace it with LangSmith

### Goal (Danish's choice)
Learning and resume. The hackathon is over and the repo (`github.com/dsquarre/Simulation`) will be public.
Run the `crop_planning` episode through a LangGraph agent, traced in LangSmith, **next to** the existing
hand-written loop. The loop is kept, not replaced. No research claim, so no preregistration is needed.
If this later turns into a comparison ("graph agent beats plain loop"), register the comparison rules
first (OSF or a public timestamped commit).

### Concepts (short reference)
| Term | Meaning | Where it already lives in this repo |
|---|---|---|
| LLM | Text in, text out. Has no memory and takes no actions on its own. | the model behind `BASE_URL` / `MODEL` |
| Tool | A function the LLM may *ask* to call by returning a JSON request. Our code runs it and sends the result back. | `harness/tools.py` |
| Agent | An LLM in a loop: call model → run requested tools → append results → repeat until done. | `run_episode()` in `harness/runner.py` |
| Agent tooling | Libraries that provide that loop, state, retries, tracing and evaluation ready-made. | hand-written today |
| LangChain | Wrappers: one interface for many LLM providers, tool definitions, message types. | `OpenAIClient` in `harness/agents.py` |
| LangGraph | Builds the agent loop as a graph: *nodes* (functions) joined by *edges* (what runs next), sharing one *state*. | the `for step in ...` loop in `runner.py` |
| LangSmith | Hosted website (LangChain Inc.) that records every model and tool call as a clickable *trace*. | `logs/*/episode.jsonl` + `visualizer.html` |
| RAG | Retrieval before generation. Not needed here: the whole task fits in the prompt. | — |

### How `runner.py` maps onto LangGraph (my reading of the code; the mapping is my design choice)
| `runner.py` today | LangGraph equivalent |
|---|---|
| `messages` list | graph **state** (`messages` key with an append reducer) |
| `client.complete(...)` | **`agent` node** (chat model with tools bound) |
| `for call in reply.tool_calls: env.step(...)` | **`tools` node** (custom, so error texts stay identical) |
| `if finished: break` / no tool call | **conditional edge**: `agent → tools` or `→ END` |
| `max_steps`, `max_seconds` | `recursion_limit` + a time check in the `agent` node |
| `trim_history()` | reused before each model call |
| `JsonlLog` | kept (feeds `visualizer.html`), plus LangSmith traces |

### Step 0. Public-repo cleanup — wording DONE 2026-10-31 (not committed)
- Done: judge/hackathon wording removed from `.env.example`, `requirements.txt`, `harness/agents.py`,
  `harness/run.py`, `harness/tasks.py`, `README.md`, `guide.html`, `gui.html`.
- Left on purpose: `LICENSE` ("Copyright (c) 2026 Hackathon organisers"; see open question), and the
  model-facing prompt in `crop_planning.py` ("complete your submission"; changing it changes the benchmark).
- Open: `gui.html` is byte-identical to `guide.html` and nothing links to either; delete `gui.html`?
- Done: GUI fixes in `visualizer.html` (hover no longer moves tiles; water/capital bars show what is left).
Original proposal:
- Remove hackathon/judge wording from: `README.md`, `guide.html`, `gui.html`, `harness/agents.py`,
  `requirements.txt`, `.env.example`, `docker-compose.yml` ("judges' machine", "API key email", "gate budget for
  the live run", "hackathon-compliant", ...).
- `RL_Hackathon_by_2une_PS.pdf`: `git rm` done 2026-10-31 (staged, not committed). The repo is already
  PUBLIC and the PDF is in commit `caeb034` on `origin/main`, so it stays downloadable from history.
  Purging it needs `git filter-repo` + force-push (rewrites every commit hash). Open: Danish decides.
- Comments (Danish's decision): remove stale ones and judge/hackathon references; keep comments that
  explain non-obvious behaviour.
- Tools after Danish's reset to `dd0537c`: `list_files`, `read_file`, `write_file`, `run_python`,
  `plant_season`, `finish` (no `trade_water`).
- Already fine (checked): `.env` is gitignored and was never committed; `logs/` is ignored.

### Step 1. Install — DONE 2026-10-31 (approved by Danish)
No conda env has `langgraph` (checked: base = Python 3.13, web = 3.10, whosnext has no Python).
Proposal: new conda env `agents` with Python 3.12 (same as the Docker image), install
`langgraph langchain-openai langsmith`, then pin the exact versions in `requirements-langgraph.txt`
(kept separate from `requirements.txt`).
Tested 2026-10-31 with `pip install --dry-run`: the three packages resolve to 43 pure-API packages
(langgraph 1.2.18, langchain-core 1.7.3, langchain-openai 1.7.3, langsmith 0.14.16, openai 3.41.0, ...).
No torch, tensorflow, numpy or CUDA. Note the conflict: langchain-openai wants `openai` 3.x, while
`requirements.txt` pins `openai==1.109.1`. The separate env avoids it; check `agents.py` still works on 3.x
before merging the requirements.
Result: env `agents` created; `from langgraph.graph import StateGraph` imports (tested). Exact versions
frozen in `requirements-langgraph.txt` (44 packages, no torch/tensorflow/numpy).

### Step 2. Falsify first: does the port reproduce the old loop?
- `harness/graph_runner.py`: `agent` node + `tools` node + one conditional edge (~80 lines),
  selected with `RUNNER=langgraph`; default stays `runner.py`.
- Model: the offline `mock/solver` and `mock/idle` clients (no API key, no cost).
- Check: seed 0, both tasks, both mocks → same final reward, termination and step count as `runner.py`.
  Any difference means the port is wrong; fix before going on.

### Step 3. Real model through LangChain
- `ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY)` with the 6 tools bound.
- Check: one `crop_planning` episode on seed 0 finishes and `episode.jsonl` still loads in `visualizer.html`.
  A single run is a smoke test, not evidence about quality.

### Step 4. LangSmith tracing
- Free LangSmith account; `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=crop-harness` in `.env`.
- Note: prompts, replies and tool results are sent to LangSmith's servers. The oracle and hidden checks
  never enter the prompt, so they are not sent.
- Check: in the LangSmith UI, find season 3's `plant_season` call and its result.

### Step 5. One graph-only feature (resume-worthy part)
A `validate` node between `agent` and `tools` that rejects an allocation exceeding the *visible*
land/water/budget and asks the model to retry. Leakage rule: it may use only information already in the
agent's prompt (never the oracle, `inst` internals or future prices).

### Step 6 (optional). LangSmith dataset/experiment over several seeds; scoring stays our verifier.

### Status of claims in this plan
- Mapping tables and file lists: from reading the code this session (not run).
- LangGraph/LangSmith API names (`StateGraph`, `add_messages`, `recursion_limit`, `ChatOpenAI`, `LANGSMITH_*`):
  from my training knowledge; verify against current docs at install time.
- Add-alongside design, `validate` node, conda env name: my proposals.

## 2026-10-31 (later) — Static web version for friends, file cleanup, history reset

### Done (not committed)
- **Parity bug found and fixed (tested).** The browser game recomputed profits differently from the Python
  rules: 356/774 test plans disagreed, by up to $36. Causes: JS did not round the yield multiplier to 2
  decimals, and `Math.round` rounds halves up while Python's `round` rounds halves to even. Added
  `pyRound()` in `visualizer.html`; after the fix 0/774 mismatches (200 seeds × optimal + random plans),
  and the multiplier matches for every nitrogen level 1–100. Test scripts were in the session scratchpad.
- `visualizer.html` now loads seeds from `seeds/<n>.json` on the static site and from `/api/instance` locally.
  Removed the fake fallback farm (made-up optimum $8,500) shown when loading failed; now it shows an error.
- On the static site the "Play as GPT Luna" button opens a popup explaining how to clone and run the harness.
- `build_web.py` builds `web/` (index.html + seeds 0–999, 4 MB). `web/` is gitignored (a build output).
  Checked in headless Chrome over a plain static server: seed 0 loads, bars start full.
- Removed redundant files: `farm_seeds.js`, `farm_seeds.json` (12 preloaded seeds, replaced by the above),
  `gui.html` (byte-identical copy of `guide.html`). `play_gui.py` no longer writes into `farm_seeds.json`.
- Backup of all git history: `~/Simulation-history-backup-2026-10-31.bundle` (verified).

### Known limits (from reading the code)
- The browser can't hide anything: future prices and the optimal plan are in `seeds/<n>.json`. Website scores
  are for fun, not benchmark results.
- Not tested by hand: hover feel, the popup and drag-and-drop on the built site (only a static screenshot).

### Pending (Danish's confirmation)
- Reset git history to a single commit and force-push to the public `origin/main`.
