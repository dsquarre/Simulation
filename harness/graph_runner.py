"""The episode loop of runner.run_episode, rebuilt as a LangGraph graph.

    START -> agent -> tools -> agent -> tools -> ... -> END

agent  asks the model for its next reply (one step)
tools  runs the tool calls in that reply against the env
Two routing functions decide after each node whether to go on or stop.

Same limits, error messages and log records as runner.run_episode, so both runners
give the same result for the same model replies. Select it with RUNNER=langgraph.
"""
import copy
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .env import PASS_SCORE
from .runner import SYSTEM_PROMPT, now, show, trim_history
from .tools import SPECS, TOOLS, TOOL_NAMES, check_args


class EpisodeState(TypedDict):
    """Everything that changes during an episode. Each node returns the keys it updates."""
    messages: list          # the conversation, in OpenAI chat format
    step: int               # model replies so far
    started: float | None   # clock starts at the first model request
    finished: bool          # the model called `finish`
    termination: str | None # set when the episode must stop early
    reply: object           # the latest agents.Reply, handed from agent to tools
    record: dict            # this step's log record, filled by both nodes


def build_graph(env, client, task_id, seed, limits, log):
    """The env, model client and log are not state: they never change, so the nodes
    reach them through this closure instead."""

    def agent(state):
        if state["started"] is not None and time.monotonic() - state["started"] >= limits.max_seconds:
            return {"termination": "time_cap"}
        started = state["started"] if state["started"] is not None else time.monotonic()
        step = state["step"] + 1
        messages = copy.deepcopy(state["messages"])     # never edit state in place
        trim_history(messages, limits.context_budget_tokens)
        sent = copy.deepcopy(messages)
        remaining = limits.max_seconds - (time.monotonic() - started)
        call_start = time.monotonic()
        record = {"timestamp": now(), "agent": "model", "task": task_id, "step": step, "seed": seed,
                  "model_input": sent, "model_output_raw": None, "parsed_actions": [], "tool_result": [],
                  "errors": [], "state": {"hash": env.state_hash()}, "reward": None, "tokens": 0, "latency_ms": 0}
        try:
            reply = client.complete(sent, TOOLS, timeout=max(5, min(120, remaining)))
        except Exception as exc:    # includes provider errors and context-window overflow
            record["errors"].append({"kind": "model_call_failed", "detail": f"{type(exc).__name__}: {exc}"})
            log.write(record)
            show(record)
            return {"step": step, "started": started, "termination": "crash"}
        record["latency_ms"] = int((time.monotonic() - call_start) * 1000)
        record["model_output_raw"], record["tokens"] = reply.raw, reply.tokens

        # keep every tool call in the history so each one gets a matching result
        assistant = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": c.history_arguments()}}
                                       for c in reply.tool_calls]
        messages.append(assistant)
        return {"messages": messages, "step": step, "started": started, "reply": reply, "record": record}

    def tools(state):
        reply, record, finished = state["reply"], state["record"], state["finished"]
        messages = list(state["messages"])
        if not reply.tool_calls:
            record["errors"].append({"kind": "no_tool_call", "detail": "reply contained no tool call"})
            messages.append({"role": "user", "content": "No tool call was received. Call one of the "
                             "available tools, or call `finish` if you are done."})

        for index, call in enumerate(reply.tool_calls):
            args, parse_error = call.parse()
            record["parsed_actions"].append({"tool": call.name, "args": args, "parse_error": parse_error})
            if finished:
                result = "not run: the episode already finished"
            elif index >= limits.max_tool_calls:
                result = f"not run: at most {limits.max_tool_calls} tool calls are allowed per reply"
                record["errors"].append({"kind": "too_many_tool_calls", "detail": call.name})
            elif parse_error:
                result = f"error: {parse_error}"
                record["errors"].append({"kind": "unparsable_arguments", "detail": call.name})
            elif call.name not in SPECS:
                result = f"error: unknown tool '{call.name}'. Available tools: {', '.join(TOOL_NAMES)}"
                record["errors"].append({"kind": "unknown_tool", "detail": call.name})
            elif (problem := check_args(call.name, args)):
                result = f"error: {problem}"
                record["errors"].append({"kind": "invalid_arguments", "detail": f"{call.name}: {problem}"})
            elif call.name == "finish":
                finished, result = True, "Episode finished."
            else:
                result, is_error = env.step(call.name, args)
                if is_error:
                    record["errors"].append({"kind": "tool_error", "detail": result})
            result = result[:limits.max_output_chars]
            record["tool_result"].append(result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        record["state"] = {"hash": env.state_hash()}
        record["reward"] = env.score()
        log.write(record)
        show(record)
        termination = "step_cap" if not finished and state["step"] >= limits.max_steps else None
        return {"messages": messages, "finished": finished, "termination": termination}

    # routing: which node runs next
    def after_agent(state):
        return END if state["termination"] else "tools"

    def after_tools(state):
        return END if state["finished"] or state["termination"]  else "agent"

    graph = StateGraph(EpisodeState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile()


def run_episode_graph(env, task_id, seed, client, append_text, limits, log):
    """Drop-in replacement for runner.run_episode: same arguments, same return value."""
    prompt = env.reset(task_id, seed)
    client.start(env.task)
    system = (env.task.system_prompt or SYSTEM_PROMPT) + (("\n\n" + append_text) if append_text else "")
    start = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
             "step": 0, "started": None, "finished": False, "termination": None, "reply": None, "record": {}}

    app = build_graph(env, client, task_id, seed, limits, log)
    # each node visit is one LangGraph "super-step"; one model step = agent + tools
    state = app.invoke(start, {"recursion_limit": 2 * limits.max_steps + 5})

    started, termination, steps = state["started"], state["termination"], state["step"]
    try:
        final = env.score()
    except Exception as exc:
        final = {"total": None, "error": str(exc)}
        termination = termination or "crash"
    if termination is None:
        termination = "success" if (final["total"] or 0) >= PASS_SCORE else "failure"
    elapsed = round(time.monotonic() - started, 1) if started else 0
    log.write({"timestamp": now(), "agent": "model", "task": task_id, "event": "episode_end", "seed": seed,
               "steps": steps, "termination": termination, "reward": final, "elapsed_s": elapsed})
    return {"termination": termination, "steps": steps, "reward": final, "elapsed_s": elapsed}
