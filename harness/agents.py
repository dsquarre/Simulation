"""Agents: the model client, two offline mock models, and the baselines."""
import json
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw text exactly as the model sent it

    def parse(self):
        """Return (args, error). `args` is a dict when the arguments are valid JSON."""
        try:
            args = json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError as exc:
            return None, f"arguments are not valid JSON ({exc.msg})"
        if not isinstance(args, dict):
            return None, "arguments must be a JSON object"
        return args, None

    def history_arguments(self):
        """Arguments as stored in the conversation. Invalid JSON is replaced with {} so
        the next request is still accepted by every provider."""
        args, error = self.parse()
        return json.dumps(args) if error is None else "{}"


@dataclass
class Reply:
    text: str | None
    tool_calls: list = field(default_factory=list)
    raw: str = ""
    tokens: int = 0


# ------------------------------------------------------------ real models
class OpenAIClient:
    """Calls any model through an OpenAI-compatible endpoint.

    The harness reads the base_url, api_key and model name from the environment variables
    BASE_URL, API_KEY and MODEL. The same call is used for every model, so switching models
    only means changing those values.
    """

    def __init__(self, model, api_key, base_url=None, max_tokens=None):
        import openai
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url or None, max_retries=2)
        self.model, self.max_tokens = model, max_tokens

    def start(self, task):
        pass

    def complete(self, messages, tools, timeout):
        kwargs = dict(model=self.model, messages=messages, tools=tools, timeout=timeout)
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens      # optional: set MAX_TOKENS to cap each reply
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = []
        for index, call in enumerate(message.tool_calls or []):
            function = getattr(call, "function", None)
            calls.append(ToolCall(id=call.id or f"call_{index}",
                                  name=(function.name if function else "") or "",
                                  arguments=(function.arguments if function else "") or ""))
        usage = getattr(response, "usage", None)
        return Reply(text=message.content, tool_calls=calls, raw=response.model_dump_json(),
                     tokens=getattr(usage, "total_tokens", 0) or 0)


class LangChainClient:
    """The same job as OpenAIClient, done through LangChain (CLIENT=langchain).

    LangChain has its own message classes instead of plain dicts:
        {"role": "system"}     -> SystemMessage
        {"role": "user"}       -> HumanMessage
        {"role": "assistant"}  -> AIMessage (tool requests in .tool_calls, args already parsed to a dict)
        {"role": "tool"}       -> ToolMessage (linked to its request by tool_call_id)
    The harness keeps its dict history, so this class converts on the way in and on the way out.
    """

    def __init__(self, model, api_key, base_url=None, max_tokens=None):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url or None,
                              max_tokens=max_tokens, max_retries=2, timeout=120)
        self.bound = None

    def start(self, task):
        pass

    @staticmethod
    def to_langchain(messages):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        out = []
        for m in messages:
            if m["role"] == "system":
                out.append(SystemMessage(m["content"]))
            elif m["role"] == "user":
                out.append(HumanMessage(m["content"]))
            elif m["role"] == "assistant":
                calls = [{"name": c["function"]["name"], "args": json.loads(c["function"]["arguments"]),
                          "id": c["id"]} for c in m.get("tool_calls", [])]
                out.append(AIMessage(content=m["content"] or "", tool_calls=calls))
            else:
                out.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"]))
        return out

    @staticmethod
    def from_langchain(message):
        """AIMessage -> Reply. Calls whose arguments were not valid JSON land in invalid_tool_calls with
        the raw text; they are passed on as-is so the harness reports them as unparsable."""
        calls = [ToolCall(c["id"] or f"call_{i}", c["name"] or "", json.dumps(c["args"]))
                 for i, c in enumerate(message.tool_calls)]
        calls += [ToolCall(c.get("id") or f"bad_{i}", c.get("name") or "", c.get("args") or "")
                  for i, c in enumerate(message.invalid_tool_calls)]
        text = message.content
        if isinstance(text, list):      # some providers return a list of content parts
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        usage = message.usage_metadata or {}
        return Reply(text=text or None, tool_calls=calls, raw=message.model_dump_json(),
                     tokens=usage.get("total_tokens", 0))

    def complete(self, messages, tools, timeout):
        # timeout is fixed at 120 s in __init__ (the runner's per-call maximum)
        if self.bound is None:
            self.bound = self.llm.bind_tools(tools)       # tells the model which tools exist
        return self.from_langchain(self.bound.invoke(self.to_langchain(messages)))


# ------------------------------------------------------------ mock models
class MockClient:
    """Offline stand-ins so the harness runs with no API key.

    mock/solver   plays the reference answer automatically, no input needed: crop tasks plant the
                  reference schedule one season per turn through plant_season, other tasks write
                  the reference files. Then finishes.
    mock/idle     finishes immediately without doing anything.
    human/solver  you play as the model. Crop tasks: you see exactly the messages and tools the
                  model sees and type tool calls (see _read_command); Enter plays the reference move.
                  box_packing asks for the whole packing. With no terminal, Enter is assumed.
    """

    def __init__(self, kind):
        self.kind = kind
        self.turn = 0
        self.task = None

    def start(self, task):
        self.task, self.turn = task, 0

    def complete(self, messages, tools, timeout):
        self.turn += 1
        calls = []
        crop = self.task is not None and self.task.task_id.startswith("crop_planning")
        if self.kind == "human" and crop:
            calls = self._crop_turn(messages, tools)
        elif self.kind == "solver" and crop:
            calls = [self._reference_move()]
        elif self.kind in ("solver", "human") and self.turn == 1:
            if self.kind == "human" and self.task and self.task.task_id == "box_packing":
                print("\n" + "=" * 70, flush=True)
                print(f" [HUMAN AS LLM INTERACTIVE SOLVER - {self.task.task_id.upper()}]", flush=True)
                print("=" * 70, flush=True)
                print(self.task.instructions, flush=True)
                print("=" * 70, flush=True)
                print("Enter your solution JSON (e.g. [[0, 2], [1, 3], ...])", flush=True)
                print("or press Enter to submit oracle reference solution directly:", flush=True)
                try:
                    user_str = input("> ").strip()
                except (EOFError, OSError):
                    user_str = ""

                if user_str:
                    try:
                        parsed = json.loads(user_str)
                        if isinstance(parsed, list):
                            content = json.dumps({"boxes": parsed}, indent=2)
                        elif isinstance(parsed, dict) and "boxes" in parsed:
                            content = json.dumps(parsed, indent=2)
                        else:
                            content = user_str
                    except Exception as err:
                        print(f"\n⚠️  JSON Syntax Error: {err}", flush=True)
                        content = user_str
                else:
                    content = self.task.reference.get("solution.json", "{}")

                calls.append(("write_file", {"path": "solution.json", "content": content}))
                calls.append(("finish", {}))
            else:
                for path, content in self.task.reference.items():
                    calls.append(("write_file", {"path": path, "content": content}))
                calls.append(("finish", {}))
        else:
            calls.append(("finish", {}))
        tool_calls = [ToolCall(f"mock-{self.turn}-{i}", name, json.dumps(args))
                      for i, (name, args) in enumerate(calls)]
        return Reply(text=None, tool_calls=tool_calls, raw=json.dumps({"mock": self.kind, "turn": self.turn}))

    def _reference_move(self):
        """The reference schedule's move for the current season, or finish when the game is over."""
        if self.task.is_finished:
            return "finish", {}
        return "plant_season", {"allocation": json.dumps(self.task.hint_schedule[self.task.current_season - 1])}

    def _crop_turn(self, messages, tools):
        """Show what the model would see since its last reply, then read one tool call."""
        if self.turn == 1:
            print("\n" + "=" * 70 + f"\n [YOU ARE THE MODEL - {self.task.task_id.upper()}]\n" + "=" * 70, flush=True)
            print("[tools]", flush=True)
            for spec in tools:
                f = spec["function"]
                args = ", ".join(f["parameters"]["properties"]) or "-"
                print(f"  {f['name']}({args}): {f['description']}", flush=True)
            print(f"\n[system]\n{messages[0]['content']}\n\n[user]\n{messages[1]['content']}", flush=True)
        else:
            last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
            for m in messages[last + 1:]:
                print(f"\n[{m['role']}]\n{m['content']}", flush=True)
        default = self._reference_move()
        print("=" * 70, flush=True)
        print("Your tool call:  plant_season {\"rice\": 10}  |  run_python print(1+1)  |  read_file notes.txt\n"
              "                 write_file notes.txt  |  list_files  |  finish  |  bare {...} = plant_season\n"
              "                 run_python / write_file with nothing after them read lines until a line with only '.'\n"
              f"Harness shortcut (the model does not get this): Enter = {default[0]} {default[1].get('allocation', '')}",
              flush=True)
        return [self._read_command(default)]

    @staticmethod
    def _read_lines():
        lines = []
        while True:
            try:
                line = input("... ")
            except (EOFError, OSError):
                break
            if line == ".":
                break
            lines.append(line)
        return "\n".join(lines)

    def _read_command(self, default):
        """Turn one typed command into the (tool, args) a model would send. Unknown tools and bad
        arguments are passed through unchanged, so the harness answers with the model's error."""
        try:
            text = input("> ").strip()
        except (EOFError, OSError):
            return default
        if not text:
            return default
        if text.startswith("{"):
            return "plant_season", {"allocation": text}
        name, _, rest = text.partition(" ")
        rest = rest.strip()
        if rest.startswith("{"):
            try:
                parsed = json.loads(rest)
            except json.JSONDecodeError:
                parsed = None
            # tool {"arg": ...} sends the arguments as typed (plant_season {"rice": 3} is an allocation)
            if isinstance(parsed, dict) and (name != "plant_season" or "allocation" in parsed):
                return name, parsed
        if name == "plant_season":
            return name, {"allocation": rest}
        if name == "run_python":
            return name, {"code": rest or self._read_lines()}
        if name == "read_file":
            return name, {"path": rest}
        if name == "write_file":
            path, _, inline = rest.partition(" ")
            return name, {"path": path, "content": inline or self._read_lines()}
        return name, {}


def make_client(model, api_key, base_url, max_tokens=None, library="openai"):
    if model == "human/solver":
        return MockClient("human")
    if model.startswith("mock/"):
        kind = model.split("/", 1)[1]
        if kind not in ("solver", "idle"):
            raise ValueError("offline models are 'mock/solver', 'mock/idle' and 'human/solver'")
        return MockClient(kind)
    if not api_key:
        raise ValueError("API_KEY is not set")
    if library == "langchain":
        return LangChainClient(model, api_key, base_url, max_tokens)   # needs requirements-langgraph.txt
    return OpenAIClient(model, api_key, base_url, max_tokens)


# -------------------------------------------------------------- baselines
def oracle_actions(task):
    """The reference solution as a list of (tool, args) actions. Must score full marks."""
    if hasattr(task, "oracle_actions"):
        return task.oracle_actions()
    return [("write_file", {"path": path, "content": content}) for path, content in task.reference.items()]


def noop_actions(task):
    """An agent that gives up immediately. Must score about zero."""
    return []
