"""Demo tasks. Replace these with your own.

Every task is built from a seed, so the same (task_id, seed) always gives the
same starting files, hidden checks and reference solution. Most tasks here
ignore the seed; `bulk_discount` uses it to generate a different variant each
time, which is a cheap way to get many tasks out of one design.
"""
import random
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


def _case(group, code, expect):
    return Case(group, textwrap.dedent(code).strip(), expect)


# ---------------------------------------------------------------- slugify
def slugify_task(seed):
    return TaskInstance(
        task_id="slugify", title="Fix slugify", difficulty="easy",
        instructions=("`textutils.slugify` should turn a title into a URL slug: lowercase, "
                      "punctuation removed, any run of spaces or dashes collapsed into a single "
                      "dash, and no leading or trailing dash. Fix it."),
        files={"textutils.py": 'def slugify(text):\n    """Turn a title into a URL slug."""\n'
                               '    return text.lower().replace(" ", "-")\n'},
        cases=[
            _case("target", 'from textutils import slugify\nresult = slugify("Hello, World!")', "hello-world"),
            _case("target", 'from textutils import slugify\nresult = slugify("  Multiple   spaces  ")', "multiple-spaces"),
            _case("target", 'from textutils import slugify\nresult = slugify("a--b")', "a-b"),
            _case("target", 'from textutils import slugify\nresult = slugify("C++ rocks")', "c-rocks"),
            _case("target", 'from textutils import slugify\nresult = slugify("---Already---dashed---")', "already-dashed"),
            _case("regression", 'from textutils import slugify\nresult = slugify("simple")', "simple"),
            _case("regression", 'from textutils import slugify\nresult = slugify("two words")', "two-words"),
        ],
        reference={"textutils.py": 'import re\n\n\ndef slugify(text):\n'
                                   '    text = re.sub(r"[^a-z0-9]+", "-", text.lower())\n'
                                   '    return text.strip("-")\n'},
    )


# --------------------------------------------------------- parse_duration
def duration_task(seed):
    invalid = ('from durations import parse_duration\ntry:\n    parse_duration("{}")\n'
               '    result = "no error"\nexcept ValueError:\n    result = "ValueError"')
    return TaskInstance(
        task_id="parse_duration", title="Parse combined durations", difficulty="easy",
        instructions=("`durations.parse_duration` should also accept combined values such as '1h30m' "
                      "or '1h1m1s' and return the total number of seconds as an int. Invalid input "
                      "(an empty string, an unknown unit, or a number with no unit) must raise ValueError."),
        files={"durations.py": 'UNITS = {"s": 1, "m": 60, "h": 3600}\n\n\ndef parse_duration(text):\n'
                               '    """Convert a duration like \'90s\', \'5m\' or \'2h\' to seconds."""\n'
                               '    return int(text[:-1]) * UNITS[text[-1]]\n'},
        cases=[
            _case("target", 'from durations import parse_duration\nresult = parse_duration("1h30m")', 5400),
            _case("target", 'from durations import parse_duration\nresult = parse_duration("1h1m1s")', 3661),
            _case("target", invalid.format("5x"), "ValueError"),
            _case("target", invalid.format("10d"), "ValueError"),
            _case("target", invalid.format("30"), "ValueError"),
            _case("regression", 'from durations import parse_duration\nresult = parse_duration("45s")', 45),
            _case("regression", 'from durations import parse_duration\nresult = parse_duration("10m")', 600),
            _case("regression", 'from durations import parse_duration\nresult = parse_duration("2h")', 7200),
            _case("regression", invalid.format(""), "ValueError"),
        ],
        reference={"durations.py": 'import re\n\nUNITS = {"s": 1, "m": 60, "h": 3600}\n'
                                   '_PART = re.compile(r"(\\d+)([smh])")\n\n\ndef parse_duration(text):\n'
                                   '    if not text or not re.fullmatch(r"(?:\\d+[smh])+", text):\n'
                                   '        raise ValueError(f"invalid duration: {text!r}")\n'
                                   '    return sum(int(n) * UNITS[u] for n, u in _PART.findall(text))\n'},
    )


# -------------------------------------------------------- merge_intervals
def intervals_task(seed):
    def call(data):
        return f"from intervals import merge_intervals\nresult = merge_intervals({data})"
    return TaskInstance(
        task_id="merge_intervals", title="Merge intervals", difficulty="medium",
        instructions=("`intervals.merge_intervals` should merge overlapping [start, end] intervals and "
                      "return them as a list of [start, end] lists sorted by start. Intervals that touch, "
                      "such as [1, 2] and [2, 3], count as overlapping. The input may be unsorted."),
        files={"intervals.py": 'def merge_intervals(intervals):\n'
                               '    """Merge overlapping [start, end] intervals and return them sorted."""\n'
                               '    merged = []\n    for start, end in intervals:\n'
                               '        if merged and start < merged[-1][1]:\n'
                               '            merged[-1][1] = max(merged[-1][1], end)\n'
                               '        else:\n            merged.append([start, end])\n    return merged\n'},
        cases=[
            _case("target", call([[8, 10], [1, 3], [2, 6], [15, 18]]), [[1, 6], [8, 10], [15, 18]]),
            _case("target", call([[1, 2], [2, 3]]), [[1, 3]]),
            _case("target", call([[5, 6], [1, 5], [7, 8]]), [[1, 6], [7, 8]]),
            _case("target", call([[2, 3], [1, 10]]), [[1, 10]]),
            _case("regression", call([[1, 3], [5, 6]]), [[1, 3], [5, 6]]),
            _case("regression", call([[1, 10], [2, 3]]), [[1, 10]]),
            _case("regression", call([]), []),
        ],
        reference={"intervals.py": 'def merge_intervals(intervals):\n    merged = []\n'
                                   '    for start, end in sorted(intervals):\n'
                                   '        if merged and start <= merged[-1][1]:\n'
                                   '            merged[-1][1] = max(merged[-1][1], end)\n'
                                   '        else:\n            merged.append([start, end])\n    return merged\n'},
    )


# -------------------------------------------------------------- lru_cache
def lru_task(seed):
    head = "from cache import LRUCache\n"
    return TaskInstance(
        task_id="lru_cache", title="Fix the LRU cache", difficulty="medium",
        instructions=("`cache.LRUCache` should evict the least recently used key when it is full. Both "
                      "`get` and `put` must mark a key as the most recently used. `get` returns -1 for a "
                      "missing key."),
        files={"cache.py": 'class LRUCache:\n    """A fixed-size cache that evicts the least recently used key."""\n\n'
                           '    def __init__(self, capacity):\n        self.capacity = capacity\n'
                           '        self.data = {}  # insertion order is age order\n\n'
                           '    def get(self, key):\n        return self.data.get(key, -1)\n\n'
                           '    def put(self, key, value):\n'
                           '        if len(self.data) >= self.capacity and key not in self.data:\n'
                           '            del self.data[next(iter(self.data))]\n'
                           '        self.data[key] = value\n'},
        cases=[
            _case("target", head + "c = LRUCache(2)\nc.put(1, 1); c.put(2, 2); c.get(1); c.put(3, 3)\n"
                                   "result = [c.get(1), c.get(2), c.get(3)]", [1, -1, 3]),
            _case("target", head + "c = LRUCache(2)\nc.put(1, 1); c.put(2, 2); c.put(1, 10); c.put(3, 3)\n"
                                   "result = [c.get(1), c.get(2), c.get(3)]", [10, -1, 3]),
            _case("target", head + "c = LRUCache(3)\nc.put(1, 1); c.put(2, 2); c.put(3, 3)\n"
                                   "c.get(1); c.get(2); c.put(4, 4)\n"
                                   "result = [c.get(1), c.get(2), c.get(3), c.get(4)]", [1, 2, -1, 4]),
            _case("regression", head + "c = LRUCache(1)\nresult = c.get(5)", -1),
            _case("regression", head + "c = LRUCache(1)\nc.put(1, 1); c.put(2, 2)\n"
                                       "result = [c.get(1), c.get(2)]", [-1, 2]),
        ],
        reference={"cache.py": 'class LRUCache:\n    def __init__(self, capacity):\n'
                               '        self.capacity = capacity\n        self.data = {}\n\n'
                               '    def get(self, key):\n        if key not in self.data:\n            return -1\n'
                               '        value = self.data.pop(key)\n        self.data[key] = value\n'
                               '        return value\n\n    def put(self, key, value):\n'
                               '        if key in self.data:\n            self.data.pop(key)\n'
                               '        elif len(self.data) >= self.capacity:\n'
                               '            del self.data[next(iter(self.data))]\n        self.data[key] = value\n'},
    )


# ---------------------------------------------- bulk_discount (seeded)
def discount_task(seed):
    rng = random.Random(seed)
    unit = rng.choice([4.99, 7.25, 12.5, 19.99])
    t1 = rng.randint(5, 9)
    t2 = t1 + rng.randint(5, 10)
    d1 = rng.choice([5, 8, 10])
    d2 = d1 + rng.choice([5, 7, 10])
    header = (f"UNIT_PRICE = {unit}\n"
              f"TIERS = [({t2}, {d2}), ({t1}, {d1})]  # (minimum quantity, percent off), highest first\n\n\n")
    body = ('def order_total(quantity):\n    """Total price for `quantity` items after the best tier discount."""\n'
            '    total = quantity * UNIT_PRICE\n    for minimum, percent in TIERS:\n'
            '        if quantity {cmp} minimum:\n            total = total * (100 - percent) / 100\n'
            '            break\n    return {ret}\n')
    buggy = header + body.format(cmp=">", ret="int(total * 100) / 100")
    fixed = header + body.format(cmp=">=", ret="round(total, 2)")

    def total(src, quantity):
        ns = {}
        exec(src, ns)
        return ns["order_total"](quantity)

    def case(group, q):
        return _case(group, f"from pricing import order_total\nresult = order_total({q})", total(fixed, q))

    edge = [q for q in range(0, 40) if q not in (t1, t2) and total(buggy, q) != total(fixed, q)][:2]
    safe = [q for q in range(0, 40) if q not in (t1, t2) and total(buggy, q) == total(fixed, q)][:3]
    cases = [case("target", q) for q in [t1, t2] + edge] + [case("regression", q) for q in safe]
    return TaskInstance(
        task_id="bulk_discount", title="Fix the bulk discount (seeded)", difficulty="medium",
        instructions=("`pricing.order_total` should give an order the discount of the highest tier whose "
                      "minimum quantity it reaches (reaching the minimum counts), and round the total to "
                      "2 decimal places. The tiers are defined at the top of the file."),
        files={"pricing.py": buggy}, cases=cases, reference={"pricing.py": fixed},
    )


def box_packing_task(seed):
    from .box_packing import box_packing_task as _bpt
    return _bpt(seed)


def blinkit_rebalance_task(seed):
    from .blinkit_rebalance import blinkit_rebalance_task as _brt
    return _brt(seed)


TASKS = {
    "box_packing": box_packing_task,
    "blinkit_rebalance": blinkit_rebalance_task,
    "slugify": slugify_task,
    "parse_duration": duration_task,
    "merge_intervals": intervals_task,
    "lru_cache": lru_task,
    "bulk_discount": discount_task,
}
DEMO_TASK = "box_packing"


def build_task(task_id, seed):
    if task_id not in TASKS:
        raise KeyError(f"unknown task '{task_id}'. Available: {', '.join(TASKS)}")
    return TASKS[task_id](seed)

