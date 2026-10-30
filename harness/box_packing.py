"""Box Packing Problem with Incompatibility Constraints (NP-Hard).

Given:
  - N items (IDs 0 to N-1), each with an integer weight w_i
  - Uniform box capacity C (maximum allowed weight per box)
  - An undirected conflict graph G = (V, E) where edge (u, v) means
    items u and v CANNOT be placed in the same box.

Objective:
  Pack all N items into the minimum possible number of boxes such that:
    1. Every item appears in exactly one box.
    2. Sum of weights in any box <= C.
    3. No two conflicting items share a box.

Computational Guarantee:
  For N < 20 (N in [11, 15]), the Branch-and-Bound solver finds the
  exact optimal solution in under 50 milliseconds (< 1.0 second required).
"""
import json
import random
import textwrap
import time

from .tasks import Case, TaskInstance, _case


def solve_box_packing(weights: list[int], capacity: int, conflicts: list[list[int]]) -> tuple[int, list[list[int]]]:
    """Exact branch-and-bound solver for box packing with conflict constraints.
    
    Returns (optimal_box_count, optimal_boxes).
    Runs internally in < 25ms for N <= 16.
    """
    n = len(weights)
    adj = {i: set() for i in range(n)}
    for u, v in conflicts:
        adj[u].add(v)
        adj[v].add(u)

    # Internal ordering: descending weight to prune branches aggressively
    # (Note: this is strictly internal to the solver; the public problem state
    # retains the original unsorted order).
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)

    # Greedy First-Fit-Decreasing with conflict avoidance for initial upper bound
    def get_initial_upper_bound():
        bins = []  # list of (current_weight, set_of_items)
        for item in order:
            w = weights[item]
            placed = False
            for b_idx in range(len(bins)):
                bw, bitems = bins[b_idx]
                if bw + w <= capacity and not (adj[item] & bitems):
                    bins[b_idx] = (bw + w, bitems | {item})
                    placed = True
                    break
            if not placed:
                bins.append((w, {item}))
        return len(bins), [sorted(bitems) for _, bitems in bins]

    best_bin_count, best_solution = get_initial_upper_bound()

    order_weights = [weights[i] for i in order]
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + order_weights[i]

    current_bins_weight = []
    current_bins_items = []

    def backtrack(idx):
        nonlocal best_bin_count, best_solution
        if idx == n:
            if len(current_bins_weight) < best_bin_count:
                best_bin_count = len(current_bins_weight)
                best_solution = [sorted(b) for b in current_bins_items]
            return

        # Continuous L1 lower bound pruning
        rem_weight = suffix_sum[idx]
        open_bins = len(current_bins_weight)
        remaining_capacity_in_open = sum(capacity - w for w in current_bins_weight)
        min_extra_bins = max(0, (rem_weight - remaining_capacity_in_open + capacity - 1) // capacity)
        if open_bins + min_extra_bins >= best_bin_count:
            return

        item = order[idx]
        w = weights[item]
        item_conflicts = adj[item]

        # Try placing item in already opened bins
        for b in range(open_bins):
            if current_bins_weight[b] + w <= capacity and not (item_conflicts & current_bins_items[b]):
                current_bins_weight[b] += w
                current_bins_items[b].add(item)
                backtrack(idx + 1)
                current_bins_items[b].remove(item)
                current_bins_weight[b] -= w

        # Try opening a new bin only if it doesn't exceed current best
        if open_bins + 1 < best_bin_count:
            current_bins_weight.append(w)
            current_bins_items.append({item})
            backtrack(idx + 1)
            current_bins_items.pop()
            current_bins_weight.pop()

    backtrack(0)
    return best_bin_count, best_solution


def generate_box_packing_instance(seed):
    """Generate problem instance deterministically from seed.
    
    Changing the seed produces different N, different box capacity,
    different weight array, and different conflict graph.
    """
    # Use seed as integer
    try:
        s_int = int(seed)
    except (ValueError, TypeError):
        s_int = abs(hash(str(seed))) % (2**31)

    rng = random.Random(s_int)

    # 1. Vary N between 11 and 15 (always N < 20)
    n = rng.randint(11, 15)

    # 2. Vary box capacity between 90 and 140
    capacity = rng.randint(90, 140)

    # 3. Vary weights (unsorted, arbitrary order)
    weights = [rng.randint(18, int(capacity * 0.55)) for _ in range(n)]

    # 4. Vary conflict graph
    num_conflicts = rng.randint(3, max(4, n - 3))
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    conflicts = sorted(rng.sample(all_pairs, num_conflicts))

    # 5. Exact optimal solution via branch-and-bound
    t0 = time.perf_counter()
    optimal_box_count, optimal_boxes = solve_box_packing(weights, capacity, conflicts)
    solve_time_ms = (time.perf_counter() - t0) * 1000

    return {
        "seed": seed,
        "num_items": n,
        "box_capacity": capacity,
        "weights": weights,  # original unsorted order
        "conflicts": conflicts,
        "optimal_box_count": optimal_box_count,
        "optimal_boxes": optimal_boxes,
        "solve_time_ms": solve_time_ms,
    }


def make_box_packing_cases(n: int, weights: list[int], capacity: int, conflicts: list[list[int]], optimal_boxes: int):
    """Create hidden test cases for programmatic verification in the harness/sandbox."""
    check_script = textwrap.dedent(f"""
        import json
        N = {n}
        weights = {weights}
        capacity = {capacity}
        conflicts = {conflicts}
        optimal_boxes = {optimal_boxes}

        def verify():
            try:
                with open("solution.json", "r") as f:
                    data = json.load(f)
            except Exception:
                return {{"feasible": False, "optimal": False, "tight": False, "reason": "file_error"}}

            if not isinstance(data, dict) or "boxes" not in data or not isinstance(data["boxes"], list):
                return {{"feasible": False, "optimal": False, "tight": False, "reason": "schema_error"}}

            boxes = data["boxes"]
            if len(boxes) == 0:
                return {{"feasible": False, "optimal": False, "tight": False, "reason": "empty_boxes"}}

            seen = set()
            for b_idx, box in enumerate(boxes):
                if not isinstance(box, list):
                    return {{"feasible": False, "optimal": False, "tight": False, "reason": f"box_{{b_idx}}_not_list"}}
                b_weight = 0
                b_set = set()
                for item in box:
                    if not isinstance(item, int) or item < 0 or item >= N:
                        return {{"feasible": False, "optimal": False, "tight": False, "reason": f"invalid_item_{{item}}"}}
                    if item in seen:
                        return {{"feasible": False, "optimal": False, "tight": False, "reason": f"duplicate_item_{{item}}"}}
                    seen.add(item)
                    b_set.add(item)
                    b_weight += weights[item]
                if b_weight > capacity:
                    return {{"feasible": False, "optimal": False, "tight": False, "reason": f"capacity_overflow_box_{{b_idx}}"}}
                for u, v in conflicts:
                    if u in b_set and v in b_set:
                        return {{"feasible": False, "optimal": False, "tight": False, "reason": f"conflict_{{u}}_{{v}}"}}

            if len(seen) != N:
                return {{"feasible": False, "optimal": False, "tight": False, "reason": "missing_items"}}

            is_optimal = (len(boxes) <= optimal_boxes)
            is_tight = (len(boxes) <= optimal_boxes + 1)
            return {{"feasible": True, "optimal": is_optimal, "tight": is_tight, "num_boxes": len(boxes)}}

        v_res = verify()
    """)

    # Target 1: Feasibility (all items present, no capacity overflow, no conflicts)
    case_feasible = _case(
        "target",
        check_script + "\nresult = v_res.get('feasible', False)",
        True,
    )

    # Target 2: Tight packing (feasible AND within 1 box of optimal)
    case_tight = _case(
        "target",
        check_script + "\nresult = (v_res.get('feasible', False) and v_res.get('tight', False))",
        True,
    )

    # Target 3: Exact optimal box count
    case_optimal = _case(
        "target",
        check_script + "\nresult = (v_res.get('feasible', False) and v_res.get('optimal', False))",
        True,
    )

    # Regression: Solution must be feasible
    case_regression = _case(
        "regression",
        check_script + "\nresult = v_res.get('feasible', False)",
        True,
    )

    return [case_feasible, case_tight, case_optimal, case_regression]


def box_packing_task(seed) -> TaskInstance:
    """Build the box packing TaskInstance."""
    inst = generate_box_packing_instance(seed)

    n = inst["num_items"]
    capacity = inst["box_capacity"]
    weights = inst["weights"]
    conflicts = inst["conflicts"]
    optimal_boxes = inst["optimal_boxes"]
    optimal_box_count = inst["optimal_box_count"]

    # Formatted description without leaking optimizer shortcuts
    items_desc = "\n".join(f"  Item {i}: weight {weights[i]}" for i in range(n))
    conflicts_desc = "\n".join(f"  Item {u} and Item {v} CANNOT share a box" for u, v in conflicts)

    instructions = textwrap.dedent(f"""
        Solve the Box Packing Problem with Incompatibility Constraints:

        PROBLEM STATE:
        - Total items: N = {n} (IDs 0 to {n-1})
        - Max capacity per box: {capacity}
        - Items and their weights:
        {items_desc}
        - Conflict constraints (items that cannot be placed together in the same box):
        {conflicts_desc}

        OBJECTIVE:
        Pack all {n} items into the minimum possible number of boxes such that:
        1. Every item from 0 to {n-1} is placed in exactly one box (no missing items, no duplicates).
        2. The sum of weights in any box does not exceed {capacity}.
        3. No two conflicting items share a box.

        ACTION / OUTPUT:
        Write your solution to `solution.json` as a JSON object with a single key "boxes",
        which maps to a list of lists of item IDs. For example:
        {{
          "boxes": [
            [0, 2],
            [1, 3],
            ...
          ]
        }}
        After writing `solution.json`, call the `finish` tool.
    """).strip()

    problem_json_content = json.dumps({
        "task": "box_packing",
        "seed": seed,
        "num_items": n,
        "box_capacity": capacity,
        "items": [{"id": i, "weight": weights[i]} for i in range(n)],
        "conflicts": conflicts,
    }, indent=2)

    initial_solution = json.dumps({"boxes": []}, indent=2)
    reference_solution = json.dumps({"boxes": optimal_boxes}, indent=2)

    cases = make_box_packing_cases(n, weights, capacity, conflicts, optimal_box_count)

    return TaskInstance(
        task_id="box_packing",
        title=f"Box packing with conflicts (N={n}, C={capacity}, seed={seed})",
        difficulty="hard",
        instructions=instructions,
        files={
            "problem.json": problem_json_content,
            "solution.json": initial_solution,
        },
        cases=cases,
        reference={
            "solution.json": reference_solution,
        },
    )
