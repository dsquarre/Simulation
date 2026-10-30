"""Tool definitions shown to the model, plus argument checking."""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List every file in the project.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the project.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "Path relative to the project directory."}}}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a text file in the project.",
        "parameters": {"type": "object", "required": ["path", "content"], "properties": {
            "path": {"type": "string", "description": "Path relative to the project directory."},
            "content": {"type": "string", "description": "The full new contents of the file."}}}}},
    {"type": "function", "function": {
        "name": "run_python", "description": "Run a snippet of Python code in the project directory and return its output.",
        "parameters": {"type": "object", "required": ["code"], "properties": {
            "code": {"type": "string", "description": "Python source code to run."}}}}},
    {"type": "function", "function": {
        "name": "plant_season", "description": "Plant crop allocations for the current season in the multi-season crop POMDP. Advances the simulation to the next season and returns season profit, revealed market prices, and next season's budget/conditions.",
        "parameters": {"type": "object", "required": ["allocation"], "properties": {
            "allocation": {"type": "string", "description": "JSON object string mapping crop name to integer cell count, e.g. '{\"rice\": 10, \"cotton\": 5}'"}}}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Call this when the task is complete.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

SPECS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
TOOL_NAMES = list(SPECS)


def check_args(name, args):
    """Return an error message if `args` don't match the tool's schema, else None."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    spec = SPECS[name]
    for key in spec.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    for key, value in args.items():
        if key not in spec["properties"]:
            return f"unexpected argument '{key}'"
        if not isinstance(value, str):
            return f"argument '{key}' must be a string"
    return None
