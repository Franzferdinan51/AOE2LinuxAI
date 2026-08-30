"""Tool-schema definitions for the executor's tool-use loop."""

def _click_schema(description: str) -> dict:
    return {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate on game screen"},
            "y": {"type": "integer", "description": "Y coordinate on game screen"},
            "target_class": {"type": "string", "description": "Entity class to target nearest of, e.g. 'sheep'"},
            "intent": {"type": "string", "description": description},
        },
        "required": ["x", "y", "intent"],
        "additionalProperties": False,
    }


_ACTION_TOOLS: list[dict] = [
    {
        "name": "click",
        "description": "Left click at screen coordinates. Use for building placement and UI interaction.",
        "input_schema": _click_schema("What this click does"),
    },
    {
        "name": "right_click",
        "description": "Right click at screen coordinates. Use for resource gathering, setting gather points, and unit commands.",
        "input_schema": _click_schema("What this right click does"),
    },
    {
        "name": "press",
        "description": "Press a keyboard key. Use for hotkeys, queuing units, opening build menus.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key to press, e.g. 'h', 'q', '.', ','"},
                "rescan": {"type": "boolean", "description": "Take fresh screenshot+detection after this key press"},
                "modifiers": {"type": "array", "items": {"type": "string"}, "description": "Modifier keys e.g. ['ctrl']"},
                "intent": {"type": "string", "description": "What this key press does"},
            },
            "required": ["key", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "drag",
        "description": "Drag mouse from start to end position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_x": {"type": "integer"}, "start_y": {"type": "integer"},
                "end_x": {"type": "integer"}, "end_y": {"type": "integer"},
                "intent": {"type": "string"},
            },
            "required": ["start_x", "start_y", "end_x", "end_y", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "wait",
        "description": "Wait for a duration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ms": {"type": "integer", "description": "Milliseconds to wait (0-5000)"},
                "intent": {"type": "string"},
            },
            "required": ["ms", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "scroll",
        "description": "Scroll mouse wheel for zoom in/out.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clicks": {"type": "integer", "description": "Positive=zoom in, negative=zoom out"},
                "intent": {"type": "string"},
            },
            "required": ["clicks", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "detect",
        "description": "Request full SAHI detection scan. SLOW (~5-10s) — only use when target_class keeps failing.",
        "input_schema": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "build",
        "description": "Composite: select a villager, open a build menu, press building_key, place the building. Menus: q=economic, w=military, v=advanced. Placement is chosen by the executor AFTER the camera settles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "enum": ["q", "w", "v"], "description": "Build menu"},
                "building_key": {"type": "string", "description": "Key within that menu"},
                "intent": {"type": "string"},
            },
            "required": ["menu", "building_key", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "research",
        "description": "Composite: go to the building that researches this technology, then press its panel key. Named, not keyed — the executor owns the hotkeys.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tech": {"type": "string", "enum": ["castle_age", "loom", "wheelbarrow", "horse_collar", "double_bit_axe", "gold_mining"]},
                "intent": {"type": "string"},
            },
            "required": ["tech", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_villager",
        "description": "Composite: select idle villager (press .) then right_click target. target_class only (e.g. 'sheep', 'tree').",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_class": {"type": "string"},
                "intent": {"type": "string"},
            },
            "required": ["target_class", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_all_idle",
        "description": "Composite: select ALL idle villagers (Shift-.) then right_click target. Dispatches every idle villager at once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_class": {"type": "string"},
                "intent": {"type": "string"},
            },
            "required": ["target_class", "intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "queue_villager",
        "description": "Composite: go to TC (press h), then queue villager (press q).",
        "input_schema": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
    {
        "name": "reassign_villager",
        "description": "Composite: jump to a work site (camera hotkey), pick a working villager of from_job, then build.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_job": {"type": "string", "enum": ["food", "wood", "gold", "stone"], "description": "Source job to pull a worker from"},
                "building_key": {"type": "string", "description": "Key in econ menu (default 'a' = Farm)"},
                "intent": {"type": "string"},
            },
            "required": ["from_job", "intent"],
            "additionalProperties": False,
        },
    },
]


def to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic-shaped tools to OpenAI Chat Completions shape.

    Renames input_schema → parameters and strips the human description; the
    tool descriptions are reused verbatim.
    """
    converted: list[dict] = []
    for tool in anthropic_tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
        )
    return converted


__all__ = ["_ACTION_TOOLS", "to_openai_tools"]
