"""Load terminal context for the same effective model that receives the request."""

import copy
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import multi_model_council as council
from open_webui.utils import terminals as core_terminals


@pytest.mark.parametrize("route", ["arena", "direct"])
@pytest.mark.parametrize("enabled", [True, False])
async def test_council_resolves_route_before_loading_terminal_context(monkeypatch, route, enabled):
    from open_webui.utils import tools as core_tools

    def model(model_id, terminal):
        return {"id": model_id, "info": {"meta": {"capabilities": {"terminal": terminal}}}}

    target = model("target", enabled)
    nominal = model("selected", not enabled)
    models = {"selected": nominal, "other": model("other", False)}
    if route == "arena":
        nominal["owned_by"] = "arena"
        nominal["info"]["meta"]["model_ids"] = ["target"]
        models["target"] = target
    else:
        target["id"] = "selected"
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(MODELS=models)),
        state=SimpleNamespace(direct=route == "direct", model=target),
        body=AsyncMock(return_value=b"{}"),
    )
    loaded = []
    agents_read = []
    resolved = []
    sent = []

    async def noop():
        return "ok"

    regular_tool = {"spec": {"name": "noop"}, "callable": noop, "type": "tool"}

    async def terminal_tools(**kwargs):
        loaded.append(kwargs["extra_params"]["__model__"])
        return {"run_command": {**regular_tool, "spec": {"name": "run_command"}, "type": "terminal"}}, "Terminal prompt"

    async def agents(*args):
        agents_read.append(args[3]["__model__"])
        return "# AGENTS.md\n\nProject rules"

    resolve = council.resolve_model_filter_pipeline

    async def resolve_once(*args):
        resolved.append(args[2])
        return await resolve(*args)

    async def complete(**kwargs):
        body = copy.deepcopy(kwargs["form_data"])
        sent.append(body)
        if sum(item["model"] == body["model"] for item in sent) == 1:
            return {"choices": [{"message": {"content": "", "tool_calls": [{
                "id": "noop", "type": "function", "function": {"name": "noop", "arguments": "{}"},
            }]}}]}
        return {"choices": [{"message": {"content": json.dumps({"vote": "A", "reasoning": "ok"})}}]}

    chat = ModuleType("open_webui.utils.chat")
    chat.generate_chat_completion = complete
    monkeypatch.setitem(sys.modules, chat.__name__, chat)
    monkeypatch.setitem(sys.modules, "open_webui.utils.terminals", core_terminals)
    monkeypatch.setattr(core_terminals, "get_terminal_agents_md", agents)
    monkeypatch.setattr(core_tools, "get_terminal_tools", terminal_tools)
    monkeypatch.setattr(core_tools, "get_tools", AsyncMock(return_value={"noop": regular_tool}))
    monkeypatch.setattr(council, "resolve_model_filter_pipeline", resolve_once)
    monkeypatch.setattr(council, "get_available_models", AsyncMock(return_value=list(models.values())))
    tool = council.Tools()
    tool.valves.APPLY_INLET_FILTERS = False
    tool.valves.MAX_ITERATIONS = 1
    result = await tool.council_decide(
        proposition="Choose", option_a="A", option_b="B", models="selected,other",
        __request__=request, __model__=nominal,
        __user__={"id": "u1", "role": "user"},
        __metadata__={"tool_ids": ["regular"], "terminal_id": "term"},
    )
    assert json.loads(result)["decision"] == "A"
    assert resolved == ["selected", "other"]
    assert loaded == ([target] if enabled else [])
    assert agents_read == loaded
    assert len(sent) == 4
    for body in sent:
        is_enabled = body["model"] == target["id"] and enabled
        assert ("Project rules" in str(body["messages"])) == is_enabled
        assert ("Terminal prompt" in str(body["messages"])) == is_enabled
        if "tools" in body:
            names = {item["function"]["name"] for item in body["tools"]}
            assert "noop" in names
            assert ("run_command" in names) == is_enabled
