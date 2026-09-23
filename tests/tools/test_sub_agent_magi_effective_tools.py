"""Load nested tools from the pinned Core route before starting each loop."""

import asyncio
import copy
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import magi_decision_support
import sub_agent
from open_webui.utils import terminals as core_terminals


def model(model_id, terminal):
    return {"id": model_id, "info": {"meta": {"capabilities": {"terminal": terminal}}}}


async def invoke(tool, mode, request, parent, metadata, user, **kwargs):
    args = {
        "__request__": request, "__model__": parent, "__metadata__": metadata,
        "__user__": user, "__id__": "self", **kwargs,
    }
    if mode == "magi":
        return await tool.magi_decide(proposition="Choose", option_a="A", option_b="B", **args)
    if mode == "parallel":
        return await tool.run_parallel_sub_agents(
            tasks=[{"description": f"Task {i}", "prompt": f"Task {i}"} for i in range(2)], **args,
        )
    return await tool.run_sub_agent(description="Task", prompt="Task", **args)


@pytest.mark.parametrize("mode", ["single", "parallel", "magi"])
@pytest.mark.parametrize("route", ["direct", "arena"])
@pytest.mark.parametrize("allowed", [False, True])
async def test_effective_route_controls_loading_and_every_provider_request(
    monkeypatch, mock_user, mode, route, allowed,
):
    from open_webui.models.config import Config

    module = magi_decision_support if mode == "magi" else sub_agent
    nominal = model("nominal", not allowed)
    effective = model("nominal" if route == "direct" else "child", allowed)
    if route == "arena":
        nominal.update(owned_by="arena")
        nominal["info"]["meta"]["model_ids"] = ["child"]
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(MODELS={"nominal": nominal, "child": effective})),
        state=SimpleNamespace(direct=route == "direct", model=effective),
        body=AsyncMock(return_value=b"{}"),
    )
    metadata = {"terminal_id": "terminal", "tool_ids": ["regular"], "filter_ids": []}
    original_metadata = copy.deepcopy(metadata)
    loaded_models, terminal_models, agents_models, payloads = [], [], [], []
    selections = []

    def choose(candidates):
        selections.append(list(candidates))
        return candidates[0]

    monkeypatch.setattr(module.random, "choice", choose)

    async def noop():
        return "ok"

    async def get_tools(**kwargs):
        return {"noop": {"spec": {"name": "noop", "parameters": {"type": "object", "properties": {}}}, "callable": noop}}

    async def get_builtin_tools(**kwargs):
        loaded_models.append(kwargs["model"])
        return {}

    async def get_terminal_tools(**kwargs):
        terminal_models.append(kwargs["extra_params"]["__model__"])
        return ({"run_command": {"spec": {"name": "run_command"}, "type": "terminal", "tool_id": "terminal:terminal"}}, "Terminal instructions")

    async def get_agents(request, user, metadata, extra_params):
        agents_models.append(extra_params["__model__"])
        await asyncio.sleep(0)
        return "# AGENTS.md\nTerminal project instructions"

    async def config_get(key, default=None):
        return [{"id": "terminal"}] if key == "terminal_server.connections" else default

    async def completion(**kwargs):
        body = copy.deepcopy(kwargs["form_data"])
        payloads.append(body)
        if not any(message["role"] == "tool" for message in body["messages"]):
            return {"choices": [{"message": {"content": "", "tool_calls": [{
                "id": "noop-call", "type": "function", "function": {"name": "noop", "arguments": "{}"},
            }]}}]}
        return {"choices": [{"message": {"content": '{"vote":"A","reasoning":"done"}'}}]}

    tool_utils = ModuleType("open_webui.utils.tools")
    tool_utils.get_tools = get_tools
    tool_utils.get_builtin_tools = get_builtin_tools
    tool_utils.get_terminal_tools = get_terminal_tools
    tool_utils.get_updated_tool_function = lambda function, extra_params: function
    chat = ModuleType("open_webui.utils.chat")
    chat.generate_chat_completion = completion
    monkeypatch.setitem(sys.modules, "open_webui.utils.tools", tool_utils)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat)
    monkeypatch.setitem(sys.modules, "open_webui.utils.terminals", core_terminals)
    monkeypatch.setattr(core_terminals, "get_terminal_agents_md", get_agents)
    monkeypatch.setattr(Config, "get", staticmethod(config_get))
    monkeypatch.setattr(magi_decision_support, "generate_single_completion", AsyncMock(return_value="Summary"))
    tool = module.Tools()
    tool.valves.APPLY_INLET_FILTERS = False
    tool.valves.MAX_ITERATIONS = 1
    if mode != "magi":
        tool.valves.ENABLE_CONTEXT_COMPACTION = False
        tool.valves.LARGE_TOOL_RESULT_MODE = "raw"
    await invoke(tool, mode, request, nominal, metadata, mock_user)

    loops = {"single": 1, "parallel": 2, "magi": 3}[mode]
    assert loaded_models == [effective]
    assert terminal_models == ([effective] if allowed else [])
    assert agents_models == ([effective] if allowed else [])
    assert len(selections) == (loops if route == "arena" else 0)
    assert len(payloads) == 2 * loops
    assert metadata == original_metadata
    initial_payloads = [body for body in payloads if not any(m["role"] == "tool" for m in body["messages"])]
    assert len(initial_payloads) == loops
    for body in initial_payloads:
        names = {entry["function"]["name"] for entry in body["tools"]}
        assert "noop" in names
        assert ("run_command" in names) is allowed
    for body in payloads:
        assert body["model"] == effective["id"]
        assert all(m["content"] == "ok" for m in body["messages"] if m["role"] == "tool")
        assert ("Terminal instructions" in json.dumps(body["messages"])) is allowed
        agents = [m for m in body["messages"] if m["role"] == "user" and m["content"].startswith("# AGENTS.md")]
        assert len(agents) == int(allowed)


@pytest.mark.parametrize("mode", ["parallel", "magi"])
async def test_branches_share_effective_tools_but_keep_routes_and_metadata_separate(
    monkeypatch, mock_user, mode,
):
    module = magi_decision_support if mode == "magi" else sub_agent
    allowed, denied = model("allowed", True), model("denied", False)
    arena = {"id": "arena", "owned_by": "arena", "info": {"meta": {"model_ids": ["allowed", "denied"]}}}
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(MODELS={"arena": arena, "allowed": allowed, "denied": denied})))
    choices = iter(["allowed", "denied", "allowed"])
    monkeypatch.setattr(module.random, "choice", lambda _candidates: next(choices))
    loaded, closed, calls, pipelines, skills = [], [], [], [], []
    live_clients, cleanup_checks = [], []
    resolve = module.resolve_model_filter_pipeline
    cleanup_clients = module.cleanup_mcp_clients

    class FakeClient:
        async def connect(self):
            self.owner = asyncio.current_task()
            live_clients.append(self)

        async def disconnect(self):
            cleanup_checks.append((self.owner is asyncio.current_task(), live_clients[-1] is self))
            live_clients.remove(self)

    async def resolve_pipeline(*args):
        pipeline = await resolve(*args)
        pipeline["context"] = object()
        pipelines.append(pipeline)
        return pipeline

    async def load(**kwargs):
        target = kwargs["model"]
        extra = kwargs["extra_params"]
        extra["__metadata__"]["terminal_id"] = "resolved-terminal"
        if target is allowed:
            extra["__terminal_system_prompt__"] = "Terminal instructions"
            extra["__direct_tool_server_system_prompts__"] = ["Direct instructions"]
            extra["__terminal_agents_md__"] = "Project instructions"
        client = FakeClient()
        await client.connect()
        clients = {"mcp": client}
        tools = {"catalogue_model": target["id"]}
        loaded.append((target, tools, clients))
        await asyncio.sleep(0)
        return tools, clients

    async def register(tools, request, extra):
        skills.append(tools)

    async def loop(**kwargs):
        assert closed == []
        calls.append(kwargs)
        assert kwargs["extra_params"]["__metadata__"].pop("terminal_id") == "resolved-terminal"
        await asyncio.sleep(0)
        return '{"vote":"A","reasoning":"done"}'

    async def cleanup(*clients):
        closed.extend(clients)
        await cleanup_clients(*clients)

    monkeypatch.setattr(module, "resolve_model_filter_pipeline", resolve_pipeline)
    monkeypatch.setattr(module, "load_sub_agent_tools" if mode == "parallel" else "build_tools_dict", load)
    monkeypatch.setattr(module, "run_sub_agent_loop" if mode == "parallel" else "run_agent_loop", loop)
    monkeypatch.setattr(module, "register_view_skill", register)
    monkeypatch.setattr(module, "extract_skill_manifest", lambda _messages: "<available_skills />")
    monkeypatch.setattr(module, "cleanup_mcp_clients", cleanup)
    monkeypatch.setattr(magi_decision_support, "generate_single_completion", AsyncMock(return_value="Summary"))
    tool = module.Tools()
    tool.valves.APPLY_INLET_FILTERS = False
    metadata = {"filter_ids": []}
    if mode == "parallel":
        await tool.run_parallel_sub_agents(
            tasks=[{"description": f"Task {i}", "prompt": f"Task {i}"} for i in range(3)],
            __request__=request, __model__=arena, __metadata__=metadata, __user__=mock_user,
        )
    else:
        await invoke(tool, mode, request, arena, metadata, mock_user)

    assert [entry[0] for entry in loaded] == [allowed, denied]
    assert len(pipelines) == len(calls) == 3
    assert len({id(pipeline["context"]) for pipeline in pipelines}) == 3
    assert len({id(call["extra_params"]["__metadata__"]) for call in calls}) == 3
    assert skills == [entry[1] for entry in loaded]
    for call in calls:
        pipeline = call["filter_pipeline"]
        assert any(pipeline is item for item in pipelines)
        extra = call["extra_params"]
        expected_tools = next(entry[1] for entry in loaded if entry[0] is pipeline["model"])
        assert call["tools_dict"] is expected_tools
        assert extra["__model__"] is pipeline["model"]
        enabled = pipeline["model"] is allowed
        assert (extra.get("__terminal_system_prompt__") == "Terminal instructions") is enabled
        assert (extra.get("__direct_tool_server_system_prompts__") == ["Direct instructions"]) is enabled
        assert (extra.get("__terminal_agents_md__") == "Project instructions") is enabled
    assert closed == [entry[2] for entry in loaded]
    assert cleanup_checks == [(True, True), (True, True)]
    assert live_clients == []
    assert metadata == {"filter_ids": []}


@pytest.mark.parametrize("mode", ["parallel", "magi"])
async def test_branches_keep_separate_arena_routes_and_close_mcp_after_setup_failure(
    monkeypatch, mock_user, mode,
):
    module = magi_decision_support if mode == "magi" else sub_agent
    allowed, denied = model("allowed", True), model("denied", False)
    arena = {"id": "arena", "owned_by": "arena", "info": {"meta": {"model_ids": ["allowed", "denied"]}}}
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(MODELS={"arena": arena, "allowed": allowed, "denied": denied})))
    choices = iter(["allowed", "denied", "allowed"])
    monkeypatch.setattr(module.random, "choice", lambda _candidates: next(choices))
    loaded, closed, skills = [], [], []

    async def load(**kwargs):
        target = kwargs["model"]
        extra = kwargs["extra_params"]
        assert extra["__model__"] is target
        extra["__metadata__"]["branch"] = len(loaded)
        clients = {"mcp": object()}
        loaded.append((target, extra, clients))
        await asyncio.sleep(0)
        return {}, clients

    async def register(tools, request, extra):
        skills.append(extra)
        if extra["__model__"] is denied:
            raise RuntimeError("Skill setup failed")

    async def cleanup(*clients):
        closed.extend(clients)

    monkeypatch.setattr(module, "load_sub_agent_tools" if mode == "parallel" else "build_tools_dict", load)
    monkeypatch.setattr(module, "register_view_skill", register)
    monkeypatch.setattr(module, "extract_skill_manifest", lambda _messages: "<available_skills />")
    monkeypatch.setattr(module, "cleanup_mcp_clients", cleanup)
    monkeypatch.setattr(magi_decision_support, "generate_single_completion", AsyncMock(return_value="Summary"))
    tool = module.Tools()
    tool.valves.APPLY_INLET_FILTERS = False
    metadata = {"filter_ids": []}
    with pytest.raises(RuntimeError, match="Skill setup failed"):
        await invoke(tool, mode, request, arena, metadata, mock_user)

    expected = [allowed, denied]
    assert [entry[0] for entry in loaded] == expected
    assert [entry[1]["__metadata__"]["branch"] for entry in loaded] == list(range(len(expected)))
    assert skills == [entry[1] for entry in loaded]
    assert closed == [entry[2] for entry in loaded]
    assert metadata == {"filter_ids": []}


@pytest.mark.parametrize("mode", ["parallel", "magi"])
@pytest.mark.parametrize("outcome", ["cancel", "error"])
async def test_failure_closes_shared_mcp_clients_after_all_branches(monkeypatch, mock_user, mode, outcome):
    module = magi_decision_support if mode == "magi" else sub_agent
    target = model("target", True)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(MODELS={"target": target})))
    clients, closed = [], []
    started = 0
    finished = 0
    all_started = asyncio.Event()
    branch_count = 3 if mode == "magi" else 2

    async def load(**kwargs):
        branch_clients = {"mcp": object()}
        clients.append(branch_clients)
        return {}, branch_clients

    async def loop(**kwargs):
        nonlocal started, finished
        started += 1
        if started == branch_count:
            all_started.set()
        try:
            if outcome == "error":
                await all_started.wait()
                raise RuntimeError("Model request failed")
            await asyncio.Future()
        finally:
            finished += 1

    async def cleanup(*branch_clients):
        assert finished == branch_count
        closed.extend(branch_clients)

    monkeypatch.setattr(module, "load_sub_agent_tools" if mode == "parallel" else "build_tools_dict", load)
    monkeypatch.setattr(module, "run_sub_agent_loop" if mode == "parallel" else "run_agent_loop", loop)
    monkeypatch.setattr(module, "cleanup_mcp_clients", cleanup)
    monkeypatch.setattr(magi_decision_support, "generate_single_completion", AsyncMock(return_value="Summary"))
    tool = module.Tools()
    tool.valves.APPLY_INLET_FILTERS = False
    task = asyncio.create_task(invoke(tool, mode, request, target, {}, mock_user))
    try:
        await asyncio.wait_for(all_started.wait(), timeout=5)
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert len(clients) == 1
    assert closed == clients
