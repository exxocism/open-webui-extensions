"""MCP acquisition failures must unwind real cancel scopes in their owner task."""

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest

from open_webui.utils import access_control
from open_webui.utils.mcp import client as core_mcp
from owui_ext.shared import mcp_tools, tool_loader


@pytest.fixture
def mcp_lifecycle(monkeypatch):
    clients = []
    closed = []
    state = SimpleNamespace(
        fail_at=None, error=None, clients=clients, closed=closed, emitter=AsyncMock(),
    )

    async def checkpoint(location):
        if state.fail_at == location:
            if state.error is not None:
                raise state.error
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

    class ScopedMCPClient(core_mcp.MCPClient):
        async def connect(self, url, headers=None):
            self.name = url
            self.owner = asyncio.current_task()
            self.scope = anyio.CancelScope()
            self.exit_stack = AsyncExitStack()
            self.exit_stack.enter_context(self.scope)
            self.exit_stack.push_async_callback(checkpoint, f"{url}:disconnect")
            self.session = object()
            clients.append(self)
            await checkpoint(f"{url}:connect")

        async def list_tool_specs(self):
            await checkpoint(f"{self.name}:list")
            return []

        async def disconnect(self):
            closed.append((self.name, self.owner is asyncio.current_task()))
            # Real Core disconnect clears its handles and can suppress invalid
            # AnyIO exits; checking scope state detects those hidden failures.
            await super().disconnect()

    connections = [
        {"type": "mcp", "info": {"id": name}, "url": name}
        for name in ("first", "second")
    ]

    async def headers(**kwargs):
        await checkpoint(f"{kwargs['server_id']}:headers")
        return {}

    monkeypatch.setattr(core_mcp, "MCPClient", ScopedMCPClient)
    monkeypatch.setattr(mcp_tools, "_get_tool_server_connections", AsyncMock(return_value=connections))
    monkeypatch.setattr(mcp_tools, "_build_mcp_headers_with_core", headers)
    monkeypatch.setattr(access_control, "has_connection_access", AsyncMock(return_value=True))

    async def resolve(**kwargs):
        return await mcp_tools.resolve_mcp_tools(
            request=SimpleNamespace(), user=SimpleNamespace(id="user"),
            mcp_tool_ids=["server:mcp:first", "server:mcp:second"],
            extra_params={"__event_emitter__": state.emitter}, metadata={},
        )

    async def cancelled_acquisition(acquire):
        observed = {}

        async def worker():
            try:
                await acquire()
            finally:
                observed["closed"] = list(closed)
                observed["scope_active"] = [client.scope._active for client in clients]
                # A failing regression must not leave scopes in pytest's task.
                # Capture first, then rescue any missing cleanup in this owner.
                for client in reversed(clients):
                    if client.exit_stack is not None:
                        try:
                            await client.disconnect()
                        except BaseException:
                            pass

        with pytest.raises(asyncio.CancelledError):
            await asyncio.create_task(worker())
        expected = [(client.name, True) for client in reversed(clients)]
        assert observed["closed"] == expected
        assert observed["scope_active"] == [False] * len(clients)

    state.checkpoint = checkpoint
    state.resolve = resolve
    state.cancelled_acquisition = cancelled_acquisition
    return state


@pytest.mark.parametrize("fail_at", [
    "first:connect", "first:list", "second:headers", "second:connect", "second:list",
])
async def test_resolver_cancellation_closes_current_and_previous_clients(mcp_lifecycle, fail_at):
    mcp_lifecycle.fail_at = fail_at
    await mcp_lifecycle.cancelled_acquisition(mcp_lifecycle.resolve)


async def test_resolver_listing_error_closes_only_failed_server(mcp_lifecycle):
    mcp_lifecycle.fail_at = "second:list"
    mcp_lifecycle.error = RuntimeError("Tool listing failed")
    try:
        _, clients = await mcp_lifecycle.resolve()
        first, second = mcp_lifecycle.clients
        assert mcp_lifecycle.closed == [("second", True)]
        assert second.scope._active is False
        assert clients == {"first": first}
        assert first.scope._active is True
        mcp_lifecycle.emitter.assert_awaited_once_with({
            "type": "notification",
            "data": {
                "type": "warning",
                "content": "Could not load MCP tools from 'second': Tool listing failed",
            },
        })
    finally:
        # Also rescue a failed regression in acquisition order's reverse.
        for client in reversed(mcp_lifecycle.clients):
            if client.exit_stack is not None:
                await client.disconnect()


async def test_cleanup_cancellation_closes_remaining_groups_then_reraises(mcp_lifecycle):
    async def acquire_and_cleanup():
        _, clients = await mcp_lifecycle.resolve()
        mcp_lifecycle.fail_at = "second:disconnect"
        await mcp_tools.cleanup_mcp_clients(
            {"first": clients["first"]}, {"second": clients["second"]},
        )

    await mcp_lifecycle.cancelled_acquisition(acquire_and_cleanup)


@pytest.mark.parametrize("fail_at", ["terminal", "builtin"])
async def test_loader_cancellation_closes_acquired_mcp_clients(monkeypatch, mcp_lifecycle, fail_at):
    from open_webui.models.config import Config
    from open_webui.utils import tools as core_tools

    mcp_lifecycle.fail_at = fail_at

    async def terminal(**kwargs):
        await mcp_lifecycle.checkpoint("terminal")
        return {}, ""

    async def builtin(**kwargs):
        await mcp_lifecycle.checkpoint("builtin")
        return {}

    monkeypatch.setattr(core_tools, "get_terminal_tools", terminal)
    monkeypatch.setattr(core_tools, "get_builtin_tools", builtin)
    monkeypatch.setattr(Config, "get", AsyncMock(return_value=[{"id": "terminal"}]))
    monkeypatch.setattr(tool_loader, "resolve_mcp_tools", mcp_lifecycle.resolve)

    async def acquire():
        return await tool_loader.build_tools_dict(
            request=SimpleNamespace(), model={"id": "model"}, metadata={},
            user=SimpleNamespace(id="user"), valves=SimpleNamespace(),
            extra_params={}, tool_id_list=["server:mcp:first", "server:mcp:second"],
            excluded_tool_ids=set(), resolved_terminal_id="terminal",
            resolved_direct_tool_servers=[],
        )

    await mcp_lifecycle.cancelled_acquisition(acquire)
