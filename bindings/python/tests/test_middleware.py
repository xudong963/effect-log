"""Tests for framework middleware integrations.

These tests mock the framework dependencies so they can run without
installing LangGraph, OpenAI Agents SDK, or CrewAI.
"""

import json
import sys
import types
from unittest.mock import MagicMock, AsyncMock

import pytest

from effect_log import EffectKind, EffectLog, ToolDef


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_tools():
    """Create a set of effect-log tools for testing."""
    call_counts = {"search": 0, "send_email": 0}

    def search(args):
        call_counts["search"] += 1
        return {"results": [f"result for {args.get('query', '')}"], "count": call_counts["search"]}

    def send_email(args):
        call_counts["send_email"] += 1
        return {"sent": True, "to": args.get("to", ""), "count": call_counts["send_email"]}

    tools = [
        ToolDef("search", EffectKind.ReadOnly, search),
        ToolDef("send_email", EffectKind.IrreversibleWrite, send_email),
    ]
    return tools, call_counts


def make_log(tools, **kwargs):
    return EffectLog(execution_id="test-mw-001", tools=tools, **kwargs)


# ── LangGraph Middleware Tests ───────────────────────────────────────────────

def _mock_langchain():
    """Install mock langchain_core modules so the middleware can import."""
    lc = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_messages = types.ModuleType("langchain_core.messages")

    class BaseTool:
        name: str = ""
        description: str = ""

    class ToolMessage:
        def __init__(self, content, tool_call_id, name):
            self.content = content
            self.tool_call_id = tool_call_id
            self.name = name

    lc_tools.BaseTool = BaseTool
    lc_messages.ToolMessage = ToolMessage
    lc.tools = lc_tools
    lc.messages = lc_messages

    sys.modules["langchain_core"] = lc
    sys.modules["langchain_core.tools"] = lc_tools
    sys.modules["langchain_core.messages"] = lc_messages
    return BaseTool, ToolMessage


class TestLangGraphMiddleware:

    def setup_method(self):
        self.BaseTool, self.ToolMessage = _mock_langchain()

    def teardown_method(self):
        for mod in ["langchain_core", "langchain_core.tools", "langchain_core.messages"]:
            sys.modules.pop(mod, None)

    def test_effect_logged_tools_wraps_correctly(self):
        from effect_log.middleware.langgraph import effect_logged_tools

        tools, counts = make_tools()
        log = make_log(tools)

        # Create mock LangChain tools
        mock_search = MagicMock()
        mock_search.name = "search"
        mock_search.description = "Search the web"

        mock_email = MagicMock()
        mock_email.name = "send_email"
        mock_email.description = "Send an email"

        wrapped = effect_logged_tools(log, [
            {"tool": mock_search, "effect": EffectKind.ReadOnly},
            {"tool": mock_email, "effect": EffectKind.IrreversibleWrite},
        ])

        assert len(wrapped) == 2
        assert wrapped[0].name == "search"
        assert wrapped[1].name == "send_email"

    def test_wrapped_tool_invoke(self):
        from effect_log.middleware.langgraph import effect_logged_tools

        tools, counts = make_tools()
        log = make_log(tools)

        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search"

        wrapped = effect_logged_tools(log, [
            {"tool": mock_tool, "effect": EffectKind.ReadOnly},
        ])

        result = wrapped[0].invoke({"query": "python"})
        parsed = json.loads(result)
        assert "results" in parsed
        assert counts["search"] == 1

    def test_tool_node_processes_tool_calls(self):
        from effect_log.middleware.langgraph import EffectLogToolNode

        tools, counts = make_tools()
        log = make_log(tools)

        mock_search_tool = MagicMock()
        mock_search_tool.name = "search"

        node = EffectLogToolNode(log, [
            {"tool": mock_search_tool, "effect": EffectKind.ReadOnly},
        ])

        # Simulate an AIMessage with tool_calls
        ai_message = MagicMock()
        ai_message.tool_calls = [
            {"id": "call-1", "name": "search", "args": {"query": "test"}},
        ]

        state = {"messages": [ai_message]}
        result = node(state)

        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg.name == "search"
        assert "results" in msg.content
        assert counts["search"] == 1

    def test_make_tooldefs_with_func(self):
        """make_tooldefs uses .func when available (custom @tool)."""
        from effect_log.middleware.langgraph import make_tooldefs

        call_log = []

        def my_search(query=""):
            call_log.append(("search", query))
            return f"results for {query}"

        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.func = my_search  # custom @tool has .func

        defs = make_tooldefs([{"tool": mock_tool, "effect": EffectKind.ReadOnly}])
        assert len(defs) == 1

        log = make_log(defs)
        result = log.execute("search", {"query": "python"})
        assert call_log == [("search", "python")]

    def test_make_tooldefs_without_func(self):
        """make_tooldefs falls back to .invoke() for pre-built tools."""
        from effect_log.middleware.langgraph import make_tooldefs

        mock_tool = MagicMock()
        mock_tool.name = "tavily_search"
        del mock_tool.func  # pre-built tool has no .func
        mock_tool.invoke.return_value = "search results"

        defs = make_tooldefs([{"tool": mock_tool, "effect": EffectKind.ReadOnly}])
        log = make_log(defs)
        result = log.execute("tavily_search", {"query": "test"})

        mock_tool.invoke.assert_called_once_with({"query": "test"})

    def test_recovery_through_langgraph(self):
        """Verify sealed results work through the LangGraph middleware."""
        import tempfile, os
        from effect_log.middleware.langgraph import effect_logged_tools

        tmpdir = tempfile.mkdtemp()
        db = f"sqlite:///{os.path.join(tmpdir, 'test.db')}"

        tools, counts1 = make_tools()
        log = make_log(tools, storage=db)

        mock_email = MagicMock()
        mock_email.name = "send_email"

        wrapped = effect_logged_tools(log, [
            {"tool": mock_email, "effect": EffectKind.IrreversibleWrite},
        ])

        # First execution
        result1 = wrapped[0].invoke({"to": "ceo@co.com"})
        assert counts1["send_email"] == 1

        # Recovery
        tools2, counts2 = make_tools()
        log2 = EffectLog(execution_id="test-mw-001", tools=tools2, storage=db, recover=True)

        mock_email2 = MagicMock()
        mock_email2.name = "send_email"

        wrapped2 = effect_logged_tools(log2, [
            {"tool": mock_email2, "effect": EffectKind.IrreversibleWrite},
        ])

        result2 = wrapped2[0].invoke({"to": "ceo@co.com"})
        # Sealed — not re-executed
        assert counts2["send_email"] == 0
        # Same result
        assert json.loads(result1) == json.loads(result2)


# ── OpenAI Agents SDK Middleware Tests ───────────────────────────────────────

def _mock_openai_agents():
    """Install mock agents module."""
    agents_mod = types.ModuleType("agents")

    class FunctionTool:
        def __init__(self, name, description, params_json_schema, on_invoke_tool):
            self.name = name
            self.description = description
            self.params_json_schema = params_json_schema
            self.on_invoke_tool = on_invoke_tool

    def function_tool(fn):
        """Mock @function_tool decorator."""
        return FunctionTool(
            name=fn.__name__,
            description=fn.__doc__ or "",
            params_json_schema={},
            on_invoke_tool=AsyncMock(),
        )

    agents_mod.FunctionTool = FunctionTool
    agents_mod.function_tool = function_tool
    sys.modules["agents"] = agents_mod
    return FunctionTool


class TestOpenAIAgentsMiddleware:

    def setup_method(self):
        self.FunctionTool = _mock_openai_agents()

    def teardown_method(self):
        sys.modules.pop("agents", None)

    def test_wrap_function_tool(self):
        from effect_log.middleware.openai_agents import wrap_function_tool

        tools, counts = make_tools()
        log = make_log(tools)

        original = self.FunctionTool(
            name="search",
            description="Search the web",
            params_json_schema={"type": "object"},
            on_invoke_tool=AsyncMock(return_value="original"),
        )

        wrapped = wrap_function_tool(log, original, EffectKind.ReadOnly)
        assert wrapped.name == "search"
        assert wrapped.description == "Search the web"

    @pytest.mark.asyncio
    async def test_wrapped_tool_invocation(self):
        from effect_log.middleware.openai_agents import wrap_function_tool

        tools, counts = make_tools()
        log = make_log(tools)

        original = self.FunctionTool(
            name="search",
            description="Search",
            params_json_schema={},
            on_invoke_tool=AsyncMock(return_value="original"),
        )

        wrapped = wrap_function_tool(log, original)
        result = await wrapped.on_invoke_tool(None, '{"query": "test"}')

        parsed = json.loads(result)
        assert "results" in parsed
        assert counts["search"] == 1

    def test_make_tools_creates_both(self):
        """make_tools returns (sdk_tools, tooldefs) from raw functions."""
        from effect_log.middleware.openai_agents import make_tools as oa_make_tools

        call_log = []

        def get_weather(city: str) -> str:
            """Get weather."""
            call_log.append(("weather", city))
            return f"sunny in {city}"

        def send_alert(message: str) -> str:
            """Send alert."""
            call_log.append(("alert", message))
            return f"sent: {message}"

        sdk_tools, tooldefs = oa_make_tools([
            {"func": get_weather, "effect": EffectKind.ReadOnly},
            {"func": send_alert, "effect": EffectKind.IrreversibleWrite},
        ])

        # SDK tools are FunctionTool instances
        assert len(sdk_tools) == 2
        assert sdk_tools[0].name == "get_weather"
        assert sdk_tools[1].name == "send_alert"

        # ToolDefs work with EffectLog
        assert len(tooldefs) == 2
        log = make_log(tooldefs)
        log.execute("get_weather", {"city": "SF"})
        assert call_log == [("weather", "SF")]

        log.execute("send_alert", {"message": "hot"})
        assert call_log == [("weather", "SF"), ("alert", "hot")]

    def test_effect_logged_agent(self):
        from effect_log.middleware.openai_agents import effect_logged_agent

        tools, counts = make_tools()
        log = make_log(tools)

        ft = self.FunctionTool(
            name="search",
            description="Search",
            params_json_schema={},
            on_invoke_tool=AsyncMock(),
        )

        agent = MagicMock()
        agent.tools = [ft]

        result = effect_logged_agent(log, agent, {"search": EffectKind.ReadOnly})
        assert len(result.tools) == 1
        assert result.tools[0].name == "search"


# ── CrewAI Middleware Tests ──────────────────────────────────────────────────

def _mock_crewai():
    """Install mock crewai modules."""
    crewai = types.ModuleType("crewai")
    crewai_tools = types.ModuleType("crewai.tools")

    class BaseTool:
        name: str = ""
        description: str = ""
        def _run(self, **kwargs):
            pass

    crewai_tools.BaseTool = BaseTool
    crewai.tools = crewai_tools
    sys.modules["crewai"] = crewai
    sys.modules["crewai.tools"] = crewai_tools
    return BaseTool


class TestCrewAIMiddleware:

    def setup_method(self):
        self.BaseTool = _mock_crewai()

    def teardown_method(self):
        for mod in ["crewai", "crewai.tools"]:
            sys.modules.pop(mod, None)

    def test_effect_logged_tool_wraps(self):
        from effect_log.middleware.crewai import effect_logged_tool

        tools, counts = make_tools()
        log = make_log(tools)

        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search"

        wrapped = effect_logged_tool(log, mock_tool, EffectKind.ReadOnly)
        assert wrapped.name == "search"

    def test_effect_logged_tool_run(self):
        from effect_log.middleware.crewai import effect_logged_tool

        tools, counts = make_tools()
        log = make_log(tools)

        mock_tool = MagicMock()
        mock_tool.name = "search"

        wrapped = effect_logged_tool(log, mock_tool)
        result = wrapped.run({"query": "test"})

        parsed = json.loads(result)
        assert "results" in parsed
        assert counts["search"] == 1

    def test_effect_logged_tool_kwargs(self):
        from effect_log.middleware.crewai import effect_logged_tool

        tools, counts = make_tools()
        log = make_log(tools)

        mock_tool = MagicMock()
        mock_tool.name = "search"

        wrapped = effect_logged_tool(log, mock_tool)
        result = wrapped._run(query="test")

        parsed = json.loads(result)
        assert "results" in parsed

    def test_effect_logged_crew(self):
        from effect_log.middleware.crewai import effect_logged_crew

        tools, counts = make_tools()
        log = make_log(tools)

        mock_tool = MagicMock()
        mock_tool.name = "search"

        agent = MagicMock()
        agent.tools = [mock_tool]

        crew = MagicMock()
        crew.agents = [agent]

        result = effect_logged_crew(log, crew, {"search": EffectKind.ReadOnly})
        assert len(result.agents[0].tools) == 1
        assert result.agents[0].tools[0].name == "search"

    def test_make_tooldefs_with_func(self):
        """make_tooldefs uses .func when available (custom @tool)."""
        from effect_log.middleware.crewai import make_tooldefs

        call_log = []

        def my_search(query=""):
            call_log.append(("search", query))
            return f"results for {query}"

        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.func = my_search

        defs = make_tooldefs([{"tool": mock_tool, "effect": EffectKind.ReadOnly}])
        log = make_log(defs)
        log.execute("search", {"query": "ai"})
        assert call_log == [("search", "ai")]

    def test_make_tooldefs_without_func(self):
        """make_tooldefs falls back to ._run() for pre-built tools."""
        from effect_log.middleware.crewai import make_tooldefs

        mock_tool = MagicMock()
        mock_tool.name = "serper_search"
        del mock_tool.func  # pre-built tool has no .func
        mock_tool._run.return_value = "search results"

        defs = make_tooldefs([{"tool": mock_tool, "effect": EffectKind.ReadOnly}])
        log = make_log(defs)
        log.execute("serper_search", {"query": "test"})

        mock_tool._run.assert_called_once_with(query="test")

    def test_recovery_through_crewai(self):
        """Verify sealed results work through the CrewAI middleware."""
        import tempfile, os
        from effect_log.middleware.crewai import effect_logged_tool

        tmpdir = tempfile.mkdtemp()
        db = f"sqlite:///{os.path.join(tmpdir, 'test.db')}"

        tools, counts1 = make_tools()
        log = make_log(tools, storage=db)

        mock_email = MagicMock()
        mock_email.name = "send_email"

        wrapped = effect_logged_tool(log, mock_email)
        result1 = wrapped.run({"to": "ceo@co.com"})
        assert counts1["send_email"] == 1

        # Recovery
        tools2, counts2 = make_tools()
        log2 = EffectLog(execution_id="test-mw-001", tools=tools2, storage=db, recover=True)

        mock_email2 = MagicMock()
        mock_email2.name = "send_email"

        wrapped2 = effect_logged_tool(log2, mock_email2)
        result2 = wrapped2.run({"to": "ceo@co.com"})

        assert counts2["send_email"] == 0
        assert json.loads(result1) == json.loads(result2)
