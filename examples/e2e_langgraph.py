#!/usr/bin/env python3
"""
End-to-end LangGraph + effect-log example.

Setup:
    pip install ai-effectlog langchain-core langchain-openai langgraph

Set your API key:
    export OPENAI_API_KEY="sk-..."

This example defines each tool function ONCE using @lc_tool and uses
make_tooldefs() to create EffectLog ToolDef entries automatically — no
duplicate definitions needed.

The place_order tool is classified as IrreversibleWrite — on recovery it will
NOT be re-executed.
"""

from __future__ import annotations

import os
import tempfile

from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph

from effect_log import EffectKind, EffectLog
from effect_log.middleware.langgraph import EffectLogToolNode, make_tooldefs

# -- Tool implementations (defined once) -------------------------------------

call_counts: dict[str, int] = {"lookup_stock": 0, "place_order": 0}


@lc_tool
def lookup_stock(symbol: str) -> str:
    """Look up the current price of a stock by its ticker symbol."""
    call_counts["lookup_stock"] += 1
    prices = {"AAPL": 187.50, "GOOG": 141.20, "MSFT": 415.30}
    price = prices.get(symbol.upper(), 100.00)
    return f"{symbol.upper()}: ${price:.2f}"


@lc_tool
def place_order(symbol: str, quantity: int) -> str:
    """Place a stock buy order. This action is irreversible."""
    call_counts["place_order"] += 1
    return f"Order placed: BUY {quantity} shares of {symbol.upper()}"


# -- Graph construction ------------------------------------------------------


def should_continue(state: MessagesState) -> str:
    """Route to tools if the last message has tool calls, otherwise end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph(tool_node):
    """Build a simple ReAct-style graph with an LLM node and a tool node."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([lookup_stock, place_order])

    def call_llm(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    return graph.compile()


# -- Main --------------------------------------------------------------------


def main():
    tmpdir = tempfile.mkdtemp()
    db = f"sqlite:///{os.path.join(tmpdir, 'effects.db')}"

    # 1. Create EffectLog ToolDefs from the same @lc_tool functions
    tool_specs = [
        {"tool": lookup_stock, "effect": EffectKind.ReadOnly},
        {"tool": place_order, "effect": EffectKind.IrreversibleWrite},
    ]
    tooldefs = make_tooldefs(tool_specs)

    # 2. Build EffectLog with the generated ToolDefs
    log = EffectLog(execution_id="trading-agent-run-1", tools=tooldefs, storage=db)

    # 3. Create the EffectLogToolNode (drop-in replacement for LangGraph's ToolNode)
    tool_node = EffectLogToolNode(log, tool_specs)

    # 4. Build and run the graph
    app = build_graph(tool_node)

    print("=== Running LangGraph agent ===\n")
    result = app.invoke(
        {"messages": [("user", "Look up the price of AAPL and buy 10 shares.")]}
    )

    # Print conversation
    for msg in result["messages"]:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if content:
            print(f"  [{role}] {content[:120]}")

    # 5. Show effect-log history
    print("\n=== Effect-log history ===\n")
    for entry in log.history():
        print(
            f"  seq={entry['sequence']}  tool={entry['tool_name']:<15}  "
            f"effect={entry['effect_kind']:<20}  outcome={entry['outcome']}"
        )

    print(f"\n  lookup_stock called: {call_counts['lookup_stock']} time(s)")
    print(f"  place_order  called: {call_counts['place_order']} time(s)")

    # 6. Demonstrate recovery
    print("\n=== Simulating recovery ===\n")
    for k in call_counts:
        call_counts[k] = 0

    tooldefs2 = make_tooldefs(tool_specs)
    log2 = EffectLog(
        execution_id="trading-agent-run-1", tools=tooldefs2, storage=db, recover=True
    )

    for entry in log.history():
        log2.execute(entry["tool_name"], {})

    print(
        f"  lookup_stock re-executed: {call_counts['lookup_stock']} (ReadOnly -> replayed)"
    )
    print(
        f"  place_order  re-executed: {call_counts['place_order']} (IrreversibleWrite -> SEALED)"
    )
    assert call_counts["place_order"] == 0, (
        "place_order must not re-execute on recovery"
    )
    print("\n  PASS: order not re-placed on recovery")


if __name__ == "__main__":
    main()
