from __future__ import annotations

from typing import Any, Callable

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    START = "__start__"
    END = "__end__"

    class _CompiledMiniGraph:
        def __init__(self, nodes, edges, cond_edges):
            self._nodes = nodes
            self._edges = edges
            self._cond_edges = cond_edges

        def invoke(self, initial_state):
            state = dict(initial_state or {})
            cursor = self._edges.get(START, [END])[0]
            while cursor != END:
                fn = self._nodes[cursor]
                updates = fn(state) or {}
                state.update(updates)
                if cursor in self._cond_edges:
                    router, mapping = self._cond_edges[cursor]
                    branch = router(state)
                    cursor = mapping.get(branch, END)
                    continue
                nxt = self._edges.get(cursor, [END])
                cursor = nxt[0] if nxt else END
            return state

    class StateGraph:  # fallback when langgraph is unavailable
        def __init__(self, _state_type):
            self._nodes = {}
            self._edges = {}
            self._cond_edges = {}

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def add_edge(self, src, dst):
            self._edges.setdefault(src, []).append(dst)

        def add_conditional_edges(self, src, router, mapping):
            self._cond_edges[src] = (router, mapping)

        def compile(self):
            return _CompiledMiniGraph(self._nodes, self._edges, self._cond_edges)

from .types import GraphState

ToolExecutor = Callable[[str, dict[str, Any], GraphState], Any]
ModelInvoker = Callable[[GraphState], dict[str, Any]]


def build_tool_graph(executor: ToolExecutor):
    graph = StateGraph(GraphState)

    def plan_node(state: GraphState) -> GraphState:
        tool = str(state.get("requested_tool") or "").strip()
        args = state.get("requested_args") or {}
        return {"tool_call": {"tool": tool, "args": args}}

    def execute_node(state: GraphState) -> GraphState:
        tool_call = state.get("tool_call") or {}
        tool = str(tool_call.get("tool") or "").strip()
        args = tool_call.get("args") or {}
        result = executor(tool, args, state)
        logs = list(state.get("tool_logs") or [])
        logs.append({"call": {"tool": tool, "args": args}, "result": result})
        return {"tool_result": result, "tool_logs": logs}

    def approval_gate_node(state: GraphState) -> GraphState:
        result = state.get("tool_result")
        if not isinstance(result, dict):
            return {"need_approval": False, "approval_status": "not_required"}
        token = str(result.get("approval_token") or "")
        request_hash = str(result.get("request_hash") or "")
        if token:
            return {
                "need_approval": True,
                "approval_status": "pending",
                "approval_token": token,
                "request_hash": request_hash,
                "change_summary": result.get("change_summary") or {},
            }
        return {"need_approval": False, "approval_status": "not_required"}

    def finalize_node(state: GraphState) -> GraphState:
        tool_call = state.get("tool_call") or {}
        tool = str(tool_call.get("tool") or "").strip()
        result = state.get("tool_result")
        return {
            "response": {
                "status": "ok",
                "tool": tool,
                "result": result,
                "need_approval": bool(state.get("need_approval")),
            }
        }

    def route_after_execute(state: GraphState) -> str:
        tool_call = state.get("tool_call") or {}
        tool = str(tool_call.get("tool") or "").strip()
        if tool == "agent.preview_patch":
            return "approval_gate"
        return "finalize"

    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("approval_gate", approval_gate_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {"approval_gate": "approval_gate", "finalize": "finalize"},
    )
    graph.add_edge("approval_gate", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def build_model_graph(invoker: ModelInvoker):
    graph = StateGraph(GraphState)

    def invoke_node(state: GraphState) -> GraphState:
        output = invoker(state)
        return {
            "model_reply": str(output.get("reply") or ""),
            "model_reasoning": str(output.get("reasoning") or ""),
            "response": output,
        }

    graph.add_node("invoke_model", invoke_node)
    graph.add_edge(START, "invoke_model")
    graph.add_edge("invoke_model", END)
    return graph.compile()
