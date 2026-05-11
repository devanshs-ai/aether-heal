import sys, os
sys.path.insert(0, os.getcwd())
import json
import uuid
from typing import TypedDict
from dotenv import load_dotenv
from groq import Groq
from agent.tools import TOOLS, log_agent_action
from agent.prompts import SYSTEM_PROMPT


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

class AgentState(TypedDict):
    run_id: str
    context: str
    reasoning: str
    tool_called: str
    tool_result: str
    conclusion: str
    iterations: int
    done: bool
    _tool_args: dict

def diagnose(state: AgentState) -> AgentState:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": state["context"]}
        ]
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        decision = json.loads(raw.strip())
    except Exception:
        decision = {
            "reasoning": raw,
            "tool": "get_drift_report",
            "tool_args": {},
            "conclusion": "Fallback to drift report"
        }

    return {**state,
            "reasoning":   decision.get("reasoning", ""),
            "tool_called": decision.get("tool", ""),
            "conclusion":  decision.get("conclusion", ""),
            "_tool_args":  decision.get("tool_args", {})}

def execute(state: AgentState) -> AgentState:
    tool_name = state["tool_called"]
    tool_args  = state.get("_tool_args", {})

    if tool_name not in TOOLS:
        result = f"Unknown tool: {tool_name}"
    else:
        try:
            result = TOOLS[tool_name](**tool_args)
            if isinstance(result, dict):
                result = json.dumps(result, indent=2)
        except Exception as e:
            result = f"Tool error: {e}"

    log_agent_action(state["run_id"], state["reasoning"], tool_name, str(result))
    print(f"\n[Agent] Tool     : {tool_name}")
    print(f"[Agent] Reasoning: {state['reasoning'][:200]}...")
    print(f"[Agent] Result   : {str(result)[:300]}")

    return {**state, "tool_result": str(result), "iterations": state["iterations"] + 1}

def should_continue(state: AgentState) -> str:
    if state["iterations"] >= 3:
        return "end"
    try:
        from agent.tools import get_drift_report
        report = get_drift_report()
        if not report.get("drift_detected", False):
            return "end"
    except:
        return "end"
    return "end"

def build_graph():
    from langgraph.graph import StateGraph, START, END
    g = StateGraph(AgentState)
    g.add_node("diagnose", diagnose)
    g.add_node("execute",  execute)
    g.add_edge(START, "diagnose")
    g.add_edge("diagnose", "execute")
    g.add_conditional_edges("execute", should_continue, {"end": END, "diagnose": "diagnose"})
    return g.compile()

def run_agent(context: str) -> AgentState:
    graph = build_graph()
    state = AgentState(
        run_id=str(uuid.uuid4())[:8],
        context=context,
        reasoning="",
        tool_called="",
        tool_result="",
        conclusion="",
        iterations=0,
        done=False,
        _tool_args={}
    )
    return graph.invoke(state)

if __name__ == "__main__":
    from agent.tools import get_drift_report
    report = get_drift_report()
    context = f"""
    Pipeline alert: Data drift has already been confirmed.
    
    Drift Report (already fetched, do NOT call get_drift_report again):
    {json.dumps(report, indent=2)}
    
    Analysis:
    - max_js_divergence={report['max_js_divergence']} which is > 0.3 threshold
    - No null values detected in null_counts
    - MonthlyCharges is the sick feature with JS=0.487
    
    Per your decision logic:
    - drift_detected=True, null_counts are zero → NOT a data corruption issue
    - JS divergence = 0.487 > 0.3 → this is concept drift, NOT transient noise
    - Required action: trigger_retraining immediately
    
    Call trigger_retraining now. Do not call get_drift_report.
    """
    result = run_agent(context)
    print(f"\n[Done] Agent complete | tool={result['tool_called']} | iterations={result['iterations']}")
    print(f"Conclusion: {result['conclusion']}")