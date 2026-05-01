"""
main.py
========
CLI entry point for the Food Recommendation Multi-Agent System.

Uses the LangGraph StateGraph defined in workflow.py to run the full
four-agent pipeline:
    Agent 1: Preference Analyzer
    Agent 2: Menu Fetcher
    Agent 3: Nutrition Analyzer
    Agent 4: Recommendation & Report

Usage:
    python main.py
"""

from workflow import build_graph, make_initial_state


def main():
    user_input = input("Enter your food preference: ").strip()
    if not user_input:
        print("No input provided. Exiting.")
        return

    print("\n[MAS] Building LangGraph pipeline...")
    graph = build_graph()

    state = make_initial_state(user_input)
    print(f"[MAS] Run ID: {state['run_id']}\n")

    result = graph.invoke(state)

    print("\n" + "=" * 60)
    print("  TOP RECOMMENDATIONS")
    print("=" * 60)
    for meal in result.get("final_recommendations", []):
        print(
            f"  [{meal.get('rank', '?')}] {meal.get('name', 'Unknown'):<35}"
            f" | Score: {meal.get('final_score', 0):.2f}"
            f" | {meal.get('calories', '?')} kcal"
            f" | {meal.get('category', '?').capitalize()}"
        )

    if result.get("report_path"):
        print(f"\n  Report  -> {result['report_path']}")
    if result.get("json_path"):
        print(f"  Results -> {result['json_path']}")

    errors = result.get("errors", [])
    if errors:
        print("\n  Warnings:")
        for e in errors:
            print(f"    ⚠ {e}")

    print(f"\n  Agents executed: {len(result.get('logs', []))}")
    print(f"  Tool calls:      {len(result.get('tool_calls', []))}")


if __name__ == "__main__":
    main()