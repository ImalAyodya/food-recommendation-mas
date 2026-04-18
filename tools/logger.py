def log(state, agent_name, data):
    state["logs"].append({
        "agent": agent_name,
        "data": data
    })