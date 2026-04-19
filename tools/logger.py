import datetime

def log_agent_step(state, agent_name, input_data, output_data):
    """
    Appends a log entry to the state's logs array for auditing.
    """
    if "logs" not in state:
        state["logs"] = []
    
    state["logs"].append({
        "agent": agent_name,
        "input": input_data,
        "output": output_data,
        "timestamp": datetime.datetime.now().isoformat()
    })
    return state