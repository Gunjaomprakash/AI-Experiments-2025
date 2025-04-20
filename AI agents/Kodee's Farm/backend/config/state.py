# state.py

import time
import copy

# ------------------
# Global Simulation State
# ------------------

# Thought and tool logs
thoughts_list = []
tools_list = []

# Field states
fields = []  # Will be updated when user sends request
field_snapshots = []  # Snapshots after every tool usage

# Timer
simulation_start_time = time.time()

# ------------------
# Utility Functions
# ------------------

def reset_simulation():
    """
    Resets the simulation environment for a new user session.
    Clears logs, field snapshots, and resets the simulation timer.
    """
    global thoughts_list, tools_list, fields, field_snapshots, simulation_start_time
    thoughts_list.clear()
    tools_list.clear()
    fields.clear()
    field_snapshots.clear()
    simulation_start_time = time.time()

def update_fields(new_fields: list):
    """
    Updates the global fields variable with user-provided field data.

    Args:
    - new_fields (list): A list of field dictionaries from client request.
    """
    # print(f"Updating fields with new data: {new_fields}")
    global fields
    fields.clear()
    fields.extend(copy.deepcopy(new_fields))
    # print(f"Updated fields: {fields}")

def snapshot_fields():
    """
    Takes a snapshot of the current field states.
    """
    print("Taking a snapshot of the current field states.")
    global field_snapshots, fields, simulation_start_time
    current_time = time.time()
    elapsed_time = round(current_time - simulation_start_time, 2)
    field_snapshots.append({
        "fields": copy.deepcopy(fields),
        "timestamp": elapsed_time
    })
    print(f"Snapshot taken at {elapsed_time} seconds: {field_snapshots[-1]}")
