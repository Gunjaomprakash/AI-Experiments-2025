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

def snapshot_fields(timestamp=None):
    """
    Takes a snapshot of the current field states.
    
    Args:
        timestamp (float, optional): The timestamp to use for this snapshot.
                                     If None, the current elapsed time will be used.
    """
    print("Taking a snapshot of the current field states.")
    global field_snapshots, fields, simulation_start_time
    
    if timestamp is None:
        current_time = time.time()
        timestamp = round(current_time - simulation_start_time, 2)
        
    field_snapshots.append({
        "fields": copy.deepcopy(fields),
        "timestamp": timestamp
    })
    
    print(f"Snapshot taken at {timestamp} seconds: {field_snapshots[-1]}")
