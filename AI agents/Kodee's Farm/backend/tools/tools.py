# tools/tools.py

import copy
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
from typing import Optional
import time
from ..config.state import  thoughts_list, tools_list, field_snapshots, snapshot_fields, fields

simulation_start_time = time.time()
# --- Helper Function ---
def modify_field_metrics(field_id, updates):
    """
    Modifies the metrics of the specified field in the global 'fields' variable.

    Args:
        field_id: The id of the field to modify.
        updates: A dictionary where keys are metric names and values are the amounts to change them.
    """
    for field in fields:
        if field["id"] == field_id:
            for metric in field["metrics"]:
                if metric["name"] in updates:
                    metric["value"] += updates[metric["name"]]
            return
    print(f"Field with id {field_id} not found.")

# --- Field Action Tools ---

def start_irrigation(field_id: int) -> dict:
    """
    Starts irrigation for the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to irrigate.

    Returns:
        dict: The updated fields data after irrigation.
    """
    print(f"[DEBUG] start_irrigation called for field {field_id}")
    updates = {"Temperature": -2, "Humidity": +5, "Soil Fertility": 10}
    modify_field_metrics(field_id, updates)
    return fields

def toggle_shade(field_id: int):
    """
    Toggles shade for the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to toggle shade.
    """
    print(f"[DEBUG] toggle_shade called for field {field_id}")
    updates = {"Temperature": 10, "Humidity": -5, "Soil Fertility": 3}
    modify_field_metrics(field_id, updates)

def trigger_fungicide_spray(field_id: int):
    """
    Triggers a fungicide spray for the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to spray fungicide.
    """
    print(f"[DEBUG] trigger_fungicide_spray called for field {field_id}")
    updates = {"Disease": -20, "Soil Fertility": -5}
    modify_field_metrics(field_id, updates)
    return fields

def boost_fertilizer(field_id: int):
    """
    Boosts fertilizer for the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to boost fertilizer.
    """
    print(f"[DEBUG] boost_fertilizer called for field {field_id}")
    updates = {"Soil Fertility": 20, "Disease": 3}
    modify_field_metrics(field_id, updates)
    return fields

def trigger_pesticide_spray(field_id: int):
    """
    Triggers a pesticide spray for the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to spray pesticide.
    """
    print(f"[DEBUG] trigger_pesticide_spray called for field {field_id}")
    updates = {"Disease": -10, "Soil Fertility": -2}
    modify_field_metrics(field_id, updates)
    return fields

def emergency_cooling(field_id: int):
    """
    Applies emergency cooling to the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to cool.
    """
    print(f"[DEBUG] emergency_cooling called for field {field_id}")
    updates = {"Temperature": -8, "Humidity": 10}
    modify_field_metrics(field_id, updates)
    return fields

def humidify_field(field_id: int):
    """
    Humidifies the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to humidify.
    """
    print(f"[DEBUG] humidify_field called for field {field_id}")
    updates = {"Humidity": 15, "Temperature": -1}
    modify_field_metrics(field_id, updates)
    return fields

def soil_recovery_treatment(field_id: int):
    """
    Applies soil recovery treatment to the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to treat.
    """
    print(f"[DEBUG] soil_recovery_treatment called for field {field_id}")
    updates = {"Soil Fertility": 25, "Disease": 5}
    modify_field_metrics(field_id, updates)
    return fields


def record_execution(reason: str , tool_name: str ):
    """
    Logs the agent's reasoning (thought) and tool usage, and captures a snapshot of field states.

    Args:
        reason (str, optional): Reasoning or thought text before taking an action.
        tool_name (str, optional): Name of the tool being executed.

    Effect:
        - Appends reasoning with timestamp to thoughts_list
        - Appends tool call with timestamp to tools_list
        - Captures the current field snapshot in field_snapshots
    """
    global simulation_start_time

    current_time = time.time()
    elapsed_time = round(current_time - simulation_start_time, 2)  # seconds since simulation started

    if reason:
        thoughts_list.append({
            "text": reason,
            "timestamp": elapsed_time
        })
    
    if tool_name:
        tools_list.append({
            "tool": tool_name,
            "timestamp": elapsed_time
        })

    # Always capture the fields snapshot after any tool execution (or important thought)
    snapshot_fields(timestamp=elapsed_time)

    return {"status": "logged"}