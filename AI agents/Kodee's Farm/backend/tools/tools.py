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

# --- Helper Functions ---
def _record_tool_usage(tool_name: str):
    """Internal function to record tool usage without showing in the UI"""
    current_time = time.time()
    elapsed_time = round(current_time - simulation_start_time, 2)
    tools_list.append({
        "tool": tool_name,
        "timestamp": elapsed_time,
        "internal": False  # This will be shown in the UI
    })
    # Always capture the fields snapshot after any tool execution
    snapshot_fields(timestamp=elapsed_time)

def modify_field_metrics(field_id, updates):
    """
    Modifies the metrics of the specified field in the global 'fields' variable.
    Case-insensitive matching of metric names to handle naming inconsistencies.

    Args:
        field_id: The id of the field to modify.
        updates: A dictionary where keys are metric names and values are the amounts to change them.
    """
    for field in fields:
        if field["id"] == field_id:
            metrics_updated = set()
            for metric in field["metrics"]:
                # Use case-insensitive comparison and handle spaces/underscores
                metric_name_normalized = metric["name"].lower().replace(" ", "_").replace("-", "_")
                
                for update_key, update_value in updates.items():
                    update_key_normalized = update_key.lower().replace(" ", "_").replace("-", "_")
                    
                    if metric_name_normalized == update_key_normalized:
                        old_value = metric["value"]
                        metric["value"] += update_value
                        # Cap the values between 0 and 100
                        metric["value"] = max(0, min(100, metric["value"]))
                        print(f"[DEBUG] Updating {metric['name']} from {old_value} to {metric['value']}")
                        metrics_updated.add(update_key_normalized)
            
            # Log metrics that weren't found
            for update_key in updates.keys():
                update_key_normalized = update_key.lower().replace(" ", "_").replace("-", "_")
                if update_key_normalized not in metrics_updated:
                    print(f"[WARNING] No matching metric found for '{update_key}' in field {field_id}")
            
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
    _record_tool_usage("start_irrigation")
    print(f"[DEBUG] start_irrigation called for field {field_id}")
    updates = {"temperature": -2, "humidity": +5, "soil fertility": 10}
    modify_field_metrics(field_id, updates)
    return fields

def toggle_shade(field_id: int):
    """
    Toggles shade for the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to toggle shade.
    """
    _record_tool_usage("toggle_shade")
    print(f"[DEBUG] toggle_shade called for field {field_id}")
    updates = {"temperature": -10, "humidity": -5, "soil fertility": 3}
    modify_field_metrics(field_id, updates)
    return fields

def trigger_fungicide_spray(field_id: int):
    """
    Triggers a fungicide spray for the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to spray fungicide.
    """
    _record_tool_usage("trigger_fungicide_spray")
    print(f"[DEBUG] trigger_fungicide_spray called for field {field_id}")
    updates = {"disease": -20, "soil fertility": -5}
    modify_field_metrics(field_id, updates)
    return fields

def boost_fertilizer(field_id: int):
    """
    Boosts fertilizer for the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to boost fertilizer.
    """
    _record_tool_usage("boost_fertilizer")
    print(f"[DEBUG] boost_fertilizer called for field {field_id}")
    updates = {"soil fertility": 20, "disease": 3}
    modify_field_metrics(field_id, updates)
    return fields

def trigger_pesticide_spray(field_id: int):
    """
    Triggers a pesticide spray for the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to spray pesticide.
    """
    _record_tool_usage("trigger_pesticide_spray")
    print(f"[DEBUG] trigger_pesticide_spray called for field {field_id}")
    updates = {"disease": -10, "soil fertility": -2}
    modify_field_metrics(field_id, updates)
    return fields

def emergency_cooling(field_id: int):
    """
    Applies emergency cooling to the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to cool.
    """
    _record_tool_usage("emergency_cooling")
    print(f"[DEBUG] emergency_cooling called for field {field_id}")
    updates = {"temperature": -8, "humidity": 10}
    modify_field_metrics(field_id, updates)
    return fields

def humidify_field(field_id: int):
    """
    Humidifies the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to humidify.
    """
    _record_tool_usage("humidify_field")
    print(f"[DEBUG] humidify_field called for field {field_id}")
    updates = {"humidity": 15, "temperature": -1}
    modify_field_metrics(field_id, updates)
    return fields

def soil_recovery_treatment(field_id: int):
    """
    Applies soil recovery treatment to the specified field and updates its metrics.

    Args:
        field_id (int): The ID of the field to treat.
    """
    _record_tool_usage("soil_recovery_treatment")
    print(f"[DEBUG] soil_recovery_treatment called for field {field_id}")
    updates = {"soil fertility": 25, "disease": -10}
    modify_field_metrics(field_id, updates)
    return fields


def record_execution(reason: str , tool_name: str ):
    """
    Logs the agent's reasoning (thought) and tool usage, and captures a snapshot of field states.
    Note: This function now only records thoughts and not tool usage.

    Args:
        reason (str, optional): Reasoning or thought text before taking an action.
        tool_name (str, optional): Name of the tool being executed (no longer used for tools_list).

    Effect:
        - Appends reasoning with timestamp to thoughts_list
    """
    global simulation_start_time

    current_time = time.time()
    elapsed_time = round(current_time - simulation_start_time, 2)  # seconds since simulation started

    if reason:
        thoughts_list.append({
            "text": reason,
            "timestamp": elapsed_time
        })
    
    # We don't record tools here anymore, each tool records itself
    # We still snapshot for thoughts to preserve ordered timeline
    snapshot_fields(timestamp=elapsed_time)

    return {"status": "logged"}