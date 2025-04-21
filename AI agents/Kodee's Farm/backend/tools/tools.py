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
    Waters the field to increase soil moisture and humidity. Great for drought conditions.

    Args:
        field_id (int): The ID of the field to irrigate.

    Returns:
        dict: The updated fields data after irrigation.
    """
    _record_tool_usage("start_irrigation")
    print(f"[DEBUG] start_irrigation called for field {field_id}")
    updates = {
        "humidity": +15,        # Significant humidity increase
        "temperature": -5,      # Cooling effect
        "soil fertility": +5    # Slight fertility boost from water nutrients
    }
    modify_field_metrics(field_id, updates)
    return fields

def toggle_shade(field_id: int):
    """
    Deploys shade structures to protect crops from excessive heat and sun exposure.
    
    Args:
        field_id (int): The ID of the field to add shade.
    """
    _record_tool_usage("toggle_shade")
    print(f"[DEBUG] toggle_shade called for field {field_id}")
    updates = {
        "temperature": -15,     # Major temperature reduction
        "humidity": +5,         # Slight humidity increase from reduced evaporation
        "heat wave": -10        # Reduced heat wave impact
    }
    modify_field_metrics(field_id, updates)
    return fields

def trigger_fungicide_spray(field_id: int):
    """
    Applies fungicide to treat and prevent fungal diseases in crops.
    Very effective against disease but may impact soil health slightly.

    Args:
        field_id (int): The ID of the field to spray fungicide.
    """
    _record_tool_usage("trigger_fungicide_spray")
    print(f"[DEBUG] trigger_fungicide_spray called for field {field_id}")
    updates = {
        "disease": -25,        # Major disease reduction 
        "soil fertility": -5    # Slight negative impact on soil microbiome
    }
    modify_field_metrics(field_id, updates)
    return fields

def boost_fertilizer(field_id: int):
    """
    Applies balanced fertilizer to significantly improve soil fertility.
    The gold standard for poor soil conditions.

    Args:
        field_id (int): The ID of the field to fertilize.
    """
    _record_tool_usage("boost_fertilizer")
    print(f"[DEBUG] boost_fertilizer called for field {field_id}")
    updates = {
        "soil fertility": +30,  # Major fertility boost
        "disease": +5           # Slight disease increase (over-fertilization risk)
    }
    modify_field_metrics(field_id, updates)
    return fields

def trigger_pesticide_spray(field_id: int):
    """
    Applies pesticide to control insect pests that damage crops.
    Reduces disease risk from pest vectors but affects soil health.

    Args:
        field_id (int): The ID of the field to spray pesticide.
    """
    _record_tool_usage("trigger_pesticide_spray")
    print(f"[DEBUG] trigger_pesticide_spray called for field {field_id}")
    updates = {
        "disease": -15,         # Good disease reduction from pest control
        "soil fertility": -10   # Moderate negative impact on soil health
    }
    modify_field_metrics(field_id, updates)
    return fields

def emergency_cooling(field_id: int):
    """
    Activates emergency cooling systems like misters and fans.
    The most effective solution for extreme heat conditions.

    Args:
        field_id (int): The ID of the field to cool.
    """
    _record_tool_usage("emergency_cooling")
    print(f"[DEBUG] emergency_cooling called for field {field_id}")
    updates = {
        "temperature": -20,     # Dramatic temperature reduction
        "humidity": +10,        # Moderate humidity increase
        "heat wave": -20        # Major heat wave mitigation
    }
    modify_field_metrics(field_id, updates)
    return fields

def humidify_field(field_id: int):
    """
    Increases field humidity using specialized misting systems.
    Perfect for dry conditions affecting crop growth.

    Args:
        field_id (int): The ID of the field to humidify.
    """
    _record_tool_usage("humidify_field")
    updates = {
        "humidity": +25,        # Major humidity increase
        "temperature": -5       # Slight cooling effect
    }
    modify_field_metrics(field_id, updates)
    return fields

def soil_recovery_treatment(field_id: int):
    """
    Applies comprehensive soil treatment including organic matter, beneficial microbes, and soil conditioners.
    The best option for both improving soil fertility and fighting disease.

    Args:
        field_id (int): The ID of the field to treat.
    """
    _record_tool_usage("soil_recovery_treatment")
    print(f"[DEBUG] soil_recovery_treatment called for field {field_id}")
    updates = {
        "soil fertility": +25,  # Major fertility improvement
        "disease": -15          # Significant disease reduction through beneficial microbes
    }
    modify_field_metrics(field_id, updates)
    return fields

def organic_treatment(field_id: int):
    """
    Applies organic farming treatments (compost tea, beneficial insects, plant-based sprays).
    Balanced improvement across multiple metrics without negative side effects.

    Args:
        field_id (int): The ID of the field for organic treatment.
    """
    _record_tool_usage("organic_treatment")
    print(f"[DEBUG] organic_treatment called for field {field_id}")
    updates = {
        "soil fertility": +15,  # Good fertility improvement
        "disease": -10,         # Moderate disease reduction
        "humidity": +5          # Slight humidity improvement from mulching
    }
    modify_field_metrics(field_id, updates)
    return fields

def rainwater_harvesting(field_id: int):
    """
    Uses collected rainwater for sustainable irrigation, particularly effective during low rain forecasts.
    Provides moisture without affecting temperature significantly.

    Args:
        field_id (int): The ID of the field to apply rainwater to.
    """
    _record_tool_usage("rainwater_harvesting")
    print(f"[DEBUG] rainwater_harvesting called for field {field_id}")
    updates = {
        "humidity": +20,        # Major humidity increase
        "soil fertility": +5,   # Slight fertility boost from natural minerals in rainwater
        "rain forecast": -10    # Reduces impact of low rain forecast
    }
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