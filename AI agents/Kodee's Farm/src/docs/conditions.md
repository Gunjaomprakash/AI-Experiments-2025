[
  {
    "id": "drought_stress",
    "name": "Drought Stress",
    "field_color": "red",
    "trigger_conditions": {
      "temperature": { "gt": 75 },
      "humidity": { "lt": 30 },
      "rain_forecast": { "lt": 30 }
    },
    "actions": [
      { "name": "analyze_soil" },
      { "name": "trigger_irrigation", "params": { "duration_minutes": 60 } },
      { "name": "update_forecast_model" }
    ],
    "final_color": "green"
  },
  {
    "id": "nutrient_deficiency",
    "name": "Nutrient Deficiency",
    "field_color": "orange",
    "trigger_conditions": {
      "soil_fertility": { "lt": 40 },
      "humidity": { "gt": 60 }
    },
    "actions": [
      { "name": "soil_sample_analysis" },
      { "name": "dispense_fertilizer" },
      { "name": "recheck_soil_health" }
    ],
    "final_color": "green"
  },
  {
    "id": "flood_risk",
    "name": "Flood Risk",
    "field_color": "blue",
    "trigger_conditions": {
      "humidity": { "gt": 90 },
      "rain_forecast": { "gt": 80 }
    },
    "actions": [
      { "name": "divert_drainage" },
      { "name": "raise_crop_beds" },
      { "name": "apply_root_protection" }
    ],
    "final_color": "green"
  },
  {
    "id": "disease_outbreak",
    "name": "Disease Outbreak",
    "field_color": "purple",
    "trigger_conditions": {
      "disease": { "gt": 60 },
      "humidity": { "gt": 70 }
    },
    "actions": [
      { "name": "identify_disease_type" },
      { "name": "apply_pesticide" },
      { "name": "quarantine_patch" }
    ],
    "final_color": "green"
  },
  {
    "id": "heatwave_hazard",
    "name": "Heatwave Hazard",
    "field_color": "yellow",
    "trigger_conditions": {
      "heat_wave": { "gt": 60 },
      "temperature": { "gt": 75 },
      "soil_fertility": { "gt": 70 }
    },
    "actions": [
      { "name": "activate_shade_nets" },
      { "name": "trigger_cooling_irrigation" },
      { "name": "monitor_evaporation" }
    ],
    "final_color": "green"
  }
]