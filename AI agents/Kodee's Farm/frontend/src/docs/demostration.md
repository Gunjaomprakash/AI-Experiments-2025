# Kodee's Farm Demonstration Scenarios

This document outlines 5 distinct demonstration scenarios for showcasing Kodee's capabilities across different farming situations. Each scenario includes specific field settings, the prompt to use, and whether to include image/memory access.
## Scenario 0 :  Hey kodee i found this in feld 1, what is it? and is it common in chicago? is it due to prev crop we harveted? lets fix it


## Scenario 1: Heat Wave Emergency - works

**Description:** Demonstrate emergency cooling response to extreme heat conditions

**Field Values to Set:**
- Field 1:
  - Temperature: 95
  - Humidity: 20
  - Rain Forecast: 10
  - Soil Fertility: 65
  - Heat Wave: 80
  - Disease: 25

**Message to Use:**
```
My crops in Field 1 look stressed from the extreme heat. What actions should I take immediately to protect them?
```

**Settings:**
- Image: No
- Memory Toggle: Off

**Expected Tools Used:** emergency_cooling, possibly followed by humidify_field

## Scenario 2: Banana Disease Analysis - works

**Description:** Showcase image analysis with a diseased banana plant

**Field Values to Set:**
- Field 2:
  - Temperature: 75
  - Humidity: 45
  - Rain Forecast: 30
  - Soil Fertility: 40
  - Heat Wave: 25
  - Disease: 70

**Message to Use:**
```
I've uploaded a photo of my banana plant with yellow spots. Can you analyze what's wrong and tell me what to do about it?
```

**Settings:**
- Image: Use "banana.jpeg"
- Memory Toggle: Off

**Expected Tools Used:** image_analysis, trigger_fungicide_spray, boost_fertilizer

## Scenario 3: Crop Planning with Historical Data

**Description:** Demonstrate memory access and Google search integration

**Field Values to Set:**
- Field 3:
  - Temperature: 65
  - Humidity: 50
  - Rain Forecast: 40
  - Soil Fertility: 55
  - Heat Wave: 20
  - Disease: 15

**Message to Use:**
```
My corn crop did well last year. Given the current field conditions and weather forecast for Chicago this month, should I plant corn again or try a different crop?
```

**Settings:**
- Image: No
- Memory Toggle: On

**Expected Tools Used:** queryKodeeMemories, google_search

## Scenario 4: Low Fertility and High Disease Crisis - worked

**Description:** Showcase soil recovery capabilities for severe soil issues

**Field Values to Set:**
- Field 1:
  - Temperature: 70
  - Humidity: 50
  - Rain Forecast: 45
  - Soil Fertility: 15
  - Heat Wave: 30
  - Disease: 85

**Message to Use:**
```
Field 1 has very poor soil metrics and high disease levels. What's the most efficient approach to make it farmable again?
```

**Settings:**
- Image: No
- Memory Toggle: Off

**Expected Tools Used:** soil_recovery_treatment, organic_treatment

## Scenario 5: Sustainable Farming in Drought Conditions

**Description:** Demonstrate rainwater harvesting and shade management for drought

**Field Values to Set:**
- All Fields:
  - Temperature: 85
  - Humidity: 25
  - Rain Forecast: 5
  - Soil Fertility: 60
  - Heat Wave: 60
  - Disease: 30

**Message to Use:**
```
Weather forecasts show we're entering a drought period with no rain expected for the next two weeks. How can I sustain my crops across all fields using sustainable methods?
```

**Settings:**
- Image: No
- Memory Toggle: Off

**Expected Tools Used:** rainwater_harvesting, toggle_shade, possibly organic_treatment

---

## Instructions for Running Demonstrations

1. Set the field values according to the scenario specifications
2. Configure any required images or memory toggle settings
3. Enter the provided message exactly as written
4. Observe Kodee's analysis and tool usage
5. Note how field metrics change in response to the applied tools