# Example Prompt for Multiple Tool Calls

The following prompt is designed to demonstrate Kodee's capabilities in handling complex farming scenarios requiring multiple sequential tool calls.

## Recommended Prompt:

```
I've uploaded a photo of a banana plant with yellow leaves. Can you analyze what might be wrong with it, tell me the current growing conditions for bananas in Chicago this month, and recommend actions to improve Field 1's health based on both the image analysis and current field metrics?
```

## What This Prompt Demonstrates:

1. **Image Analysis**: Kodee will analyze the uploaded banana plant image to identify issues
2. **Information Retrieval**: Kodee will use Google Search to find current growing conditions for bananas in Chicago
3. **Current State Assessment**: Kodee will analyze the current metrics of Field 1
4. **Recommendation & Action**: Kodee will recommend and execute appropriate tools to improve field health

## Tool Call Sequence:

1. `record_execution` - Log the initial analysis plan
2. `image_analysis` - Analyze the uploaded banana plant image
3. `record_execution` - Log findings from the image
4. `google_search` - Research banana growing conditions in Chicago for the current month
5. `record_execution` - Log findings from the search
6. `humidify_field` or other tools - Take corrective actions based on findings
7. `boost_fertilizer` - Potentially adjust soil nutrients based on image analysis
8. `record_execution` - Log the final recommendations and actions taken

## Expected Output:

Kodee will provide a comprehensive analysis including:
- Diagnosis of the plant issue (likely nutrient deficiency, pest, or disease)
- Current ideal growing conditions for bananas in Chicago this month
- How the current field metrics compare to ideal conditions
- Specific actions taken to improve Field 1's health
- Changes in field metrics after applying the interventions