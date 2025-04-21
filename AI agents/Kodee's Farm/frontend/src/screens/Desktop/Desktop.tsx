import { useState, useCallback, useEffect } from "react";
import { Slider } from "@mui/material"; // Import Material-UI Slider
import { ChatMessages } from "../../components/chat/ChatMessages";
import { ChatInput } from "../../components/chat/ChatInput";
import Field from "../../components/ui/Field";
import { ToolUsage } from "../../components/processing/ToolUsage";
import { ChatMessage } from "../../types";
import { BsRobot } from "react-icons/bs";

interface Metric {
  name: string;
  value: number;
}

interface Field {
  id: number;
  name: string;
  metrics: Metric[];
  color: string;
}

interface Environment {
  temperature?: number;
  humidity?: number;
  rain_forecast?: number;
  soil_fertility?: number;
  heat_wave?: number;
  disease?: number;
}

const optimalRanges = {
  temperature: [60, 80],
  humidity: [40, 70],
  rain_forecast: [20, 80],
  soil_fertility: [60, 100],
  heat_wave: [0, 40],
  disease: [0, 25],
};

function calculateHealthScore(env: Environment): number {
  let score = 0;
  const weights = {
    temperature: 1.2,
    humidity: 1.2,
    soil_fertility: 1.5,
    disease: 2.0,
    heat_wave: 0.8,
    rain_forecast: 0.8,
  };
  
  for (const [key, value] of Object.entries(env)) {
    if (value !== undefined) {
      const [min, max] = optimalRanges[key as keyof typeof optimalRanges] || [];
      if (min !== undefined && max !== undefined) {
        const weight = weights[key as keyof typeof weights] || 1.0;
        if (value < min) score += (min - value) * weight;
        else if (value > max) score += (value - max) * weight;
      }
    }
  }
  return score;
}

function getFieldColor(score: number): string {
  if (score < 15) return "#008000"; // Healthy green
  if (score < 25) return "#90EE90"; // Light green
  if (score < 35) return "#FFFF00"; // Yellow
  if (score < 45) return "#FFA500"; // Orange
  if (score < 55) return "#FF4500"; // Orange red
  if (score < 65) return "#FF0000"; // Red
  if (score < 75) return "#8B0000"; // Dark red
  return "#800080"; // Purple (severely unhealthy)
}

const processSequentially = async (
  items: { timestamp: number }[],
  onItem: (item: any) => void
) => {
  // Sort items by timestamp
  const sortedItems = [...items].sort((a, b) => a.timestamp - b.timestamp);

  // Sequential replay with delay
  for (const item of sortedItems) {
    await new Promise((resolve) => setTimeout(resolve, 500)); // 500 ms = 0.5 sec
    onItem(item);
  }
};

// Helper function to clean agent responses
const cleanAgentResponse = (message: string): string => {
  // Check for the specific prefix and remove it if found
  const prefix = "Response (no formatted final message found):";
  if (message.startsWith(prefix)) {
    return message.substring(prefix.length).trim();
  }
  return message;
};

export const Desktop = (): JSX.Element => {
  // Add new state for video source and visibility
  const [videoSource, setVideoSource] = useState<string>("/idle.mp4");
  const [videoVisible, setVideoVisible] = useState<boolean>(true);

  // Function to handle video source change with fade effect
  const changeVideoSource = (newSource: string) => {
    setVideoVisible(false); // Start fade-out
    setTimeout(() => {
      setVideoSource(newSource); // Change video source
      setVideoVisible(true); // Start fade-in
    }, 300); // Match the CSS transition duration
  };

  // Field data with associated metrics
  const initialFields: Field[] = [
    {
      id: 1,
      name: "Field 1",
      metrics: [
        { name: "temperature", value: 80 },
        { name: "humidity", value: 20 },
        { name: "Rain Forecast", value: 60 },
        { name: "Soil Fertility", value: 80 },
        { name: "Heat Wave", value: 50 },
        { name: "Disease", value: 50 },
      ],
      color: "#30792e", // Default color
    },
    {
      id: 2,
      name: "Field 2",
      metrics: [
        { name: "Temperature", value: 72 },
        { name: "Humidity", value: 75 },
        { name: "Rain Forecast", value: 60 },
        { name: "Soil Fertility", value: 85 },
        { name: "Heat Wave", value: 50 },
        { name: "Disease", value: 45 },
      ],
      color: "#30792e", // Default color
    },
    {
      id: 3,
      name: "Field 3",
      metrics: [
        { name: "Temperature", value: 65 },
        { name: "Humidity", value: 78 },
        { name: "Rain Forecast", value: 60 },
        { name: "Soil Fertility", value: 90 },
        { name: "Heat Wave", value: 50 },
        { name: "Disease", value: 40 },
      ],
      color: "#30792e", // Default color
    },
  ];

  const evaluateConditions = useCallback((fields: Field[]): Field[] => {
    return fields.map((field) => {
      const env = Object.fromEntries(
        field.metrics.map((metric) => [metric.name.toLowerCase(), metric.value])
      );
      const healthScore = calculateHealthScore(env);
      const newColor = getFieldColor(healthScore);
      return { ...field, color: newColor };
    });
  }, []);

  // Initialize fields with evaluated conditions
  const [fields, setFields] = useState<Field[]>(() =>
    evaluateConditions(initialFields)
  );

  useEffect(() => {
    setFields((prevFields) => evaluateConditions(prevFields)); // Re-evaluate conditions on updates
  }, [evaluateConditions]);

  // Process field snapshots sequentially
  const processFieldSnapshots = async (snapshots: any[]) => {
    if (!snapshots || snapshots.length === 0) {
      console.log("No field snapshots to process");
      return;
    }
    
    console.log("Processing field snapshots:", snapshots);
    
    // Sort snapshots by timestamp
    const sortedSnapshots = [...snapshots].sort((a, b) => a.timestamp - b.timestamp);
    
    for (const snapshot of sortedSnapshots) {
      // Wait before applying the next snapshot
      await new Promise((resolve) => setTimeout(resolve, 800));
      
      console.log("Processing snapshot:", snapshot);
      
      // Check if snapshot has the expected structure
      if (!snapshot || !snapshot.fields) {
        console.warn("Invalid snapshot format:", snapshot);
        continue;
      }
      
      try {
        console.log(`Applying snapshot at timestamp ${snapshot.timestamp}:`, snapshot.fields);
        
        // Update the fields with the new metrics
        setFields((prevFields: Field[]) => {
          // Create a new array of fields with updated metrics
          const updatedFields = prevFields.map((currentField: Field) => {
            // Find the corresponding field in the snapshot
            const snapshotField = snapshot.fields.find((f: any) => f.id === currentField.id);
            
            if (snapshotField && snapshotField.metrics && snapshotField.metrics.length > 0) {
              // Create a mapping of metric names to values from the snapshot
              const metricMap = new Map<string, number>();
              snapshotField.metrics.forEach((metric: any) => {
                metricMap.set(metric.name.toLowerCase(), metric.value);
              });
              
              // Update the metrics in the current field
              return {
                ...currentField,
                metrics: currentField.metrics.map(metric => {
                  const newValue = metricMap.get(metric.name.toLowerCase());
                  return newValue !== undefined ? { ...metric, value: newValue } : metric;
                })
              };
            }
            return currentField;
          });
          
          return evaluateConditions(updatedFields);
        });
      } catch (error) {
        console.error("Error processing field snapshot:", error);
      }
    }
  };

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { type: "bot", text: "Hello! How can I help you with your crops today?" },
  ]);

  // Thinking messages state
  const [thinkingMessages, setThinkingMessages] = useState<string[]>([]); // Explicitly define the type as an array of strings

  // Tool usage state
  const [toolChains, setToolChains] = useState<{ id: number; tools: { name: string; status: string }[] }[]>([]);

  // State to track the active toggle button
  const [activeToggle, setActiveToggle] = useState<"user" | "robot">("user");

  // Add state for selected image file
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);

  // Handle chat submission
  const handleChatSubmit = (
    message: string,
    imageUrl?: string | null,
    attachmentEnabled?: boolean
  ): void => {
    // Add user message immediately
    setChatMessages((prev) => [
      ...prev,
      { type: "user", text: message, imageUrl: imageUrl || undefined },
    ]);

    // Clear previous state
    setThinkingMessages([]);
    setToolChains([]);

    // Set appropriate video based on attachment mode
    changeVideoSource(attachmentEnabled ? "/working.mp4" : "/thinking.mp4");

    const formData = new FormData();
    formData.append("userMessage", message);
    formData.append("activeToggle", activeToggle);
    formData.append("attachmentEnabled", String(attachmentEnabled)); // This controls RAG, not image
    formData.append("fields", JSON.stringify(fields));

    // Use selectedImageFile from state
    if (selectedImageFile) {
      formData.append("image", selectedImageFile); // This is handled separately
    }

    fetch("http://127.0.0.1:5000/api/agent", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) throw new Error("Network error");
        return response.json();
      })
      .then(async (data) => {
        console.log("Agent response:", data);
        console.log("Field snapshots:", data.fieldsSnapshots); 

        // Create an integrated timeline of all events (thoughts, tools, and field changes)
        let integratedTimeline: {type: string; data: any; timestamp: number}[] = [];
        
        // Add thoughts to timeline
        if (data.thoughtsList && data.thoughtsList.length > 0) {
          integratedTimeline = [
            ...integratedTimeline,
            ...data.thoughtsList.map((thought: any) => ({
              type: "thought",
              data: thought,
              timestamp: thought.timestamp
            }))
          ];
        }
        
        // Add tools to timeline
        if (data.toolList && data.toolList.length > 0) {
          integratedTimeline = [
            ...integratedTimeline,
            ...data.toolList.map((tool: any) => ({
              type: "tool",
              data: tool,
              timestamp: tool.timestamp
            }))
          ];
        }
        
        // Add field snapshots to timeline
        if (data.fieldsSnapshots && data.fieldsSnapshots.length > 0) {
          integratedTimeline = [
            ...integratedTimeline,
            ...data.fieldsSnapshots.map((snapshot: any) => ({
              type: "snapshot",
              data: snapshot,
              timestamp: snapshot.timestamp
            }))
          ];
        }

        // Sort the integrated timeline by timestamp
        integratedTimeline.sort((a, b) => a.timestamp - b.timestamp);
        
        console.log("Integrated timeline:", integratedTimeline);
        
        // Process the integrated timeline sequentially
        for (const event of integratedTimeline) {
          // Add a small delay between events
          await new Promise((resolve) => setTimeout(resolve, 500));
          
          switch(event.type) {
            case "thought":
              setThinkingMessages((prev) => [...prev, event.data.text]);
              break;
              
            case "tool":
              // Update tool chains
              setToolChains((prev) => {
                let chain = prev[0] || { id: 1, tools: [] };
                const toolName = event.data.tool.replace("default_api.", "");
                
                // Only add tool if it's not record_execution
                if (toolName !== "record_execution") {
                  chain = {
                    ...chain,
                    tools: [...chain.tools, { name: toolName, status: "completed" }]
                  };
                }
                
                return [chain];
              });
              break;
              
            case "snapshot":
              // Apply field snapshot
              try {
                const snapshot = event.data;
                if (snapshot && snapshot.fields) {
                  setFields((prevFields: Field[]) => {
                    // Create a new array of fields with updated metrics
                    const updatedFields = prevFields.map((currentField: Field) => {
                      // Find the corresponding field in the snapshot
                      const snapshotField = snapshot.fields.find((f: any) => f.id === currentField.id);
                      
                      if (snapshotField && snapshotField.metrics && snapshotField.metrics.length > 0) {
                        // Create a mapping of metric names to values from the snapshot
                        const metricMap = new Map<string, number>();
                        snapshotField.metrics.forEach((metric: any) => {
                          metricMap.set(metric.name.toLowerCase(), metric.value);
                        });
                        
                        // Update the metrics in the current field
                        return {
                          ...currentField,
                          metrics: currentField.metrics.map(metric => {
                            const newValue = metricMap.get(metric.name.toLowerCase());
                            return newValue !== undefined ? { ...metric, value: newValue } : metric;
                          })
                        };
                      }
                      return currentField;
                    });
                    
                    return evaluateConditions(updatedFields);
                  });
                }
              } catch (error) {
                console.error("Error processing field snapshot:", error);
              }
              break;
          }
        }

        // Add the final message and reset video to dance.mp4 temporarily
        setChatMessages((prev) => [
          ...prev,
          { type: "bot", text: cleanAgentResponse(data.finalMessage || "No response received.") },
        ]);
        changeVideoSource("/dance.mp4"); // Play dance.mp4
        setTimeout(() => {
          changeVideoSource("/idle.mp4"); // Reset to idle.mp4 after a few seconds
        }, 3000); // 3 seconds delay
      })
      .catch((err) => {
        console.error("Error:", err);
        changeVideoSource("/idle.mp4"); // Reset video on error
      });
  };

  // Handler to update selected image file
  const handleImageSelect = (file: File | null) => {
    setSelectedImageFile(file);
    if (file) {
      // Reset the input value to allow re-uploading the same file
      const inputElement = document.querySelector('input[type="file"]') as HTMLInputElement;
      if (inputElement) {
        inputElement.value = "";
      }
    }
  };

  // Handle slider change for metrics
  const handleMetricChange = useCallback(
    (fieldId: number, metricIndex: number, newValue: number) => {
      setFields((prevFields) => {
        const updatedFields = prevFields.map((field) => {
          if (field.id === fieldId) {
            return {
              ...field,
              metrics: field.metrics.map((metric, i) =>
                i === metricIndex ? { ...metric, value: newValue } : metric
              ),
            };
          } else {
            // Synchronize Rain Forecast and Heat Wave across all fields
            const metricName = prevFields[fieldId - 1].metrics[metricIndex].name.toLowerCase();
            if (metricName === "rain forecast" || metricName === "heat wave") {
              return {
                ...field,
                metrics: field.metrics.map((metric) =>
                  metric.name.toLowerCase() === metricName
                    ? { ...metric, value: newValue }
                    : metric
                ),
              };
            }
          }
          return field;
        });
        return evaluateConditions(updatedFields); // Re-evaluate conditions after updating metrics
      });
    },
    [evaluateConditions]
  );

  return (
    <div className="bg-white flex flex-row justify-center w-full">
      <div className="bg-white w-[1440px] h-[1024px]">
        {/* Title at the top */}
        <div
          className="text-center py-4 font-bold text-4xl text-[#1b2559]"
          style={{ fontFamily: "Helvetica_Neue-Bold, Helvetica" }}
        >
          Kodee's Farm
        </div>

        {/* Main layout with central divider */}
        <div className="relative h-[960px] flex">
          {/* Left side containing fields and chat */}
          <div className="w-1/2 flex flex-col">
            {/* Fields at the top */}
            <div className="flex-1 p-4 overflow-auto space-y-4">
              {fields.map((field) => (
                <Field
                  key={field.id}
                  label={field.name}
                  rows={10}
                  cols={60}
                  style={{ border: "1px solid #30792e", padding: "1px" }}
                  color={field.color} // Pass dynamic color
                />
              ))}
            </div>

            {/* Chat UI at the bottom */}
            <div className="flex-1 p-4 relative">
              <div className="h-full flex flex-col">
                <div className="flex-grow overflow-auto">
                  <ChatMessages chatMessages={chatMessages} />
                </div>
                <div className="mt-4">
                  {/* Pass handleImageSelect to ChatInput if it supports image upload */}
                  <ChatInput
                    onSubmit={handleChatSubmit}
                    onImageSelect={handleImageSelect}
                  />
                </div>
              </div>

              {/* Mascot video and toggle button container */}
              <div className="absolute top-4 left-4 flex flex-col items-center">
                {/* Mascot video */}
                <video
                  className={`w-[80px] h-[100px] object-cover transition-opacity duration-300 ${
                    videoVisible ? "opacity-100" : "opacity-0"
                  }`}
                  autoPlay
                  loop
                  muted
                  key={videoSource} // Add key to force reload when source changes
                >
                  <source src={videoSource} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>

                {/* Agent mode Toggle button */}
                <button
                  className={`mt-2 p-2 rounded-full flex items-center justify-center w-10 h-10 ${
                    activeToggle === "robot"
                      ? "bg-[#30792e] text-white"
                      : "bg-gray-200"
                  }`}
                  title="Toggle"
                  onClick={() =>
                    setActiveToggle(activeToggle === "robot" ? "user" : "robot")
                  }
                >
                  <div className="relative flex items-center justify-center">
                    <BsRobot className="text-xl" />
                    {activeToggle === "robot" && (
                      <div className="absolute inset-0 border-2 border-t-transparent border-[#ffffff] rounded-full animate-spin"></div>
                    )}
                  </div>
                </button>
              </div>
            </div>
          </div>

          {/* Central divider */}
          <div className="absolute h-full border-l-2 border-dashed border-gray-400 left-1/2"></div>

          {/* Right side containing metrics and processing */}
          <div className="w-1/2 flex flex-col">
            {/* Environmental Metrics for each field */}
            <div className="flex-1 p-4 ">
              <div className="grid grid-rows-3 gap-4">
                {fields.map((field) => (
                  <div
                    key={field.id}
                    className="p-4 border rounded-lg shadow-sm"
                  >
                    <div className="font-bold text-lg mb-2">{field.name}</div>
                    <div className="grid grid-cols-6 gap-4">
                      {field.metrics.map((metric, metricIndex) => (
                        <div
                          key={metricIndex}
                          className="flex flex-col items-center"
                        >
                          {/* Metric Name */}
                          <div className="text-sm font-medium text-gray-700">
                            {metric.name}
                          </div>
                          {/* Metric Slider */}
                          <Slider
                            value={metric.value}
                            min={0}
                            max={100}
                            onChange={(_, newValue) =>
                              handleMetricChange(
                                field.id,
                                metricIndex,
                                newValue as number
                              )
                            }
                            sx={{
                              color: "#30792e",
                              width: "100%",
                            }}
                          />
                          {/* Metric Value */}
                          <div className="text-xs mt-1 text-gray-500">
                            {metric.value}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Thoughts and Tool Usage at the bottom */}
            <div className="flex-1 p-4 grid grid-rows-2 gap-6">
              {/* Thoughts Section */}
              <div
                className="border border-gray-300 rounded-lg p-4"
                style={{
                  maxHeight: "200px",
                  overflowY: "auto",
                  boxSizing: "border-box",
                }} // Ensure content fits within the container
              >
                <div
                  className="font-medium text-[#1b2559] text-base mb-4"
                  style={{
                    fontFamily: "Helvetica_Neue-Medium, Helvetica",
                    position: "sticky", // Make the title sticky
                    top: 0, // Stick to the top of the container
                    backgroundColor: "white", // Match the background color
                    zIndex: 10, // Ensure it stays above other content
                  }}
                >
                  Thoughts
                </div>
                <div className="text-sm text-gray-700 space-y-2">
                  {thinkingMessages.map((message, index) => (
                    <div key={index}>{message}</div> // Render each message as plain text
                  ))}
                </div>
              </div>

              {/* Tool Usage Section */}
              <div
                className="border border-gray-300 rounded-lg p-4"
                style={{ maxHeight: "200px", overflowY: "auto" }} // Limit height and enable scrolling
              >
                <div
                  className="font-medium text-[#1b2559] text-base mb-4"
                  style={{
                    fontFamily: "Helvetica_Neue-Medium, Helvetica",
                    position: "sticky", // Make the title sticky
                    top: 0, // Stick to the top of the container
                    backgroundColor: "white", // Match the background color
                    zIndex: 10, // Ensure it stays above other content
                  }}
                >
                  Tool Usage
                </div>
                <ToolUsage toolChains={toolChains} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
