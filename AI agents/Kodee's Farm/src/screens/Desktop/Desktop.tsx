import { useState, useCallback, useEffect } from "react";
import { Slider } from "@mui/material"; // Import Material-UI Slider
import { ChatMessages } from "../../components/chat/ChatMessages";
import { ChatInput } from "../../components/chat/ChatInput";
import Field  from "../../components/ui/Field";
import { ToolUsage } from "../../components/processing/ToolUsage";
import { ChatMessage } from '../../types';
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
  humidity: [40, 100],
  rain_forecast: [20, 80],
  soil_fertility: [60, 100],
  heat_wave: [0, 40],
  disease: [0, 30],
};

function calculateHealthScore(env: Environment): number {
  let score = 0;
  for (const [key, value] of Object.entries(env)) {
    if (value !== undefined) { // Ensure value is not undefined
      const [min, max] = optimalRanges[key as keyof typeof optimalRanges] || [];
      if (min !== undefined && max !== undefined) {
        if (value < min) score += min - value;
        else if (value > max) score += value - max;
      }
    }
  }
  return score;
}

function getFieldColor(score: number): string {
  if (score < 20) return "green";
  if (score < 30) return "yellow";
  if (score < 40) return "lightorange"; // New step
  if (score < 50) return "orange";
  if (score < 60) return "darkorange"; // New step
  if (score < 70) return "red";
  return "darkred"; // New step
}

export const Desktop = (): JSX.Element => {
  // Field data with associated metrics
  const initialFields: Field[] = [
    {
      id: 1,
      name: "Field 1",
      metrics: [
        { name: "temperature", value: 80 },
        { name: "humidity", value: 20 },
        { name: "Rain Forecast", value: 20 },
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
        { name: "Heat Wave", value: 55 },
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
        { name: "Rain Forecast", value: 70 },
        { name: "Soil Fertility", value: 90 },
        { name: "Heat Wave", value: 60 },
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
  const [fields, setFields] = useState<Field[]>(() => evaluateConditions(initialFields));

  useEffect(() => {
    setFields((prevFields) => evaluateConditions(prevFields)); // Re-evaluate conditions on updates
  }, [evaluateConditions]);

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { type: "bot", text: "Hello! How can I help you with your crops today?" },
  ]);

  // Thinking messages state
  const [thinkingMessages, setThinkingMessages] = useState([
    "Processing: hi",
    "Analyzing crop data...",
  ]);

  // Tool usage state
  const [toolChains, setToolChains] = useState([
    {
      id: 1,
      tools: [
        { name: "Weather API", status: "completed" },
        { name: "Soil Analysis", status: "completed" },
      ],
    },
  ]);

  // State to track the active toggle button
  const [activeToggle, setActiveToggle] = useState<'user' | 'robot'>('user');

  // Handle chat submission
  const handleChatSubmit = (message: string, imageUrl?: string | null) => {
    setChatMessages((prev) => [
      ...prev,
      { type: "user", text: message, imageUrl: imageUrl || undefined },
    ]);

    // Derive environmentState from fields
    const environmentState = fields.reduce((acc, field) => {
      field.metrics.forEach((metric) => {
        acc[metric.name.toLowerCase()] = metric.value;
      });
      return acc;
    }, {} as Record<string, number>);

    // Make a fetch API call to the provided URL
    fetch("https://cors-anywhere.herokuapp.com/https://webhook.site/d68e5135-b0f0-421d-9a73-8aa33443c10e", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        "message": "hello",
      }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.text();
      })
      .then((data) => {
        console.log("API response:", data);
      })
      .catch((error) => {
        console.error("Error while making API call:", error);
      });

    setTimeout(() => {
      setChatMessages((prev) => [
        ...prev,
        {
          type: "bot",
          text: "I'm analyzing your message about the crops. Let me process that information and provide you with a detailed response.",
        },
      ]);

      // Refresh the thoughts tab with new messages
      setThinkingMessages([
        `Processing your input: "${message}"`,
        "Analyzing crop data...",
        "Checking soil moisture levels...",
        "Evaluating weather patterns...",
        "Generating insights...",
      ]);

      // Add a new tool chain dynamically
      setToolChains((prev) => [
        {
          id: prev.length + 1,
          tools: [
            { name: "Tool 1", status: "completed" },
            { name: "Tool 2", status: "completed" },
            { name: "Tool 3", status: "completed" },
          ],
        },
        ...prev,
      ]);
    }, 500);
  };

  // Handle slider change for metrics
  const handleMetricChange = useCallback(
    (fieldId: number, metricIndex: number, newValue: number) => {
      setFields((prevFields) => {
        const updatedFields = prevFields.map((field) =>
          field.id === fieldId
            ? {
                ...field,
                metrics: field.metrics.map((metric, i) =>
                  i === metricIndex ? { ...metric, value: newValue } : metric
                ),
              }
            : field
        );
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
          style={{ fontFamily: 'Helvetica_Neue-Bold, Helvetica' }}
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
                  <ChatInput onSubmit={handleChatSubmit} />
                </div>
              </div>
              
                {/* Mascot video and toggle button container */}
                <div className="absolute top-4 left-4 flex flex-col items-center">
                {/* Mascot video */}
                <video
                  className="w-[80px] h-[100px] object-cover"
                  autoPlay
                  loop
                  muted
                >
                  <source src="/idle.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>

                {/* Agent mode Toggle button */}
                <button
                  className={`mt-2 p-2 rounded-full flex items-center justify-center w-10 h-10 ${
                  activeToggle === 'robot' ? 'bg-[#30792e] text-white' : 'bg-gray-200'
                  }`}
                  title="Toggle"
                  onClick={() => setActiveToggle(activeToggle === 'robot' ? 'user' : 'robot')}
                >
                  <div className="relative flex items-center justify-center">
                  <BsRobot className="text-xl" />
                  {activeToggle === 'robot' && (
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
                  <div key={field.id} className="p-4 border rounded-lg shadow-sm">
                    <div className="font-bold text-lg mb-2">{field.name}</div>
                    <div className="grid grid-cols-6 gap-4">
                      {field.metrics.map((metric, metricIndex) => (
                        <div key={metricIndex} className="flex flex-col items-center">
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
                              handleMetricChange(field.id, metricIndex, newValue as number)
                            }
                            sx={{
                              color: "#30792e", 
                              width: "100%",
                            }}
                          />
                          {/* Metric Value */}
                          <div className="text-xs mt-1 text-gray-500">{metric.value}%</div>
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
                style={{ maxHeight: "200px", overflowY: "auto", boxSizing: "border-box" }} // Ensure content fits within the container
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