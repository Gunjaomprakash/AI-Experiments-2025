import { useState, useCallback, useEffect } from "react";
import { Slider } from "@mui/material"; // Import Material-UI Slider
import { ChatMessages } from "../../components/chat/ChatMessages";
import { ChatInput } from "../../components/chat/ChatInput";
import Field  from "../../components/ui/Field";
import { ToolUsage } from "../../components/processing/ToolUsage";
import { ChatMessage } from '../../types';
import conditions from "../../docs/conditions.json"; // Import conditions JSON

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

export const Desktop = (): JSX.Element => {
  // Function to evaluate conditions and update field color
  const evaluateConditions = useCallback((fields: Field[]): Field[] => {
    return fields.map((field) => {
      let newColor = "#30792e"; // Default color

      for (const condition of conditions) {
        const { trigger_conditions, field_color } = condition;
        const matches = Object.entries(trigger_conditions).every(([metricName, condition]) => {
          const metric = field.metrics.find((m) => m.name === metricName);
          if (!metric) return false;

          if (condition.gt !== undefined && metric.value <= condition.gt) return false;
          if (condition.lt !== undefined && metric.value >= condition.lt) return false;

          return true;
        });

        if (matches) {
          newColor = field_color;
          break;
        }
      }

      return { ...field, color: newColor };
    });
  }, []);

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

  // Handle chat submission
  const handleChatSubmit = (message: string, imageUrl?: string | null) => {
    setChatMessages((prev) => [
      ...prev,
      { type: "user", text: message, imageUrl: imageUrl || undefined }, // Ensure imageUrl is undefined if null
    ]);

    setTimeout(() => {
      setChatMessages((prev) => [
        ...prev,
        {
          type: "bot",
          text: "I'm analyzing your message about the crops. Let me process that information and provide you with a detailed response.",
        },
      ]);

      setThinkingMessages((prev) => [
        ...prev,
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
        ...prev, // Prepend the new tool chain
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
                  cols={75}
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
              
              {/* Mascot video positioned at the top-left of the chat */}
              <video
                className="absolute w-[80px] h-[100px] top-4 left-4 object-cover"
                autoPlay
                loop
                muted
              >
                <source src="/idle.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
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
                              color: "#516348", 
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