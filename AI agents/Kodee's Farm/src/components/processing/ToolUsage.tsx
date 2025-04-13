import React from "react";

interface Tool {
  name: string;
  status: string;
}

interface ToolChain {
  id: number;
  tools: Tool[];
}

interface ToolUsageProps {
  toolChains: ToolChain[];
}

export const ToolUsage: React.FC<ToolUsageProps> = ({ toolChains }) => {
  return (
    <div className="bg-white p-2 rounded-lg shadow-sm">
      {toolChains.map((chain) => (
        <div key={chain.id} className="mb-2">
          <div className="flex overflow-x-auto gap-2">
            {chain.tools.map((tool, index) => (
              <React.Fragment key={index}>
                <span 
                  className="px-2 py-1 text-sm bg-gray-100 rounded text-gray-700 whitespace-nowrap"
                >
                  {tool.name}
                </span>
                {index < chain.tools.length - 1 && (
                  <span className="text-gray-400">→</span>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};