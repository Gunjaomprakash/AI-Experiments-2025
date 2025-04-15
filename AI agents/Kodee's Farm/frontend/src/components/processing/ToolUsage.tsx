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
    <div className="bg-white p-2 rounded-lg">
      {toolChains.map((chain) => (
        <div key={chain.id} className="mb-4">
          <div className="flex items-center flex-wrap gap-2">
            {chain.tools.map((tool, index) => (
              <React.Fragment key={index}>
                <div className="inline-flex items-center">
                  <div className="px-3 py-1.5 bg-green-50 border border-green-200 rounded-lg text-sm text-green-800 whitespace-nowrap">
                    {tool.name}
                  </div>
                  {index < chain.tools.length - 1 && (
                    <svg className="mx-2" width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <path d="M5 12h14m-7-7l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  )}
                </div>
              </React.Fragment>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};