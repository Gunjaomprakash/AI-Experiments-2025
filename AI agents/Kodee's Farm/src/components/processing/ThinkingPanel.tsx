import React from 'react';
import { Card, CardContent } from '../ui/card';

interface ThinkingPanelProps {
  thinkingMessages: string[];
}

export const ThinkingPanel: React.FC<ThinkingPanelProps> = ({ thinkingMessages }) => {
  return (
    <div
      className="absolute w-[598px] h-[150px] top-[579px] left-[781px] overflow-y-auto"
      style={{
        scrollbarWidth: 'thin',
        scrollbarColor: '#30792e #f0f0f0',
      }}
    >
      {thinkingMessages.map((message, index) => (
        <Card key={index} className="mb-2 bg-white rounded-[14px] border border-solid border-black">
          <CardContent className="p-4">
            <div className="font-normal text-[#1b2559] text-base">{message}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};