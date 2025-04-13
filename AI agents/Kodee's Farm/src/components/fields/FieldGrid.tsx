import React from "react";

interface Field {
  id: number;
  name: string;
}

interface FieldGridProps {
  fields: Field[];
}

export const FieldGrid: React.FC<FieldGridProps> = ({ fields }) => {
  // Create a grid of dots for each field
  const renderDotGrid = () => {
    // Create a 20x10 grid of dots
    const rows = 10;
    const cols = 25;
    
    return (
      <div className="grid grid-cols-25 gap-[3px]">
        {Array.from({ length: rows * cols }).map((_, index) => (
          <div 
            key={index} 
            className="w-[3px] h-[3px] bg-green-500 rounded-full"
          />
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-8">
      {fields.map((field) => (
        <div key={field.id} className="space-y-2">
          <div className="font-medium text-[#1b2559]">{field.name}</div>
          <div className="border border-green-500 p-1 inline-block">
            <div className="flex flex-wrap gap-[1px]">
              {Array.from({ length: 350 }).map((_, index) => (
                <div 
                  key={index} 
                  className="w-[3px] h-[3px] bg-green-500 rounded-full"
                />
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};