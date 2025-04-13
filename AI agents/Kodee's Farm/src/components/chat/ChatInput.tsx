import React, { useState } from 'react';
import { Input } from '../ui/input';
import { Button } from '../ui/button';

interface ChatInputProps {
  onSubmit: (message: string) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSubmit }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = () => {
    if (!inputValue.trim()) return;
    onSubmit(inputValue);
    setInputValue('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="absolute bottom-[51px] left-[104px] flex items-center gap-4">
      <div className="relative w-[322px] h-[54px]">
        <Input
          className="w-full h-full rounded-[45px] border border-solid border-[#77797c] px-5 py-4 text-sm font-medium text-[#1b2559] pr-12"
          placeholder="Help me with..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyPress}
        />
        {/* Attachment Icon */}
        <button
          className="absolute right-4 top-1/2 transform -translate-y-1/2 text-[#77797c]"
          onClick={() => alert('Attachment icon clicked!')}
        >
          📎
        </button>
      </div>
      <Button
        className="w-[146px] h-[54px] bg-[#30792e] rounded-[45px] text-white text-sm font-semibold"
        onClick={handleSubmit}
      >
        Submit
      </Button>
    </div>
  );
};