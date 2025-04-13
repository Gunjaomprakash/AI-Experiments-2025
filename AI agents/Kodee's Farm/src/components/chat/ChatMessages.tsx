import React, { useRef, useEffect } from 'react';
import { Card, CardContent } from '../ui/card';
import { ChatMessage } from '../../types';

interface ChatMessagesProps {
  chatMessages: ChatMessage[];
}

export const ChatMessages: React.FC<ChatMessagesProps> = ({ chatMessages }) => {
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  return (
    <div
      ref={chatContainerRef}
      className="absolute w-[400px] h-[350px] bottom-[130px] left-[104px] flex flex-col gap-4 overflow-y-auto pr-4"
      style={{
        scrollbarWidth: 'thin',
        scrollbarColor: '#30792e #f0f0f0',
      }}
    >
      {chatMessages.map((message, index) => (
        <div
          key={index}
          className={`flex ${
            message.type === 'user' ? 'justify-end' : 'justify-start'
          }`}
        >
          {message.type === 'bot' && (
            <div className="w-10 h-10 mr-2 bg-[#30792f] rounded-[20px] flex items-center justify-center flex-shrink-0">
              <img className="w-7 h-[30px]" alt="Bot" src="/vector.svg" />
            </div>
          )}
          <Card
            className={`${
              message.type === 'user'
                ? 'bg-[#30792e] text-white'
                : 'bg-white'
            } rounded-[14px] max-w-[300px]`}
          >
            <CardContent className="p-4">
              <p className="text-base break-words">{message.text}</p>
            </CardContent>
          </Card>
        </div>
      ))}
    </div>
  );
};