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
      className="absolute w-[500px] h-[350px] bottom-[130px] left-[104px] flex flex-col gap-4 overflow-y-auto pr-4"
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
          <Card
            className={`${
              message.type === 'user'
          ? 'bg-[#30792e] text-white'
          : 'bg-white'
            } rounded-[14px] max-w-[300px]`}
          >
            <CardContent className="p-4">
              <p className="text-base break-words">{message.text}</p>
              {message.imageUrl && (
          <img
            src={message.imageUrl}
            alt="Uploaded"
            className="mt-2 rounded-lg max-w-full"
          />
              )}
            </CardContent>
          </Card>
        </div>
      ))}
    </div>
  );
};