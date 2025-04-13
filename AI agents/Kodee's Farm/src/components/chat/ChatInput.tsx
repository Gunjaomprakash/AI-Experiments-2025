import React, { useState } from "react";
import { FiSend, FiPaperclip } from "react-icons/fi"; // Import icons from react-icons

interface ChatInputProps {
  onSubmit: (message: string) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSubmit }) => {
  const [message, setMessage] = useState("");
  const [attachmentEnabled, setAttachmentEnabled] = useState(false); // State for toggling attachment

  const handleSend = () => {
    if (message.trim() !== "") {
      onSubmit(message);
      setMessage(""); // Clear the input after sending
    }
  };

  return (
    <div className="flex items-center space-x-2">
      {/* Attachment Toggle Button */}
      <button
        onClick={() => setAttachmentEnabled(!attachmentEnabled)}
        className={`p-2 rounded-full ${
          attachmentEnabled ? "bg-green-500 text-white" : "bg-gray-200 text-gray-600"
        }`}
        title="Toggle Attachment"
      >
        <FiPaperclip size={20} />
      </button>

      {/* Input Field */}
      <input
        type="text"
        className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
        placeholder="Help me with..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />

      {/* Send Button */}
      <button
        onClick={handleSend}
        className="p-2 bg-green-500 text-white rounded-full hover:bg-green-600 focus:outline-none"
        title="Send Message"
      >
        <FiSend size={20} />
      </button>
    </div>
  );
};