import React, { useState } from "react";
import { GoFileSymlinkFile } from "react-icons/go";
import { ImUpload } from "react-icons/im";
import { IoSend } from "react-icons/io5";

interface ChatInputProps {
  onSubmit: (message: string, imageUrl?: string | null) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSubmit }) => {
  const [message, setMessage] = useState("");
  const [attachmentEnabled, setAttachmentEnabled] = useState(false); // State for toggling attachment
  const [uploadedImage, setUploadedImage] = useState<File | null>(null); // State for storing uploaded image

  const handleSend = () => {
    if (message.trim() !== "" || uploadedImage) {
      const imageUrl = uploadedImage ? URL.createObjectURL(uploadedImage) : null; // Create a temporary URL for the image
      onSubmit(message, imageUrl);
      setMessage(""); // Clear the input after sending
      setUploadedImage(null); // Clear the uploaded image after sending
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedImage(file); // Store the uploaded file in state
      console.log("Image uploaded:", file.name); // Log the file name for debugging
    }
  };

  return (
    <div className="flex flex-col space-y-2">
      {/* Image Upload Indicator */}
      {uploadedImage && (
        <div className="text-sm text-green-600 font-medium">
          <ImUpload className="inline-block mr-1" /> {uploadedImage.name}
        </div>
      )}

      <div className="flex items-center space-x-2">
        {/* Attachment Toggle Button */}
        <button
          onClick={() => setAttachmentEnabled(!attachmentEnabled)}
          className={`p-2 rounded-full ${
            attachmentEnabled ? "bg-green-700 text-white" : "bg-gray-200 text-gray-600"
          }`}
          title="Toggle Attachment"
        >
          <GoFileSymlinkFile size={20} />
        </button>

        {/* Image Upload Button */}
        <label
          className={`p-2 rounded-full cursor-pointer ${
            uploadedImage ? "bg-green-700 text-white" : "bg-gray-200 text-gray-600"
          }`}
          title="Upload Image"
        >
          <ImUpload size={20} />
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload}
          />
        </label>

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
          className="p-2 bg-green-700 text-white rounded-full hover:bg-green-900 focus:outline-none"
          title="Send Message"
        >
          <IoSend size={20} />
        </button>
      </div>
    </div>
  );
};