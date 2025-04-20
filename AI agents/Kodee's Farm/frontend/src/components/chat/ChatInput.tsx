import React, { useState, useRef, useEffect } from "react";
import { GoFileSymlinkFile } from "react-icons/go";
import { ImUpload } from "react-icons/im";
import { IoSend } from "react-icons/io5";

interface ChatInputProps {
  onSubmit: (message: string, imageUrl?: string | null, attachmentEnabled?: boolean) => void;
  onImageSelect?: (file: File | null) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSubmit, onImageSelect }) => {
  const [message, setMessage] = useState("");
  const [attachmentEnabled, setAttachmentEnabled] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Handle paste events
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          const file = items[i].getAsFile();
          if (file) {
            console.log("Image pasted:", file.name);
            setUploadedImage(file);
            if (onImageSelect) {
              onImageSelect(file);
            }
            // Prevent the default paste behavior for images
            e.preventDefault();
            break;
          }
        }
      }
    };

    // Add the event listener to the document
    document.addEventListener("paste", handlePaste);

    // Clean up
    return () => {
      document.removeEventListener("paste", handlePaste);
    };
  }, [onImageSelect]);

  const handleSend = () => {
    if (message.trim() !== "" || uploadedImage) {
      const imageUrl = uploadedImage ? URL.createObjectURL(uploadedImage) : null;
      onSubmit(message, imageUrl, attachmentEnabled);
      setMessage("");
      setUploadedImage(null);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setUploadedImage(file); // Store the uploaded file in state
    if (onImageSelect) {
      onImageSelect(file); // Notify parent component
    }
    console.log("Image uploaded:", file?.name); // Log the file name for debugging
  };

  // Handle key press events (Enter to send)
  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col space-y-2">
      {/* Image Upload Indicator */}
      {uploadedImage && (
        <div className="text-sm text-green-600 font-medium">
          <ImUpload className="inline-block mr-1" /> {uploadedImage.name || "Pasted image"}
        </div>
      )}

      <div className="flex items-center space-x-2">
        {/* RAGAttachment Toggle Button */}
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
            onChange={handleFileChange} // Use handleFileChange here
          />
        </label>

        {/* Input Field */}
        <input
          ref={inputRef}
          type="text"
          className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
          placeholder="Type or paste an image here..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
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