export interface ChatMessage {
  type: "user" | "bot";
  text: string;
  imageUrl?: string; // Optional property for image URL
}