export interface ChatMessage {
  type: 'user' | 'bot';
  text: string;
}