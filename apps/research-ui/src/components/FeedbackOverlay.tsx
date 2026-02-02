import React, { useState } from 'react';
import { MessageSquarePlus, X, Send } from 'lucide-react';
import { useCopilotAction } from "@copilotkit/react-core";

interface FeedbackOverlayProps {
  artifactId: string;
  children: React.ReactNode;
  onFeedbackSubmit: (artifactId: string, comment: string) => void;
}

export function FeedbackOverlay({ artifactId, children, onFeedbackSubmit }: FeedbackOverlayProps) {
  const [isCommenting, setIsCommenting] = useState(false);
  const [comment, setComment] = useState('');
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const handleRightClick = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsCommenting(true);
    setPosition({ x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY });
  };

  const handleSubmit = () => {
    if (comment.trim()) {
      onFeedbackSubmit(artifactId, comment);
      setComment('');
      setIsCommenting(false);
    }
  };

  // Define Copilot Action to handle feedback programmatically
  useCopilotAction({
    name: "provideArtifactFeedback",
    description: "Provide user feedback on a specific research artifact.",
    parameters: [
      { name: "artifactId", type: "string", description: "ID of the artifact" },
      { name: "feedback", type: "string", description: "The user's feedback or correction" },
    ],
    handler: async ({ artifactId, feedback }) => {
      onFeedbackSubmit(artifactId, feedback);
      return "Feedback recorded and sent to research agent.";
    },
  });

  return (
    <div className="relative group" onContextMenu={handleRightClick}>
      {children}

      {/* Hover Hint */}
      <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
        <span className="bg-am-primary/90 text-am-text-secondary text-[10px] px-2 py-1 rounded backdrop-blur-sm border border-am-border">
          Right-click to comment
        </span>
      </div>

      {/* Comment Box */}
      {isCommenting && (
        <div
          className="absolute z-50 w-64 bg-am-secondary rounded-lg shadow-xl border border-am-border p-3 animate-fade-in"
          style={{ top: position.y, left: position.x }}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-am-text-muted flex items-center gap-1">
              <MessageSquarePlus className="w-3 h-3" />
              Add Feedback
            </span>
            <button
              onClick={() => setIsCommenting(false)}
              className="text-am-text-muted hover:text-am-text-secondary transition-colors"
            >
              <X className="w-3 h-3" />
            </button>
          </div>

          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="What should the agent know?"
            className="w-full text-sm bg-am-primary border border-am-border rounded-md focus:border-am-accent focus:ring-1 focus:ring-am-accent min-h-[60px] resize-none mb-2 text-am-text-primary placeholder-am-text-muted p-2"
            autoFocus
          />

          <button
            onClick={handleSubmit}
            className="w-full flex items-center justify-center gap-2 bg-am-accent hover:bg-am-accent-hover text-white text-xs font-medium py-1.5 rounded-md transition-colors"
          >
            <Send className="w-3 h-3" />
            Send to Agent
          </button>
        </div>
      )}
    </div>
  );
}
