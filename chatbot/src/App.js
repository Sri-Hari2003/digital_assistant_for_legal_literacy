import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import { useNavigate } from "react-router-dom";


const App = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const messagesEndRef = useRef(null);
  
const navigate = useNavigate();
  // Scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load messages from session on first load
  useEffect(() => {
    const savedMessages = sessionStorage.getItem("chatMessages");
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages));
    }
  }, []);

  // Save messages to session whenever they change
  useEffect(() => {
    sessionStorage.setItem("chatMessages", JSON.stringify(messages));
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
  
    const userMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
  
    // Handle 'yes' for fetching case data
    if (input.trim().toLowerCase() === "yes") {
      const lastBotMsg = [...messages]
        .reverse()
        .find((msg) => msg.sender === "bot" && !msg.text.includes("Would you like to"));
  
      if (lastBotMsg) {
        try {
          const res = await fetch("http://127.0.0.1:5000/get_case", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: lastBotMsg.text }),
          });
  
          const data = await res.json();
          if (data.results && data.results.length > 0) {
            const blobLinks = data.results.map((result) => {
              const blob = new Blob([result.content], { type: "text/html" });
              const url = URL.createObjectURL(blob);
              return { title: result.title, url };
            });
  
            setMessages((prev) => [
              ...prev,
              {
                sender: "bot",
                text: "Here are some related cases:",
                isHtml: false,
                links: blobLinks, // Attach link metadata
              },
            ]);
          } else {
            setMessages((prev) => [
              ...prev,
              { sender: "bot", text: "No related cases found." },
            ]);
          }
        } catch (err) {
          setMessages((prev) => [
            ...prev,
            { sender: "bot", text: "Error fetching case details." },
          ]);
        }
  
        setInput("");
        return;
      }
    }
  
    // Normal chatbot response
    try {
      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: input,
          api_key:
            "37d7a04f4f855143791edb2733d20b461f460c8a50ce75407210d049db58649e",
        }),
      });
  
      const data = await response.json();
      // const botText = (data.response || "No response").replace(/\n/g, "<br>");
      let botText = data.response || "No response";

      // Replace any text wrapped in '**' with <strong> tags
      botText = botText.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
      
      const botMessage = {
        sender: "bot",
        text: botText,
        isHtml: true,
      };
  
      setMessages((prev) => {
        const updated = [...prev, botMessage];
  
        const botCount = updated.filter((msg) => msg.sender === "bot").length;
  
        // After 2 bot responses, suggest checking for cases
        if (botCount % 1 === 0) {
          updated.push({
            sender: "bot",
            text: "Would you like to check if there are cases with context regarding this?",
          });
        }
  
        return updated;
      });
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error connecting to server." },
      ]);
    }
  
    setInput("");
  };
  

  return (
    <div className="chat-container">
      <div className="chat-box">
        <div className="messages">
        {messages.map((msg, index) => (
  <div key={index} className={`message ${msg.sender}`}>
    <strong>{msg.sender === "user" ? "You: " : " "}</strong>
    {msg.isHtml ? (
      <span dangerouslySetInnerHTML={{ __html: msg.text }} />
    ) : (
      <span>{msg.text}</span>
    )}

    {/* Render blob links if available */}
    {msg.links &&
      msg.links.map((link, i) => (
        <button
          key={i}
          onClick={() =>
            navigate("/case-viewer", { state: { url: link.url } })
          }
          className="case-link-button"
        >
          {link.title}
        </button>
      ))}
  </div>
))}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
};

export default App;
