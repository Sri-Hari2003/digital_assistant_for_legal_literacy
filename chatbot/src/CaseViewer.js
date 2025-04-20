import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./CaseViewer.css";

const CaseViewer = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [htmlContent, setHtmlContent] = useState("");

  useEffect(() => {
    const fetchHtml = async () => {
      try {
        const res = await fetch(location.state?.url);
        const text = await res.text();
        setHtmlContent(text);
      } catch (error) {
        setHtmlContent("<p>Error loading case file.</p>");
      }
    };

    if (location.state?.url) {
      fetchHtml();
    } else {
      setHtmlContent("<p>No case file provided.</p>");
    }
  }, [location.state]);

  return (
    <div className="case-viewer-container">
      <button className="summary-button" onClick={() => navigate("/")}>
        go back to chat
      </button>
      <div
        className="case-content"
        dangerouslySetInnerHTML={{ __html: htmlContent }}
      />
    </div>
  );
};

export default CaseViewer;
