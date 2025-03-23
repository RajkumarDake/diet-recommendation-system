import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import ReactMarkdown from "react-markdown";
import jsPDF from "jspdf"; // Import jsPDF

function Display({ dietCharts, onClose }) {
  const [dietType, setDietType] = useState("india");

  // Function to strip Markdown and format with bullets and spacing
  const formatTextForPDF = (text) => {
    if (!text) return ["No content available"];
    const plainText = text
      .replace(/#{1,6}\s/g, "") // Remove Markdown headings
      .replace(/(\*\*|__)(.*?)\1/g, "$2") // Remove bold
      .replace(/(\*|_)(.*?)\1/g, "$2") // Remove italics
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // Remove links
      .replace(/\n{2,}/g, "\n\n"); // Keep paragraph breaks

    // Split into lines and add bullets where appropriate
    const lines = plainText.split("\n").filter((line) => line.trim());
    return lines.map((line) => {
      // Add bullet points for lines that seem like list items or meal headers
      if (line.match(/^(Breakfast|Lunch|Dinner|Snacks)/i)) {
        return `• ${line}`; // Add bullet for meal headers
      }
      return `  ${line}`; // Indent other lines for spacing
    });
  };

  // Function to generate and download PDF with bullets and proper pagination
  const handleDownload = () => {
    const doc = new jsPDF();
    const pageHeight = 270; // Approximate usable height of an A4 page in points
    let currentY = 20;

    // Set font and styling
    doc.setFont("helvetica", "normal");

    // Helper function to add text with pagination
    const addTextWithPagination = (textLines, x, y, fontSize, color) => {
      doc.setFontSize(fontSize);
      doc.setTextColor(...color);
      textLines.forEach((line) => {
        if (currentY > pageHeight - 10) {
          doc.addPage();
          currentY = 20;
        }
        const wrappedLines = doc.splitTextToSize(line, 170);
        wrappedLines.forEach((wrappedLine) => {
          if (currentY > pageHeight - 10) {
            doc.addPage();
            currentY = 20;
          }
          doc.text(wrappedLine, x, currentY);
          currentY += 5; // Line spacing
        });
        currentY += 2; // Extra spacing after each original line
      });
      return currentY;
    };

    // Add Indian Diet Chart
    doc.setFontSize(16);
    doc.setTextColor(0, 128, 0); // Green heading
    doc.text("Indian Diet Chart", 20, currentY);
    currentY += 10;
    const indianLines = formatTextForPDF(dietCharts.india || "No Indian diet chart available");
    currentY = addTextWithPagination(indianLines, 20, currentY, 12, [0, 0, 0]);

    // Add US Diet Chart
    currentY += 10; // Space before next section
    if (currentY > pageHeight - 20) {
      doc.addPage();
      currentY = 20;
    }
    doc.setFontSize(16);
    doc.setTextColor(0, 128, 0);
    doc.text("US Diet Chart", 20, currentY);
    currentY += 10;
    const usLines = formatTextForPDF(dietCharts.us || "No US diet chart available");
    currentY = addTextWithPagination(usLines, 20, currentY, 12, [0, 0, 0]);

    // Add Foods to Avoid section (if available)
    if (dietCharts.foodAvoidance && dietCharts.foodAvoidance.length > 0) {
      currentY += 10;
      if (currentY > pageHeight - 20) {
        doc.addPage();
        currentY = 20;
      }
      doc.setFontSize(16);
      doc.setTextColor(0, 128, 0);
      doc.text("Foods to Avoid", 20, currentY);
      currentY += 10;

      dietCharts.foodAvoidance.forEach((item) => {
        const avoidanceText = `• ${item.food_entity}: ${item.sentence}`;
        if (currentY > pageHeight - 10) {
          doc.addPage();
          currentY = 20;
        }
        const wrappedLines = doc.splitTextToSize(avoidanceText, 170);
        wrappedLines.forEach((line) => {
          if (currentY > pageHeight - 10) {
            doc.addPage();
            currentY = 20;
          }
          doc.setFontSize(12);
          doc.setTextColor(0, 0, 0);
          doc.text(line, 20, currentY);
          currentY += 5;
        });
        currentY += 2; // Extra spacing after each item
      });
    }

    // Save the PDF
    doc.save("Diet_Charts.pdf");
  };

  return (
    <div
      className="position-fixed top-50 start-50 translate-middle"
      style={{
        zIndex: 999,
        width: "50%",
        borderRadius: "20px",
        padding: "40px",
        boxSizing: "border-box",
        background: "#ecf0f3",
        fontFamily: "Poppins, sans-serif",
        pointerEvents: "auto",
        maxHeight: "80vh",
        overflowY: "auto",
        boxShadow: "5px 5px 15px rgba(0, 0, 0, 0.2)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px",
        }}
      >
        <h5 style={{ fontWeight: 900, color: "#90EE90", margin: 0 }}>
          {dietType === "india" ? "Indian Diet Chart" : "US Diet Chart"}
        </h5>
        <div>
          <button
            style={{
              padding: "5px 15px",
              backgroundColor: dietType === "india" ? "#90EE90" : "#ccc",
              color: dietType === "india" ? "white" : "#333",
              border: "none",
              borderRadius: "10px",
              fontSize: "14px",
              fontWeight: "bold",
              cursor: "pointer",
              marginRight: "10px",
              transition: "background-color 0.3s",
            }}
            onClick={() => setDietType("india")}
          >
            Indian
          </button>
          <button
            style={{
              padding: "5px 15px",
              backgroundColor: dietType === "us" ? "#90EE90" : "#ccc",
              color: dietType === "us" ? "white" : "#333",
              border: "none",
              borderRadius: "10px",
              fontSize: "14px",
              fontWeight: "bold",
              cursor: "pointer",
              transition: "background-color 0.3s",
            }}
            onClick={() => setDietType("us")}
          >
            US
          </button>
        </div>
      </div>
      <div
        style={{
          whiteSpace: "pre-wrap",
          fontSize: "14px",
          textAlign: "left",
          color: "#333",
          background: "#f9f9f9",
          padding: "15px",
          borderRadius: "10px",
          boxShadow: "inset 2px 2px 5px rgba(0, 0, 0, 0.1)",
        }}
      >
        <ReactMarkdown>
          {dietCharts[dietType] || `No ${dietType === "india" ? "Indian" : "US"} diet chart available`}
        </ReactMarkdown>
      </div>

      {dietCharts.foodAvoidance && dietCharts.foodAvoidance.length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h6 style={{ fontWeight: 900, color: "#90EE90", marginBottom: "10px" }}>
            Foods to Avoid
          </h6>
          <ul style={{ listStyleType: "disc", paddingLeft: "20px", color: "#333" }}>
            {dietCharts.foodAvoidance.map((item, index) => (
              <li key={index}>
                <strong>{item.food_entity}</strong>: {item.sentence}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="d-flex justify-content-center mt-4 gap-3">
        <button
          onClick={onClose}
          style={{
            color: "white",
            background: "#90EE90",
            height: "40px",
            borderRadius: "20px",
            cursor: "pointer",
            fontWeight: 900,
            boxShadow: "6px 6px 6px #cbced1, -6px -6px 6px white",
            transition: "0.5s",
            border: "none",
            outline: "none",
            width: "100px",
          }}
          onMouseEnter={(e) => (e.target.style.boxShadow = "none")}
          onMouseLeave={(e) =>
            (e.target.style.boxShadow = "6px 6px 6px #cbced1, -6px -6px 6px white")
          }
        >
          Close
        </button>
        <button
          onClick={handleDownload}
          style={{
            color: "white",
            background: "red",
            height: "40px",
            borderRadius: "20px",
            cursor: "pointer",
            fontWeight: 400,
            boxShadow: "6px 6px 6px #cbced1, -6px -6px 6px white",
            transition: "0.5s",
            border: "none",
            outline: "none",
            width: "100px",
          }}
          onMouseEnter={(e) => (e.target.style.boxShadow = "none")}
          onMouseLeave={(e) =>
            (e.target.style.boxShadow = "6px 6px 6px #cbced1, -6px -6px 6px white")
          }
        >
          Download
        </button>
      </div>
    </div>
  );
}

export default Display;