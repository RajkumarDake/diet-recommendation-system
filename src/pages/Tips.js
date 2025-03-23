// Tips.js
import React from "react";

function Tips() {
  return (
    <div
      style={{
        padding: "40px",
        textAlign: "center",
        fontFamily: "Poppins, sans-serif",
        maxWidth: "800px",
        margin: "0 auto",
      }}
    >
      <h2
        style={{
          fontSize: "28px",
          fontWeight: 900,
          color: "white",
          marginBottom: "30px",
        }}
      >
        Diet & Wellness Tips
      </h2>

      <section style={{ textAlign: "left", marginBottom: "40px" }}>
        <h3
          style={{
            fontSize: "22px",
            fontWeight: 700,
            color: "#333",
            marginBottom: "15px",
          }}
        >
          1. Stay Hydrated
        </h3>
        <p style={{ fontSize: "16px", color: "#666", lineHeight: "1.6" }}>
          Water is essential for overall health and aids in digestion, metabolism, and weight management.
        </p>
        <ul
          style={{
            listStyleType: "disc",
            paddingLeft: "20px",
            fontSize: "16px",
            color: "#666",
          }}
        >
          <li>Drink at least 8-10 glasses (2-2.5 liters) of water daily.</li>
          <li>Add lemon, cucumber, or mint for flavor if plain water feels boring.</li>
          <li>Carry a reusable water bottle to stay consistent throughout the day.</li>
        </ul>
      </section>

      <section style={{ textAlign: "left", marginBottom: "40px" }}>
        <h3
          style={{
            fontSize: "22px",
            fontWeight: 700,
            color: "#333",
            marginBottom: "15px",
          }}
        >
          2. Eat Balanced Meals
        </h3>
        <p style={{ fontSize: "16px", color: "#666", lineHeight: "1.6" }}>
          A balanced diet provides the nutrients your body needs to function optimally.
        </p>
        <ul
          style={{
            listStyleType: "disc",
            paddingLeft: "20px",
            fontSize: "16px",
            color: "#666",
          }}
        >
          <li>Include a mix of protein (e.g., chicken, beans), carbs (e.g., whole grains), and healthy fats (e.g., avocado).</li>
          <li>Fill half your plate with vegetables for fiber and vitamins.</li>
          <li>Avoid processed foods high in sugar and unhealthy fats.</li>
          <li>Eat smaller, frequent meals to maintain energy levels.</li>
        </ul>
      </section>

      <section style={{ textAlign: "left", marginBottom: "40px" }}>
        <h3
          style={{
            fontSize: "22px",
            fontWeight: 700,
            color: "#333",
            marginBottom: "15px",
          }}
        >
          3. Exercise Regularly
        </h3>
        <p style={{ fontSize: "16px", color: "#666", lineHeight: "1.6" }}>
          Physical activity boosts metabolism, improves mood, and supports weight goals.
        </p>
        <ul
          style={{
            listStyleType: "disc",
            paddingLeft: "20px",
            fontSize: "16px",
            color: "#666",
          }}
        >
          <li>Aim for at least 30 minutes of moderate exercise (e.g., walking, cycling) 5 days a week.</li>
          <li>Incorporate strength training (e.g., weights, bodyweight exercises) 2-3 times weekly.</li>
          <li>Stay active throughout the day—take stairs, stretch during breaks.</li>
        </ul>
      </section>

      <section style={{ textAlign: "left", marginBottom: "40px" }}>
        <h3
          style={{
            fontSize: "22px",
            fontWeight: 700,
            color: "#333",
            marginBottom: "15px",
          }}
        >
          4. Prioritize Sleep
        </h3>
        <p style={{ fontSize: "16px", color: "#666", lineHeight: "1.6" }}>
          Good sleep regulates hunger hormones and supports recovery.
        </p>
        <ul
          style={{
            listStyleType: "disc",
            paddingLeft: "20px",
            fontSize: "16px",
            color: "#666",
          }}
        >
          <li>Aim for 7-9 hours of quality sleep per night.</li>
          <li>Establish a bedtime routine—avoid screens 1 hour before sleep.</li>
          <li>Keep your sleep environment dark, quiet, and cool.</li>
        </ul>
      </section>

      <section style={{ textAlign: "left" }}>
        <h3
          style={{
            fontSize: "22px",
            fontWeight: 700,
            color: "#333",
            marginBottom: "15px",
          }}
        >
          5. Manage Stress
        </h3>
        <p style={{ fontSize: "16px", color: "#666", lineHeight: "1.6" }}>
          Chronic stress can lead to overeating or poor food choices.
        </p>
        <ul
          style={{
            listStyleType: "disc",
            paddingLeft: "20px",
            fontSize: "16px",
            color: "#666",
          }}
        >
          <li>Practice mindfulness or meditation for 10-15 minutes daily.</li>
          <li>Engage in hobbies or activities you enjoy to unwind.</li>
          <li>Talk to a friend or professional if stress feels overwhelming.</li>
        </ul>
      </section>
    </div>
  );
}

export default Tips;