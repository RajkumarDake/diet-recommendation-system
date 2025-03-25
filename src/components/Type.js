// Type.js
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import FormComponent from "./Form";
import Display from "./Display";

function Type() {
  const navigate = useNavigate();
  const [showForm, setShowForm] = useState(false);
  const [showDisplay, setShowDisplay] = useState(false);
  const [loading, setLoading] = useState(false);
  const [dietCharts, setDietCharts] = useState({ india: null, us: null, foodAvoidance: [] });

  const handleFormSubmit = async (formData) => {
    setLoading(true);

    const fetchDietChart = async (region) => {
      const payload = {
        model: "llama-3.3-70b-versatile", 
        messages: [
          {
            role: "user",
            content: JSON.stringify({
              instruction: `Generate a ${region} diet chart for a user with the following details: Name: ${formData.name}, Age: ${formData.age}, Gender: ${formData.gender}, BMI: ${formData.bmi || "Not provided"}, Stress Level: ${formData.stressLevel || "Not provided"}, Goal: ${formData.goal}, Diet Type: ${formData.dietType}, Allergies: ${formData.allergies || "None"}, Activity Level: ${formData.activityLevel}, Disease: ${formData.disease || "None"}. Provide the diet chart in a structured format with the following sections: ## Introduction - A personalized message based on the form data, explaining how the diet supports the user's goal. ## Breakfast - List of food items according to the user's diet and their macros (**in bold**). **Total macros of Breakfast:** **{Total_Breakfast_Macros}** ## Lunch - List of food items according to the user's diet and their macros (**in bold**). **Total macros of Lunch:** **{Total_Lunch_Macros}** ## Dinner - List of food items according to the user's diet and their macros (**in bold**). **Total macros of Dinner:** **{Total_Dinner_Macros}** ## Total Macros of the Day **{Total_Day_Macros}** Ensure that macros are highlighted after each meal and the content is concise, clear, and aligned with the user's dietary needs.`,
              data: formData,
            }),
          },
        ],
      };

      console.log(`Sending request to Grok API for ${region} diet chart:`, payload);

      try {
        const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${process.env.REACT_APP_GROQ_API_KEY}`,
          },
          body: JSON.stringify(payload),
        });

        console.log(`Response status for ${region} diet chart:`, response.status);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const data = await response.json();
        console.log(`Response data for ${region} diet chart:`, data);

        if (!data.choices || !data.choices[0] || !data.choices[0].message || !data.choices[0].message.content) {
          throw new Error(`Invalid response structure: ${JSON.stringify(data)}`);
        }

        return data.choices[0].message.content;
      } catch (error) {
        console.error(`Error fetching ${region} diet chart:`, error.message);
        throw error; // Re-throw to handle in the main try-catch
      }
    };



    // Fetching BMI data from the Model
   const fetchBMIData = async (age,gender, height, weight) => {
  try {
    const response = await fetch("http://localhost:5000/bmi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gender, height, weight }),
    });

    if (!response.ok) {
      console.error(`Error fetching BMI data: HTTP status ${response.status}`);
      return { bmi: null, recommendation: "No data available" };
    }

    const result = await response.json();
    console.log("BMI Response (for demo):", result); // Log to show it’s for presentation
    return result; // Model should process gender, height, weight, but currently just returns response
  } catch (error) {
    console.error("Error fetching BMI data:", error);
    return { bmi: null, recommendation: "No data available" };
  }
};

// Fetching Stress data from the Model
const fetchStressData = async (gender, height, weight, wakeupTime, sleepTime, maritalStatus) => {
  try {
    const response = await fetch("http://localhost:5000/stress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gender, height, weight, wakeupTime, sleepTime, maritalStatus }),
    });

    if (!response.ok) {
      console.error(`Error fetching stress data: HTTP status ${response.status}`);
      return { stressImpact: "Unknown", suggestion: "No data" };
    }

    const result = await response.json();
    console.log("Stress Response (for demo):", result); // Log to show it’s for presentation
    return result; // Model should process all inputs, but currently just returns response
  } catch (error) {
    console.error("Error fetching stress data:", error);
    return { stressImpact: "Unknown", suggestion: "No data" };
  }
};

// Fetching Food Avoidance data from the Model
const fetchFoodAvoidance = async (text) => {
  try {
    const response = await fetch("http://localhost:5000/food", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      console.error(`Error fetching food avoidance: HTTP status ${response.status}`);
      return [{ food_entity: "None", sentence: "No data" }];
    }

    const result = await response.json();
    console.log("Food Avoidance Response (for demo):", result); // Log to show it’s for presentation
    return result; // Model should process text (disease), but currently just returns response
  } catch (error) {
    console.error("Error fetching food avoidance:", error);
    return [{ food_entity: "None", sentence: "No data" }];
  }
};






    try {
      const [indiaChart, usChart, foodAvoidance] = await Promise.all([
        fetchDietChart("Indian").catch((error) => {
          console.error("Failed to fetch Indian diet chart:", error);
          return "Failed to generate Indian diet chart";
        }),
        fetchDietChart("US").catch((error) => {
          console.error("Failed to fetch US diet chart:", error);
          return "Failed to generate US diet chart";
        }),
        formData.disease ? fetchFoodAvoidance(formData.disease) : Promise.resolve([]),
      ]);

      setDietCharts({ india: indiaChart, us: usChart, foodAvoidance });
      setShowDisplay(true);
    } catch (error) {
      console.error("Unexpected error in handleFormSubmit:", error);
      setDietCharts({
        india: "Failed to generate Indian diet chart",
        us: "Failed to generate US diet chart",
        foodAvoidance: [],
      });
      setShowDisplay(true);
    } finally {
      setLoading(false);
      setShowForm(false);
    }
  };

  const handleButton2Click = () => {
    setShowForm(true);
    console.log("Button 2 clicked, showForm:", true);
  };

  const handleTipsClick = () => {
    navigate("/tips");
    console.log("Tips button clicked");
  };

  return (
    <div style={{ padding: "20px", position: "relative" }}>
      <div
        style={{
          width: "100%",
          display: "flex",
          flexDirection: "row",
          justifyContent: "center",
          alignItems: "center",
          marginTop: "-2rem",
          marginLeft: "8em",
        }}
      >
        <button
          style={{
            padding: "20px 40px",
            backgroundColor: "white",
            color: "lightgreen",
            height: "60px",
            borderRadius: "20px",
            fontSize: "16px",
            fontWeight: 900,
            cursor: "pointer",
            boxShadow: "-6px -6px 6px black",
            transition: "0.5s",
            border: "none",
            outline: "none",
          }}
          onClick={handleTipsClick}
          onMouseEnter={(e) => (e.target.style.boxShadow = "none")}
          onMouseLeave={(e) =>
            (e.target.style.boxShadow = "-6px -6px 6px black")
          }
        >
          Tips
        </button>
        <button
          style={{
            padding: "20px 40px",
            backgroundColor: "green",
            color: "white",
            height: "60px",
            borderRadius: "20px",
            fontSize: "16px",
            fontWeight: 900,
            cursor: "pointer",
            boxShadow: "-6px -6px 6px black",
            transition: "0.5s",
            border: "none",
            outline: "none",
            marginLeft: "10px",
          }}
          onClick={handleButton2Click}
          onMouseEnter={(e) => (e.target.style.boxShadow = "none")}
          onMouseLeave={(e) =>
            (e.target.style.boxShadow = "-6px -6px 6px black")
          }
        >
          Show Form
        </button>
      </div>

      {showForm && !loading && (
        <>
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              backgroundColor: "rgba(0, 0, 0, 0.5)",
              zIndex: 998,
            }}
            onClick={() => setShowForm(false)}
          />
          <FormComponent
            onSubmit={handleFormSubmit}
            onClose={() => setShowForm(false)}
          />
        </>
      )}

      {showDisplay && !loading && (
        <>
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              backgroundColor: "rgba(0, 0, 0, 0.5)",
              zIndex: 998,
            }}
            onClick={() => setShowDisplay(false)}
          />
          <Display
            dietCharts={dietCharts}
            onClose={() => setShowDisplay(false)}
          />
        </>
      )}

      {loading && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            zIndex: 1001,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div className="loader" />
          <p
            style={{
              marginTop: "20px",
              fontSize: "18px",
              fontWeight: "bold",
              color: "#25b09b",
              fontFamily: "Poppins, sans-serif",
            }}
          >
            Generating Diet Chart from model...
          </p>
        </div>
      )}

      <style>
        {`
          .loader {
            width: 50px;
            aspect-ratio: 1.154;
            position: relative,
            background: conic-gradient(from 120deg at 50% 64%, #0000, #25b09b 1deg 120deg, #0000 121deg);
            animation: l27-0 1.5s infinite cubic-bezier(0.3,1,0,1);
          }
          .loader:before,
          .loader:after {
            content: '';
            position: absolute;
            inset: 0;
            background: inherit;
            transform-origin: 50% 66%;
            animation: l27-1 1.5s infinite;
          }
          .loader:after {
            --s: -1;
          }
          @keyframes l27-0 {
            0%, 30%      { transform: rotate(0) }
            70%         { transform: rotate(120deg) }
            70.01%, 100% { transform: rotate(360deg) }
          }
          @keyframes l27-1 {
            0%      { transform: rotate(calc(var(--s,1)*120deg)) translate(0) }
            30%, 70% { transform: rotate(calc(var(--s,1)*120deg)) translate(calc(var(--s,1)*-5px), 10px) }
            100%    { transform: rotate(calc(var(--s,1)*120deg)) translate(0) }
          }
        `}
      </style>
    </div>
  );
}

export default Type;