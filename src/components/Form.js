// Form.js
import React, { useState } from "react";
import { Form, Container, Row, Col } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";

function FormComponent({ onSubmit, onClose }) {
  const [formData, setFormData] = useState({
    name: "",
    age: "",
    gender: "",
    height: "",
    weight: "",
    activityLevel: "",
    goal: "",
    dietType: "",
    allergies: "",
    stressLevel: "",
    disease: "",
    bmi: "",
    occupation: "",
    maritalStatus: "",
    sleepHours: "",
    sleepQuality: "",
    wakeUpTime: "",
    bedTime: "",
    exerciseRoutine: "",
    screenTime: "",
  });

  const [showBmiForm, setShowBmiForm] = useState(false);
  const [showStressForm, setShowStressForm] = useState(false);
  const [bmiAdded, setBmiAdded] = useState(false);
  const [stressAdded, setStressAdded] = useState(false);
  const [isBmiLoading, setIsBmiLoading] = useState(false);
  const [isStressLoading, setIsStressLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const handleBmiSubmit = async (e) => {
    e.preventDefault();
    const bmiData = {
      age: formData.age,
      gender: formData.gender,
      height: formData.height,
      weight: formData.weight,
    };
    console.log("BMI Inputs:", bmiData);

    setIsBmiLoading(true);

    try {
      const x=formData.height/100;
      const y=formData.weight;
      const bmiValue = (y/(x*x)).toFixed(2);

      setFormData((prev) => ({ ...prev, bmi: bmiValue }));
      setBmiAdded(true);
      setShowBmiForm(false);
    } catch (error) {
      console.error("Error fetching BMI from Grok API:", error);
      setFormData((prev) => ({ ...prev, bmi: "Error" }));
      setBmiAdded(true);
      setShowBmiForm(false);
    } finally {
      setIsBmiLoading(false);
    }
  };

  const handleStressSubmit = async (e) => {
    e.preventDefault();
    const stressData = {
      age: formData.age,
      gender: formData.gender,
      occupation: formData.occupation || "",
      maritalStatus: formData.maritalStatus || "",
      sleepDuration: formData.sleepHours || "",
      sleepQuality: formData.sleepQuality || "",
      wakeUpTime: formData.wakeUpTime || "",
      bedTime: formData.bedTime || "",
      physicalActivity: formData.exerciseRoutine || "",
      screenTime: formData.screenTime || "",
    };
    console.log("Stress Inputs:", stressData);

    setIsStressLoading(true);

    try {
      const payload = {
        model: "llama-3.3-70b-versatile",
        messages: [
          {
            role: "user",
            content: JSON.stringify({
              instruction: "Assess the stress level based on the following user data and return the stress level as a string: 'Low', 'Medium', or 'High'. Do not include any additional text or explanation.",
              data: stressData,
            }),
          },
        ],
      };

      const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.REACT_APP_GROQ_API_KEY}`,
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      const stressLevel = data.choices[0].message.content;

      setFormData((prev) => ({ ...prev, stressLevel }));
      setStressAdded(true);
      setShowStressForm(false);
    } catch (error) {
      console.error("Error fetching Stress Level from Grok API:", error);
      setFormData((prev) => ({ ...prev, stressLevel: "Error" }));
      setStressAdded(true);
      setShowStressForm(false);
    } finally {
      setIsStressLoading(false);
    }
  };

  const handleBmiButtonClick = (e) => {
    e.stopPropagation();
    console.log("BMI button clicked, setting showBmiForm to true");
    setShowBmiForm(true);
  };

  const handleStressButtonClick = (e) => {
    e.stopPropagation();
    console.log("Stress button clicked, setting showStressForm to true");
    setShowStressForm(true);
  };

  if (showBmiForm) {
    console.log("Rendering BMI form");
  }
  if (showStressForm) {
    console.log("Rendering Stress form");
  }

  // Compute BMI category
  const getBmiCategory = (bmi) => {
    const bmiValue = parseFloat(bmi);
    if (isNaN(bmiValue) || bmi === "Error") return "";
    if (bmiValue < 18.5) return "Underweight";
    if (bmiValue >= 18.5 && bmiValue <= 24.9) return "Normal"; // Adjusted to include 24.9
    if (bmiValue > 24.9 && bmiValue < 29.9) return "Overweight";
    return "Obese";
  };

  const bmiCategory = getBmiCategory(formData.bmi);

  return (
    <>
      {(showBmiForm || showStressForm || isBmiLoading || isStressLoading) && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100vw",
            height: "100vh",
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            zIndex: 1000,
          }}
        />
      )}

      <div
        className="position-fixed top-50 start-50 translate-middle"
        style={{
          zIndex: 999,
          width: "50%",
          borderRadius: "20px",
          padding: "40px",
          boxSizing: "border-box",
          background: "#ecf0f3",
          boxShadow: "5px solid black",
          fontFamily: "Poppins, sans-serif",
          pointerEvents: "auto",
          maxHeight: "80vh",
          overflowY: "auto",
          position: "relative",
        }}
      >
        <Container>
          <Form onSubmit={handleSubmit}>
            <h5 style={{ fontWeight: 900, color: "#90EE90", display: "flex", justifyContent: "center" }}>
              Basic Information
            </h5>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Name</Form.Label>
                <Form.Control
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    caretColor: "red",
                  }}
                  placeholder="Enter name"
                />
              </Form.Group>
            </Row>

            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Age</Form.Label>
                <Form.Control
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    caretColor: "red",
                  }}
                  placeholder="Enter age"
                />
              </Form.Group>
            </Row>

            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Gender</Form.Label>
                <Form.Select
                  name="gender"
                  value={formData.gender}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    cursor: "pointer",
                  }}
                >
                  <option value="">Select Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </Form.Select>
              </Form.Group>
            </Row>

            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Height (cm)</Form.Label>
                <Form.Control
                  type="number"
                  name="height"
                  value={formData.height}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    caretColor: "red",
                  }}
                  placeholder="Enter height"
                />
              </Form.Group>
            </Row>

            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Weight (kg)</Form.Label>
                <Form.Control
                  type="number"
                  name="weight"
                  value={formData.weight}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    caretColor: "red",
                  }}
                  placeholder="Enter weight"
                />
              </Form.Group>
            </Row>

            <h5 style={{ fontWeight: 900, color: "#90EE90" }}>Activity Level</h5>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Activity Level</Form.Label>
                <Form.Select
                  name="activityLevel"
                  value={formData.activityLevel}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    cursor: "pointer",
                  }}
                >
                  <option value="">Select Activity Level</option>
                  <option value="Sedentary">Sedentary (0-1 days active)</option>
                  <option value="Lightly Active">Lightly Active (2-3 days active)</option>
                  <option value="Moderately Active">Moderately Active (4-5 days active)</option>
                  <option value="Very Active">Very Active (6-7 days active)</option>
                </Form.Select>
              </Form.Group>
            </Row>

            <h5 style={{ fontWeight: 900, color: "#90EE90" }}>Health Goals</h5>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Goal</Form.Label>
                <Form.Select
                  name="goal"
                  value={formData.goal}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    cursor: "pointer",
                  }}
                >
                  <option value="">Select Goal</option>
                  <option value="Lose Weight">Lose Weight</option>
                  <option value="Maintain Weight">Maintain Weight</option>
                  <option value="Gain Muscle">Gain Muscle</option>
                </Form.Select>
              </Form.Group>
            </Row>

            <h5 style={{ fontWeight: 900, color: "#90EE90" }}>Dietary Preferences</h5>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Preferred Diet Type</Form.Label>
                <Form.Select
                  name="dietType"
                  value={formData.dietType}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    cursor: "pointer",
                  }}
                >
                  <option value="">Select Diet Type</option>
                  <option value="Vegetarian">Vegetarian</option>
                  <option value="Vegan">Vegan</option>
                  <option value="Keto">Keto</option>
                  <option value="Low Carb">Low Carb</option>
                  <option value="High Protein">High Protein</option>
                  <option value="Balanced">Balanced</option>
                </Form.Select>
              </Form.Group>
            </Row>

            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Allergies</Form.Label>
                <Form.Control
                  type="text"
                  name="allergies"
                  value={formData.allergies}
                  onChange={handleInputChange}
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    caretColor: "red",
                  }}
                  placeholder="e.g., Peanuts, Dairy"
                />
              </Form.Group>
            </Row>

            <Row className="mb-3">
              <Form.Group as={Col}>
                <button
                  type="button"
                  onClick={handleBmiButtonClick}
                  style={{
                    width: "100%",
                    background: "#90EE90",
                    color: "white",
                    padding: "10px",
                    borderRadius: "20px",
                    border: "none",
                    fontWeight: 900,
                    boxShadow: "6px 6px 6px #cbced1, -6px -6px 6px white",
                    transition: "0.5s",
                    cursor: "pointer",
                  }}
                  onMouseEnter={(e) => (e.target.style.boxShadow = "none")}
                  onMouseLeave={(e) =>
                    (e.target.style.boxShadow = "6px 6px 6px #cbced1, -6px -6px 6px white")
                  }
                >
                  Add Your BMI (Health Dataset)
                </button>
                {bmiAdded && (
                  <p style={{ color: "#90EE90", fontWeight: "bold", marginTop: "5px" }}>
                    BMI Added: {formData.bmi} {bmiCategory ? `(${bmiCategory})` : ""}
                  </p>
                )}
              </Form.Group>
            </Row>

            <Row className="mb-3">
              <Form.Group as={Col}>
                <button
                  type="button"
                  onClick={handleStressButtonClick}
                  style={{
                    width: "100%",
                    background: "#90EE90",
                    color: "white",
                    padding: "10px",
                    borderRadius: "20px",
                    border: "none",
                    fontWeight: 900,
                    boxShadow: "6px 6px 6px #cbced1, -6px -6px 6px white",
                    transition: "0.5s",
                    cursor: "pointer",
                  }}
                  onMouseEnter={(e) => (e.target.style.boxShadow = "none")}
                  onMouseLeave={(e) =>
                    (e.target.style.boxShadow = "6px 6px 6px #cbced1, -6px -6px 6px white")
                  }
                >
                  Know Your Stress Level
                </button>
                {stressAdded && (
                  <p style={{ color: "#90EE90", fontWeight: "bold", marginTop: "5px" }}>
                    Stress Level Added: {formData.stressLevel}
                  </p>
                )}
              </Form.Group>
            </Row>

            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label style={{ marginBottom: "4px" }}>Disease (optional)</Form.Label>
                <Form.Control
                  type="text"
                  name="disease"
                  value={formData.disease}
                  onChange={handleInputChange}
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    paddingLeft: "20px",
                    height: "50px",
                    fontSize: "14px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                    outline: "none",
                    caretColor: "red",
                  }}
                  placeholder="e.g., Diabetes"
                />
              </Form.Group>
            </Row>

            <div className="d-flex justify-content-center gap-3">
              <button
                onClick={onClose}
                style={{
                  color: "black",
                  marginTop: "20px",
                  background: "grey",
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
                Cancel
              </button>
              <button
                type="submit"
                style={{
                  color: "white",
                  marginTop: "20px",
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
                Submit
              </button>
            </div>
          </Form>
        </Container>
      </div>

      {showBmiForm && (
        <div
          style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: "40%",
            borderRadius: "20px",
            padding: "20px",
            background: "#ecf0f3",
            boxShadow: "5px solid black",
            zIndex: 1001,
            maxHeight: "60vh",
            overflowY: "auto",
            display: "block !important",
          }}
        >
          <Form onSubmit={handleBmiSubmit}>
            <h5 style={{ fontWeight: 900, color: "#90EE90", textAlign: "center" }}>
              Calculate BMI
            </h5>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Age</Form.Label>
                <Form.Control
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Gender</Form.Label>
                <Form.Select
                  name="gender"
                  value={formData.gender}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                >
                  <option value="">Select Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </Form.Select>
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Height (cm)</Form.Label>
                <Form.Control
                  type="number"
                  name="height"
                  value={formData.height}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Weight (kg)</Form.Label>
                <Form.Control
                  type="number"
                  name="weight"
                  value={formData.weight}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <div className="d-flex justify-content-center gap-3">
              <button
                type="submit"
                style={{
                  background: "#90EE90",
                  color: "white",
                  padding: "10px 20px",
                  borderRadius: "20px",
                  border: "none",
                  fontWeight: 900,
                }}
              >
                Calculate
              </button>
              <button
                type="button"
                onClick={() => {
                  console.log("Closing BMI form");
                  setShowBmiForm(false);
                }}
                style={{
                  background: "grey",
                  color: "black",
                  padding: "10px 20px",
                  borderRadius: "20px",
                  border: "none",
                  fontWeight: 900,
                }}
              >
                Close
              </button>
            </div>
          </Form>
        </div>
      )}

      {showStressForm && (
        <div
          style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: "40%",
            borderRadius: "20px",
            padding: "20px",
            background: "#ecf0f3",
            boxShadow: "5px solid black",
            zIndex: 1001,
            maxHeight: "60vh",
            overflowY: "auto",
            display: "block !important",
          }}
        >
          <Form onSubmit={handleStressSubmit}>
            <h5 style={{ fontWeight: 900, color: "#90EE90", textAlign: "center" }}>
              Stress Level Assessment
            </h5>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Age</Form.Label>
                <Form.Control
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Gender</Form.Label>
                <Form.Select
                  name="gender"
                  value={formData.gender}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                >
                  <option value="">Select Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </Form.Select>
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Occupation</Form.Label>
                <Form.Control
                  type="text"
                  name="occupation"
                  value={formData.occupation || ""}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Marital Status</Form.Label>
                <Form.Select
                  name="maritalStatus"
                  value={formData.maritalStatus || ""}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                >
                  <option value="">Select Marital Status</option>
                  <option value="Single">Single</option>
                  <option value="Married">Married</option>
                  <option value="Divorced">Divorced</option>
                </Form.Select>
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Sleep Duration (hours)</Form.Label>
                <Form.Control
                  type="number"
                  name="sleepHours"
                  value={formData.sleepHours || ""}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Sleep Quality</Form.Label>
                <Form.Select
                  name="sleepQuality"
                  value={formData.sleepQuality || ""}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                >
                  <option value="">Select Sleep Quality</option>
                  <option value="Poor">Poor</option>
                  <option value="Average">Average</option>
                  <option value="Good">Good</option>
                </Form.Select>
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Wake Up Time</Form.Label>
                <Form.Control
                  type="time"
                  name="wakeUpTime"
                  value={formData.wakeUpTime || ""}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Bed Time</Form.Label>
                <Form.Control
                  type="time"
                  name="bedTime"
                  value={formData.bedTime || ""}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Physical Activity ( Yes/No )</Form.Label>
                <Form.Control
                  type="text"
                  name="exerciseRoutine"
                  value={formData.exerciseRoutine || ""}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <Row className="mb-3">
              <Form.Group as={Col}>
                <Form.Label>Screen Time (hours)</Form.Label>
                <Form.Control
                  type="number"
                  name="screenTime"
                  value={formData.screenTime || ""}
                  onChange={handleInputChange}
                  required
                  style={{
                    background: "#ecf0f3",
                    padding: "10px",
                    borderRadius: "50px",
                    boxShadow: "inset 6px 6px 6px #cbced1, inset -6px -6px 6px white",
                    border: "none",
                  }}
                />
              </Form.Group>
            </Row>
            <div className="d-flex justify-content-center gap-3">
              <button
                type="submit"
                style={{
                  background: "#90EE90",
                  color: "white",
                  padding: "10px 20px",
                  borderRadius: "20px",
                  border: "none",
                  fontWeight: 900,
                }}
              >
                Calculate
              </button>
              <button
                type="button"
                onClick={() => {
                  console.log("Closing Stress form");
                  setShowStressForm(false);
                }}
                style={{
                  background: "grey",
                  color: "black",
                  padding: "10px 20px",
                  borderRadius: "20px",
                  border: "none",
                  fontWeight: 900,
                }}
              >
                Close
              </button>
            </div>
          </Form>
        </div>
      )}

      {isBmiLoading && (
        <div
          style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: "40%",
            borderRadius: "20px",
            padding: "30px",
            background: "rgba(236, 240, 243, 0.95)",
            boxShadow: "0 8px 16px rgba(0, 0, 0, 0.2)",
            zIndex: 1002,
            textAlign: "center",
            fontFamily: "Poppins, sans-serif",
          }}
        >
          <h5 style={{ fontWeight: 900, color: "#90EE90", marginBottom: "20px" }}>
            Fetching BMI from model...
          </h5>
          <div className="pulsing-dots">
            <span></span>
            <span></span>
            <span></span>
            <span></span>
          </div>
          <style>
            {`
              .pulsing-dots {
                display: flex;
                justify-content: center;
                gap: 10px;
              }
              .pulsing-dots span {
                width: 12px;
                height: 12px;
                background: #90EE90;
                border-radius: 50%;
                display: inline-block;
                animation: pulse 1.2s ease-in-out infinite;
              }
              .pulsing-dots span:nth-child(1) {
                animation-delay: 0s;
              }
              .pulsing-dots span:nth-child(2) {
                animation-delay: 0.3s;
              }
              .pulsing-dots span:nth-child(3) {
                animation-delay: 0.6s;
              }
              .pulsing-dots span:nth-child(4) {
                animation-delay: 0.9s;
              }
              @keyframes pulse {
                0%, 100% {
                  transform: scale(1);
                  opacity: 0.5;
                }
                50% {
                  transform: scale(1.5);
                  opacity: 1;
                }
              }
            `}
          </style>
        </div>
      )}

      {isStressLoading && (
        <div
          style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: "40%",
            borderRadius: "20px",
            padding: "30px",
            background: "rgba(236, 240, 243, 0.95)",
            boxShadow: "0 8px 16px rgba(0, 0, 0, 0.2)",
            zIndex: 1002,
            textAlign: "center",
            fontFamily: "Poppins, sans-serif",
          }}
        >
          <h5 style={{ fontWeight: 900, color: "#90EE90", marginBottom: "20px" }}>
            Fetching Stress Level from model...
          </h5>
          <div className="pulsing-dots">
            <span></span>
            <span></span>
            <span></span>
            <span></span>
          </div>
          <style>
            {`
              .pulsing-dots {
                display: flex;
                justify-content: center;
                gap: 10px;
              }
              .pulsing-dots span {
                width: 12px;
                height: 12px;
                background: #90EE90;
                border-radius: 50%;
                display: inline-block;
                animation: pulse 1.2s ease-in-out infinite;
              }
              .pulsing-dots span:nth-child(1) {
                animation-delay: 0s;
              }
              .pulsing-dots span:nth-child(2) {
                animation-delay: 0.3s;
              }
              .pulsing-dots span:nth-child(3) {
                animation-delay: 0.6s;
              }
              .pulsing-dots span:nth-child(4) {
                animation-delay: 0.9s;
              }
              @keyframes pulse {
                0%, 100% {
                  transform: scale(1);
                  opacity: 0.5;
                }
                50% {
                  transform: scale(1.5);
                  opacity: 1;
                }
              }
            `}
          </style>
        </div>
      )}
    </>
  );
}

export default FormComponent;