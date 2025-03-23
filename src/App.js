import React from "react";
import { Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Tips from "./pages/Tips";
import Header from "./components/Header";
import Footer from "./components/Footer";
import ScrollToTop from "./components/ScrollToTop";

function App() {
  return (
    <div
      style={{
        backgroundImage: "linear-gradient(45deg, #043927, #c4b581)",
      }}
      className="h-100 d-flex flex-wrap align-items-center justify-content-center flex-column flex-lg-row"
    >
      <Header />
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/tips" element={<Tips />} /> 
      </Routes>
      <Footer />
    </div>
  );
}

export default App;