import React from "react";
import Type from "./Type";

function Hero() {
  return (
    <div style={{
      backgroundImage: "linear-gradient(45deg, #008453,#c4b581)",
    }} 
    className=" h-2/3 flex flex-wrap items-center justify-center flex-col lg:flex-row filter: blur(px);">
      <div className="w-full lg:w-1/2 h-1/2 lg:h-full flex items-center justify-center flex-col mt-10">
        <div className=" flex gap-5 items-center lg:items-start justify-center flex-col">
          <h3 className="flex text-white text-4xl gap-2 mx-20 font-bold">
            AI-Powered Nutrition
            <span>
              <img
                src="https://media.tenor.com/SNL9_xhZl9oAAAAi/waving-hand-joypixels.gif"
                alt="hand"
                className="h-10"
              />
            </span>
          </h3>
          
          <span className="flex text-white lg:text-xl mx-20  gap-2 font-bold justify-center items-center">
          Experience the future of personalized nutrition with our advanced AI system. Using LSTM neural networks for health analysis, Transformer models for genomic insights, and fusion algorithms for comprehensive recommendations. Get personalized nutrition plans based on your genetics, health metrics, and mental wellness data.
          </span>
        <span>
            <Type />
          </span>
        </div>
      </div>
      <div className="w-full lg:w-1/2 h-full flex items-center justify-center">
        <img src="home-main.svg" alt="" className="h-[70vh]" />
      </div>
    </div>
  );
}

export default Hero;
