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
            Hi There!
            <span>
              <img
                src="https://media.tenor.com/SNL9_xhZl9oAAAAi/waving-hand-joypixels.gif"
                alt="hand"
                className="h-10"
              />
            </span>
          </h3>
          
          <span className="flex text-white lg:text-xl mx-20  gap-2 font-bold justify-center items-center">

          Good health starts with a balanced diet and an active lifestyle. Eat plenty of fruits, vegetables, lean proteins, and whole grains. Stay hydrated, exercise regularly, and listen to your body. Ready for personalized diet tips? Fill out the form and get started on your health journey today!
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
