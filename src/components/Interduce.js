import React from "react";
import Tilt from "react-parallax-tilt";

function Introduce() { // Renamed to "Introduce" for clarity
  return (
    <div className="flex flex-wrap items-center justify-center flex-col lg:flex-row relative overflow-hidden">
      <img
        src="./home-bg.jpg"
        alt="Background"
        className="h-full w-full object-cover opacity-10 absolute bottom-0"
      />
      <div className="w-full lg:w-1/2 h-1/2 lg:h-full flex items-center justify-center flex-col my-10 lg:gap-10">
        <h1 className="flex text-white text-2xl lg:text-4xl gap-2 my-10 font-bold">
          About <p className="text-[#8b346d]">Our Diet Recommendation</p> 
        </h1>
        <div className="flex items-center m-auto justify-center w-11/12 lg:w-3/4 lg:mr-10">
          <span
            className="flex flex-col gap-4 text-white items-start text-base justify-center font-semibold lg:text-lg"
            style={{ fontFamily: "Poppins, sans-serif" }}
          >
            <p>
              Welcome to our Diet Chart Recommendation platform, designed to help you achieve your health and wellness goals with personalized nutrition plans. Whether you’re looking to lose weight, maintain a balanced lifestyle, or gain muscle, we’ve got you covered.
              Our smart system takes your unique details—like age, weight, activity level, and dietary preferences—and generates a tailored diet chart just for you. Powered by advanced technology, we ensure every recommendation is practical, sustainable, and aligned with your needs.
            </p>
          </span>
        </div>
      </div>
      <div className="w-full lg:w-1/2 h-full flex items-center justify-center p-10">
        <Tilt>
          <img
            src="logo2.png"
            alt="Diet Recommendation Logo"
            className="object-cover"
            width={500}
            height={500}
          />
        </Tilt>
      </div>
    </div>
  );
}

export default Introduce;