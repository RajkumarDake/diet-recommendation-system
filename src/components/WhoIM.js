import React from "react";
import { FaHandPointRight } from "react-icons/fa";
function WhoIM() {
  return (
    <div className="flex flex-wrap items-center justify-center flex-col lg:flex-row relative">
      <img
        src="./star.jpg"
        alt=""
        className="h-full w-full object-cover opacity-30 absolute bottom-0 z-0"
      />
      <div className="w-full lg:w-1/2 h-1/2 lg:h-full flex items-center justify-center flex-col my-10 lg:gap-10 z-20">
        <h1 className="flex text-white text-2xl lg:text-4xl gap-2 font-bold">
          Know Who <p className="text-[#ad50eb]">I'M</p>
        </h1>
        <div className="flex items-center m-auto justify-center w-11/12 lg:w-4/5">
          <span className="flex flex-col gap-3 text-white items-start  text-base justify-center font-thin lg:text-lg">
            <p className="text-start mt-3 mx-auto md:mx-0">
            I'm Jahnavi, a Frontend Web Developer from New Rajahmundry with
              expertise in React, Node.js, and Express.js. With over
              1.5 years of experience, I've built high-performing web
              applications, including blogging platforms and e-commerce
              solutions. With a passion for learning and attention to detail,
               I'm committed to delivering impactful digital experiences. 
               Let's connect and create together!
            </p>

            <p className="flex text-gray-200 text-base lg:text-lg gap-2 font-semibold ">
              Feel free to explore my portfolio for a deeper insight into my
              projects and skills. Let's connect and create something amazing
              together!
            </p>

            <span className="flex items-center justify-center gap-3 font-sans text-gray-200">
              <FaHandPointRight /> Web Development
            </span>
          </span>
        </div>
      </div>
      <div className="w-full lg:w-1/2 h-full flex items-center justify-center z-20">
        <img
          src="about.png"
          alt=""
          className="lg:p-10 object-cover lg:w-3/4 w-full"
        />
      </div>
    </div>
  );
}

export default WhoIM;
