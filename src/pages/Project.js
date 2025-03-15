import React from "react";
import ProjectCard from "../components/ProjectCard";

function Project() {
  return (
    <div className="flex flex-wrap items-center justify-center flex-col lg:flex-row relative overflow-hidden ">
      <img
        src="./star.jpg"
        alt=""
        className="h-full w-full object-cover opacity-20 absolute bottom-0"
      />
      <div className="z-20 flex items-center justify-center gap-3 m-2 flex-wrap">
        <ProjectCard
          name={"Luxora E-commerce Website"}
          image={"/project/img1.png"}
          about={
            "You can shop your favorite cloths using Luxora And it is also mobile responsive"
          }
          demo={"https://reactjs-ecommerce-app.vercel.app/"}
          code={"https://github.com/Gummadijahnavi/Luxora"}
        />
        <ProjectCard
          name={"Code Box"}
          image={"/project/img2.png"}
          about={
            "meet Code Box an online frontend ide where we can write html, css , Java Script in one page."
          }
          demo={"https://gummadijahnavi.github.io/Codebox/"}
          code={"https://github.com/Gummadijahnavi/Codebox"}
        />
      </div>
    </div>
  );
}

export default Project;
