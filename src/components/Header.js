import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AiOutlineFundProjectionScreen } from "react-icons/ai";
import { FiFileText } from "react-icons/fi";
import { RiMenu3Fill } from "react-icons/ri";
import { MdClose } from "react-icons/md";
import MobileNav from "./MobileNav";

function Header() {
  const navigate = useNavigate();
  const [showNav, setShowNav] = useState(false);
  const [scroll, setScrolled] = useState(false);
  const name = ["Diet Chart Recommendation"];

  const handleScroll = () => {
    const offset = window.scrollY;
    if (offset > 50) {
      setScrolled(true);
    } else {
      setScrolled(false);
    }
  };

  useEffect(() => {
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div
      className={
        scroll
          ? "flex items-center justify-between px-4 z-50 bg-purple-950 bg-opacity-20 backdrop-blur-lg rounded drop-shadow-lg shadow-lg relative lg:sticky top-0"
          : "flex items-center justify-between px-4 z-50 relative lg:sticky top-0"
      }
      style={{ width: "100%" }}
    >
      <span className="capitalize md:w-1/3 lg:w-1/2 flex items-center justify-center py-3 px-2 relative">
        <h1
          className="cursor-pointer text-xl lg:text-3xl font-bold bg-white from-gray-50 to-blue-100 bg-clip-text text-transparent uppercase"
          onClick={() => navigate("/")}
        >
          {name}
        </h1>
      </span>
      <div className="w-2/3 hidden lg:flex items-center justify-start text-white">
        <ul className="flex gap-8 lg:gap-12">
          <li className="relative group">
            <Link
              to={"/"}
              className="flex gap-2 items-center justify-center cursor-pointer text-lg lg:text-xl font-bold text-white hover:text-[#90EE90] transition-all duration-200 text-decoration-none"
            >
              <AiOutlineFundProjectionScreen fontSize={22} />
              <span>Generate Diet</span>
            </Link>
          </li>
          <li className="relative group">
            <Link
              to={"/tips"}
              className="flex gap-2 items-center justify-center cursor-pointer text-lg lg:text-xl font-bold text-white hover:text-[#90EE90] transition-all duration-200 text-decoration-none"
            >
              <FiFileText fontSize={22} />
              <span>Tips</span>
            </Link>
          </li>
        </ul>
      </div>
      <span>
        <div className="h-full lg:hidden flex items-center justify-center cursor-pointer relative">
          {showNav ? (
            <MdClose
              fontSize={25}
              className="text-white"
              onClick={() => setShowNav(!showNav)}
            />
          ) : (
            <RiMenu3Fill
              fontSize={25}
              className="text-white"
              onClick={() => setShowNav(!showNav)}
            />
          )}
          <MobileNav showNav={showNav} setShowNav={setShowNav} />
        </div>
      </span>
    </div>
  );
}

export default Header;