import React from "react";
import { AiOutlineFundProjectionScreen, AiFillStar } from "react-icons/ai";
import { FiFileText } from "react-icons/fi";
import { DiGitBranch } from "react-icons/di";
import { Link } from "react-router-dom";

function MobileNav({ showNav, setShowNav }) {
  return (
    <div
      className={
        showNav
          ? "fixed h-1/2 bg-gradient-to-r from-purple-900 to-violet-900 w-full top-[68px] left-0 rounded-b-md border-2 border-t-0 border-purple-700 z-50 transition-all duration-[500ms] overflow-hidden"
          : "fixed h-0 bg-gradient-to-r from-purple-900 to-violet-900 w-full top-[68px] left-0 rounded-b-md z-50 transition-all duration-[500ms] overflow-hidden"
      }
    >
      <div className="flex items-center justify-center text-white h-full">
        <ul className="flex gap-6 flex-col mt-10 text-center">
          <li className="relative group">
            <Link
              to={"/"}
              className="flex gap-2 items-center justify-center cursor-pointer text-lg font-bold text-[#90EE90] hover:text-white transition-all duration-200"
              onClick={() => setShowNav(false)}
            >
              <AiOutlineFundProjectionScreen fontSize={22} />
              <span>Generate Diet</span>
            </Link>
          </li>
          <li className="relative group">
            <Link
              to={"/tips"}
              className="flex gap-2 items-center justify-center cursor-pointer text-lg font-bold text-[#90EE90] hover:text-white transition-all duration-200"
              onClick={() => setShowNav(false)}
            >
              <FiFileText fontSize={22} />
              <span>Tips</span>
            </Link>
          </li>
        </ul>

        <a
          href="https://github.com/Gummadijahnavi"
          target="_blank"
          rel="noreferrer"
          className="flex w-24 my-5 mx-auto gap-2 justify-center items-center text-lg bg-fuchsia-900 px-3 py-[3px] border border-purple-700 rounded-sm hover:bg-fuchsia-800 transition-all duration-200"
        >
          <DiGitBranch fontSize={18} />
          <AiFillStar fontSize={18} />
        </a>
      </div>
    </div>
  );
}

export default MobileNav;