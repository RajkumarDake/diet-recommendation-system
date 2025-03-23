const fs = require("fs");
const path = require("path");

function printTree(dir, indent = "") {
  const files = fs.readdirSync(dir);
  files.forEach((file, index) => {
    const fullPath = path.join(dir, file);
    const isLast = index === files.length - 1;
    const stats = fs.statSync(fullPath);
    console.log(`${indent}${isLast ? "└── " : "├── "}${file}`);
    if (stats.isDirectory()) {
      printTree(fullPath, indent + (isLast ? "    " : "│   "));
    }
  });
}

printTree(".");