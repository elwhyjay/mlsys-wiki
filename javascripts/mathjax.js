window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    // \boldsymbol 등은 기본 번들에 없어 명시적으로 로드해야 렌더됨
    packages: { "[+]": ["boldsymbol"] }
  },
  loader: { load: ["[tex]/boldsymbol"] },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})
