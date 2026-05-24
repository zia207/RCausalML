(function () {
  "use strict";

  function headingId(heading) {
    if (heading.id) return heading.id;
    var anchorId = heading.getAttribute("data-anchor-id");
    if (anchorId) return anchorId;
    var section = heading.closest("section[id]");
    if (section) return section.id;
    return null;
  }

  function headingText(heading) {
    return heading.textContent.replace(/\s+/g, " ").trim();
  }

  function initTutorialToc() {
    var content = document.getElementById("quarto-document-content");
    var quartoContent = document.getElementById("quarto-content");
    if (!content || !quartoContent || document.getElementById("rcausalml-tutorial-toc")) {
      return;
    }

    var layout = document.createElement("div");
    layout.className = "rcausalml-tutorial-layout";

    var aside = document.createElement("aside");
    aside.id = "rcausalml-tutorial-toc";
    aside.className = "rcausalml-tutorial-toc";
    aside.innerHTML =
      '<h2 class="rcausalml-toc-title">On this page</h2>' +
      '<nav id="rcausalml-toc-nav" aria-label="Table of contents"></nav>';

    quartoContent.parentNode.insertBefore(layout, quartoContent);
    layout.appendChild(aside);
    layout.appendChild(quartoContent);

    var headings = content.querySelectorAll("h1, h2, h3, h4");
    var nav = aside.querySelector("#rcausalml-toc-nav");
    var list = document.createElement("ul");
    list.className = "rcausalml-toc-list";

    headings.forEach(function (heading) {
      var id = headingId(heading);
      var text = headingText(heading);
      if (!id || !text) return;

      if (!heading.id) heading.id = id;

      var level = parseInt(heading.tagName.charAt(1), 10);
      var item = document.createElement("li");
      item.className = "rcausalml-toc-item rcausalml-toc-level-" + level;

      var link = document.createElement("a");
      link.href = "#" + id;
      link.className = "rcausalml-toc-link";
      link.textContent = text;
      item.appendChild(link);
      list.appendChild(item);
    });

    if (!list.children.length) {
      layout.classList.add("rcausalml-tutorial-layout--no-toc");
      aside.remove();
      return;
    }

    nav.appendChild(list);

    var links = nav.querySelectorAll(".rcausalml-toc-link");
    var sections = Array.prototype.map.call(links, function (link) {
      return document.getElementById(link.getAttribute("href").slice(1));
    }).filter(Boolean);

    if (!sections.length || !("IntersectionObserver" in window)) return;

    var activeLink = null;
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var id = entry.target.id;
          links.forEach(function (link) {
            var isActive = link.getAttribute("href") === "#" + id;
            link.classList.toggle("active", isActive);
            if (isActive) activeLink = link;
          });
        });
      },
      { rootMargin: "-15% 0px -75% 0px", threshold: 0 }
    );

    sections.forEach(function (section) {
      observer.observe(section);
    });

    if (window.location.hash) {
      links.forEach(function (link) {
        link.classList.toggle("active", link.getAttribute("href") === window.location.hash);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initTutorialToc);
  } else {
    initTutorialToc();
  }
})();
