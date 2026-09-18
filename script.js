/**
 * script.js
 * ---------
 * Two jobs:
 *   1. Render the `projects` array (from data.js) into the #project-list.
 *   2. Type out `demoLines` (from data.js) into the terminal strip, once,
 *      on page load — the single orchestrated motion moment on this page.
 *
 * No build step, no dependencies — plain DOM APIs so this runs as-is on
 * GitHub Pages.
 */

/* ---------------------------------------------------------------------- */
/* 1. Render project rows                                                  */
/* ---------------------------------------------------------------------- */
function renderProjects() {
  const list = document.getElementById("project-list");
  if (!list || typeof projects === "undefined") return;

  if (projects.length === 0) {
    list.innerHTML = `<p class="section-note">Nothing published yet — check back soon.</p>`;
    return;
  }

  list.innerHTML = projects
    .map((project) => {
      const links = [];
      if (project.code) {
        links.push(`<a href="${escapeAttr(project.code)}" target="_blank" rel="noopener">Code ↗</a>`);
      }
      if (project.demo) {
        links.push(`<a href="${escapeAttr(project.demo)}" target="_blank" rel="noopener">Demo ↗</a>`);
      }

      const stackHtml = (project.stack || [])
        .map((tech) => `<span>${escapeHtml(tech)}</span>`)
        .join("");

      return `
        <article class="project-row">
          <div class="project-meta">
            <time>${escapeHtml(project.date || "")}</time>
          </div>
          <div class="project-title">
            <h3>${escapeHtml(project.title)}</h3>
            <div class="project-links">${links.join("")}</div>
          </div>
          <p class="project-desc">${escapeHtml(project.description || "")}</p>
          <div class="stack">${stackHtml}</div>
        </article>
      `;
    })
    .join("");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function escapeAttr(str) {
  return (str ?? "").replace(/"/g, "&quot;");
}

/* ---------------------------------------------------------------------- */
/* 2. Type out the terminal demo lines                                     */
/* ---------------------------------------------------------------------- */
function typeTerminal() {
  const body = document.getElementById("terminal-body");
  if (!body || typeof demoLines === "undefined" || demoLines.length === 0) return;

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduceMotion) {
    // Skip the animation; just render every line immediately.
    body.innerHTML = demoLines.map(lineHtml).join("");
    return;
  }

  let lineIndex = 0;

  function typeNextLine() {
    if (lineIndex >= demoLines.length) return;

    const { text, label, confidence } = demoLines[lineIndex];
    const promptText = `moderate("${text}")`;

    const lineEl = document.createElement("div");
    lineEl.className = "terminal-line";
    const promptSpan = document.createElement("span");
    promptSpan.className = "prompt";
    lineEl.appendChild(promptSpan);

    const cursor = document.createElement("span");
    cursor.className = "cursor";
    lineEl.appendChild(cursor);

    body.appendChild(lineEl);

    let charIndex = 0;
    const typingSpeedMs = 18;

    const interval = setInterval(() => {
      charIndex++;
      promptSpan.textContent = promptText.slice(0, charIndex);

      if (charIndex >= promptText.length) {
        clearInterval(interval);
        cursor.remove();

        const resultSpan = document.createElement("span");
        resultSpan.className = `result ${label}`;
        resultSpan.textContent = `  -> ${label}  ${confidence.toFixed(2)}`;
        lineEl.appendChild(resultSpan);

        lineIndex++;
        setTimeout(typeNextLine, 260);
      }
    }, typingSpeedMs);
  }

  typeNextLine();
}

function lineHtml({ text, label, confidence }) {
  return `<div class="terminal-line"><span class="prompt">moderate("${escapeHtml(text)}")</span><span class="result ${label}">  -> ${label}  ${confidence.toFixed(2)}</span></div>`;
}

/* ---------------------------------------------------------------------- */
/* Init                                                                     */
/* ---------------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  renderProjects();
  typeTerminal();
});
