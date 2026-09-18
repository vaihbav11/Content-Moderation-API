/**
 * data.js
 * -------
 * Edit this file to add, remove, or update portfolio projects.
 * Each project renders as one row on the site — no HTML editing needed.
 *
 * Fields:
 *   title       (string)          - project name
 *   date        (string)          - e.g. "Mar 2026"
 *   description (string)          - 1-3 sentences
 *   stack       (array of string) - tech tags shown as pills
 *   code        (string | null)   - link to the GitHub repo
 *   demo        (string | null)   - link to a live demo, or null if none
 */

const projects = [
  {
    title: "Content Moderation API",
    date: "Mar 2026",
    description:
      "Fine-tuned DistilBERT on 15K labeled social posts to flag toxic content at 91% precision, then shipped it as a scalable REST API architected for throughput and low latency.",
    stack: ["Python", "FastAPI", "PyTorch", "Transformers", "DistilBERT"],
    code: "https://github.com/vaihbav11/Content-Moderation-API",
    demo: null,
  },
];

// Sample request/response pairs shown in the typed terminal strip on the
// homepage. Purely illustrative — edit or add lines to fit your project.
const demoLines = [
  { text: "You're worthless and everyone hates you.", label: "toxic", confidence: 0.97 },
  { text: "Great meeting today, thanks for the notes!", label: "safe", confidence: 0.99 },
  { text: "Shut up, nobody asked for your opinion.", label: "toxic", confidence: 0.91 },
];
