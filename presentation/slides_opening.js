"use strict";
const { C, F, N } = require("./theme");
const H = require("./helpers");

function addTitle(pres) {
  const s = H.newDark(pres, {
    notes:
      "Good morning. Welcome the committee: Prof. Denis Fantinato (advisor), Prof. Alexandre Simoes, Prof. Joao Kleinschmidt. " +
      "State the title slowly. One-sentence framing: this dissertation asks WHEN a learning-based defender genuinely earns its place " +
      "in IoT security, and answers it with a controlled, reproducible experiment.",
  });
  // thin kill-chain color band across the top
  const bandW = H.PAGE.w / 5;
  for (let i = 0; i < 5; i++) {
    s.addShape("rect", { x: i * bandW, y: 0, w: bandW, h: 0.07, fill: { color: C.stage[i] }, line: { type: "none" } });
  }
  s.addText("MASTER'S DEFENSE", {
    x: 0.75, y: 0.78, w: 5.5, h: 0.3, fontFace: F.body, fontSize: 11.5, bold: true,
    charSpacing: 3.5, color: C.gold, margin: 0,
  });
  s.addText("A Deep Reinforcement Learning Framework for Autonomous Cybersecurity Defense in IoT Networks", {
    x: 0.75, y: 1.16, w: 8.5, h: 1.75, fontFace: F.head, fontSize: 29, bold: true,
    color: C.textOnDark, margin: 0, lineSpacingMultiple: 1.08,
  });
  s.addText("Um Arcabou\u00e7o de Aprendizado por Refor\u00e7o Profundo para Defesa Cibern\u00e9tica Aut\u00f4noma em Redes IoT", {
    x: 0.75, y: 2.98, w: 8.0, h: 0.55, fontFace: F.body, fontSize: 12.5, italic: true,
    color: C.mutedOnDark, margin: 0,
  });
  s.addShape("line", { x: 0.75, y: 3.72, w: 4.1, h: 0, line: { color: C.borderDark, width: 1 } });
  s.addText([
    { text: "Felipe Augusto Oliveira dos Santos", options: { fontSize: 15, bold: true, color: C.textOnDark, breakLine: true } },
    { text: "Advisor: Prof. Dr. Denis Fantinato", options: { fontSize: 11.5, color: C.mutedOnDark, breakLine: true } },
  ], { x: 0.75, y: 3.9, w: 5.4, h: 0.75, fontFace: F.body, margin: 0, lineSpacingMultiple: 1.25 });
  s.addText([
    { text: "Examination Committee:  Prof. Dr. Denis Fantinato (UNICAMP)  \u00b7  Prof. Dr. Alexandre da Silva Sim\u00f5es (UNESP)", options: { color: C.mutedOnDark, breakLine: true } },
    { text: "Prof. Dr. Jo\u00e3o Kleinschmidt (UFABC)", options: { color: C.mutedOnDark } },
  ], { x: 0.75, y: 4.66, w: 7.3, h: 0.44, fontFace: F.body, fontSize: 9.5, margin: 0 });
  s.addText("School of Electrical and Computer Engineering (FEEC)  \u00b7  Campinas \u2014 August 31, 2026", {
    x: 0.75, y: 5.14, w: 6.9, h: 0.26, fontFace: F.body, fontSize: 9.5, color: C.mutedOnDark, margin: 0,
  });
  // logo bottom-right
  H.img(s, "unicamp_logo_white", { x: 8.35, y: 4.1, maxW: 1.05, maxH: 1.15, align: "center" });
}

function addSpeaker(pres) {
  const s = H.newContent(pres, {
    kicker: "Presentation",
    title: "About the Speaker",
    notes:
      "TODO before the defense: fill in the biography placeholders (education, professional role, interests) and paste your photo " +
      "into the framed box. Keep it to 45 seconds \u2014 the committee knows you; this is for the public audience.",
  });
  // photo placeholder
  H.card(s, 0.55, 1.42, 2.35, 2.9, { fill: C.cardAlt });
  s.addText("PHOTO\n(replace in\nGoogle Slides)", {
    x: 0.55, y: 1.42, w: 2.35, h: 2.9, align: "center", valign: "middle",
    fontFace: F.body, fontSize: 11, color: C.muted, margin: 0,
  });
  s.addText("Felipe Augusto Oliveira dos Santos", {
    x: 3.3, y: 1.45, w: 6.1, h: 0.4, fontFace: F.head, fontSize: 19, bold: true, color: C.ink, margin: 0,
  });
  H.bullets(s, [
    { text: "M.Sc. candidate, Electrical Engineering (Computer Engineering area) \u2014 FEEC / UNICAMP", bold: true },
    { text: "[TODO: B.Sc. degree, institution, year]" },
    { text: "[TODO: current professional role / industry experience]" },
    { text: "[TODO: research interests \u2014 e.g. reinforcement learning, cybersecurity, IoT]" },
    { text: "Research line: adversarial reinforcement learning for autonomous IoT defense (advisor: Prof. Dr. Denis Fantinato)" },
    { text: "Journal article condensing this dissertation submitted to Elsevier Internet of Things (2026)" },
  ], { x: 3.3, y: 2.0, w: 6.1, h: 3.1, size: 12, gap: 9 });
  s.addText("Placeholders marked [TODO] \u2014 fill in before presenting.", {
    x: 3.3, y: 5.02, w: 6.0, h: 0.25, fontFace: F.body, fontSize: 9, italic: true, color: C.muted, margin: 0,
  });
}

function addAgenda(pres) {
  const s = H.newContent(pres, {
    kicker: "Presentation",
    title: "Agenda",
    notes:
      "Walk the agenda in ~40 seconds. Emphasize the shape of the talk: first I teach the background (kill chain + RL + partial " +
      "observability), then I state the research question, then the framework we built, then the evidence, then what it means. " +
      "Total ~35 minutes.",
  });
  const items = [
    ["1", "Context", "Why IoT defense outgrew static security \u2014 and the vocabulary we need (kill chain, RL, POMDP)", C.green],
    ["2", "Research Question", "When does learned sequential control beat per-flow classification?", C.gold],
    ["3", "Framework", "Dataset projection, reactive attacker, partially observable environment, reward contracts", C.blue],
    ["4", "Results", "The aliasing crossover \u00b7 reward ablation \u00b7 robustness \u00b7 zero-day-like classes", C.red],
    ["5", "Closing", "Findings, limitations, future work", C.ink],
  ];
  let y = 1.42;
  for (const [n, t, d, color] of items) {
    H.accentCard(s, 0.55, y, 8.9, 0.62, color);
    s.addText(n, {
      x: 0.72, y: y + 0.05, w: 0.5, h: 0.52, fontFace: F.head, fontSize: 21, bold: true,
      color, align: "center", valign: "middle", margin: 0,
    });
    s.addText([
      { text: t + "   ", options: { bold: true, fontSize: 13, color: C.ink } },
      { text: "\u2014  " + d, options: { fontSize: 10.5, color: C.inkSoft } },
    ], { x: 1.35, y: y + 0.05, w: 7.9, h: 0.52, fontFace: F.body, valign: "middle", margin: 0 });
    y += 0.72;
  }
}

module.exports = { addTitle, addSpeaker, addAgenda };
