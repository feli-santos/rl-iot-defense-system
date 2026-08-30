"use strict";
const { C, F, N } = require("./theme");
const { NOTES } = require("./notes");
const H = require("./helpers");

function addTitle(pres) {
  const s = H.newDark(pres, { notes: NOTES.title });
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
    notes: NOTES.speaker,
  });
  // Portrait (rounded card behind it keeps the deck's framing motif).
  H.img(s, "speaker_photo", { x: 0.55, y: 1.55, maxW: 2.35, maxH: 2.35, ext: "jpg" });

  s.addText("Felipe Augusto Oliveira dos Santos", {
    x: 3.3, y: 1.45, w: 6.1, h: 0.4, fontFace: F.head, fontSize: 19, bold: true, color: C.ink, margin: 0,
  });
  H.bullets(s, [
    { text: "M.Sc. Student, Electrical Engineering (Computer Engineering area) \u2014 FEEC / UNICAMP", bold: true },
    { text: "B.Sc. Electrical Engineering (Control & Automation emphasis) \u2014 UFPB. CNPq PIBIC scholarship, 2014\u20132020" },
    { text: "CAPES / Science Without Borders mobility \u2014 University of Wisconsin\u2013Milwaukee, 2015\u20132016" },
    { text: "IoT Platforms & Edge-AI Solutions Architect \u2014 Globant; aviation, smart venues, telecom, robotics, utilities, manufacturing, 2022\u2013present" },
    { text: "Research line: applied reinforcement learning (advisor: Prof. Dr. Denis Fantinato), 2024\u2013present" },
  ], { x: 3.3, y: 2.0, w: 6.1, h: 3.1, size: 12, gap: 9 });

  // Client logos, in the layout carried over from the Google Slides edit.
  H.img(s, "logo_latam", { x: 0.65, y: 4.22, maxW: 0.79, maxH: 0.24 });
  H.img(s, "logo_universal", { x: 1.60, y: 4.16, maxW: 0.66, maxH: 0.37 });
  H.img(s, "logo_disney", { x: 2.34, y: 4.01, maxW: 0.60, maxH: 0.60 });
  H.img(s, "logo_compass", { x: 0.65, y: 4.69, maxW: 0.37, maxH: 0.37 });
  H.img(s, "logo_compass_text", { x: 1.06, y: 4.67, maxW: 0.92, maxH: 0.40 });
  H.img(s, "logo_hme", { x: 2.02, y: 4.67, maxW: 0.45, maxH: 0.40 });
  H.img(s, "logo_bbva", { x: 2.55, y: 4.72, maxW: 0.60, maxH: 0.34 });
}

function addAgenda(pres) {
  const s = H.newContent(pres, {
    kicker: "Presentation",
    title: "Agenda",
    notes: NOTES.agenda,
  });
  const items = [
    ["1", "Context", "Why IoT defense outgrew static security \u2014 and the vocabulary we need (kill chain, RL, POMDP)", C.green],
    ["2", "Question & Objectives", "When does learned sequential control beat per-flow classification?", C.gold],
    ["3", "Framework", "Dataset projection, reactive attacker, partially observable environment, reward design", C.blue],
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
