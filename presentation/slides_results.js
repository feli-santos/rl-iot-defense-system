"use strict";
const { C, F, N } = require("./theme");
const { NOTES } = require("./notes");
const H = require("./helpers");

// Slide 19 — learning curves + reliability
function addLearning(pres) {
  const s = H.newContent(pres, {
    kicker: "4 \u00b7 Results",
    title: "Sparse-Reward Training: a Reliability Story",
    titleSize: 25,
    notes: NOTES.learning,
  });
  H.img(s, "F3_learning_curves", { x: 0.4, y: 1.42, maxW: 5.75, maxH: 3.7, frame: true, vAlign: "top" });
  const rows = [
    ["A2C", N.a4a2c, "sd " + N.sdA2C, C.blue, "tightest seeds \u2014 most reliable"],
    ["PPO", N.a4ppo, "sd " + N.sdPPO, C.blue, "stable plateau, close second"],
    ["DQN", N.a4dqn, "sd " + N.sdDQN, C.red, "destabilizes without shaping"],
  ];
  s.addText("Best checkpoint at \u03b1 = " + N.headlineAlpha + " (10 seeds)", {
    x: 6.55, y: 1.36, w: 2.95, h: 0.26, fontFace: F.body, fontSize: 10.5, bold: true, color: C.inkSoft, margin: 0,
  });
  let y = 1.68;
  for (const [name, val, sd, color, d] of rows) {
    H.accentCard(s, 6.55, y, 2.95, 0.88, color);
    s.addText([
      { text: name + "  ", options: { bold: true, fontSize: 13, color: C.ink } },
      { text: val + "  ", options: { bold: true, fontSize: 13, color } },
      { text: sd, options: { fontSize: 10, color: C.muted } },
    ], { x: 6.73, y: y + 0.08, w: 2.65, h: 0.3, fontFace: F.body, margin: 0 });
    s.addText(d, { x: 6.73, y: y + 0.42, w: 2.65, h: 0.4, fontFace: F.body, fontSize: 9.5, color: C.inkSoft, margin: 0 });
    y += 1.0;
  }
  s.addText("On-policy advantage = training-time stability, not peak return.", {
    x: 6.55, y: 4.78, w: 2.95, h: 0.55, fontFace: F.body, fontSize: 10, italic: true, color: C.inkSoft, margin: 0,
  });
}

// Slide 20 — two doctrines (F4)
function addDoctrines(pres) {
  const s = H.newContent(pres, {
    kicker: "4 \u00b7 Results",
    title: "Same Reward, Two Learned Doctrines",
    notes: NOTES.doctrines,
  });
  H.img(s, "F4_action_distribution", { x: 0.4, y: 1.38, maxW: 5.5, maxH: 3.62, frame: true, vAlign: "top" });
  H.accentCard(s, 6.15, 1.55, 3.35, 1.5, C.blue);
  s.addText([
    { text: "A2C \u2014 prevent at maneuver", options: { bold: true, fontSize: 12.5, color: C.ink, breakLine: true } },
    { text: "Blocks " + N.a2cManeuverBlock + "% of MANEUVER steps; suppresses the advance before impact; never isolates.", options: { fontSize: 10.5, color: C.inkSoft } },
  ], { x: 6.35, y: 1.7, w: 3.0, h: 1.2, fontFace: F.body, margin: 0 });
  H.accentCard(s, 6.15, 3.25, 3.35, 1.5, C.blue);
  s.addText([
    { text: "PPO \u2014 contain at impact", options: { bold: true, fontSize: 12.5, color: C.ink, breakLine: true } },
    { text: "Admits the late stage, then isolates at the boundary. Different doctrine, same reward.", options: { fontSize: 10.5, color: C.inkSoft } },
  ], { x: 6.35, y: 3.4, w: 3.0, h: 1.2, fontFace: F.body, margin: 0 });
  s.addText("Both keep aggressive action on benign traffic under 1%.", {
    x: 6.15, y: 4.9, w: 3.35, h: 0.3, fontFace: F.body, fontSize: 10, italic: true, color: C.muted, margin: 0,
  });
}

// Slide 21 — THE crossover (headline)
function addCrossover(pres) {
  const s = H.newContent(pres, {
    kicker: "4 \u00b7 Results \u2014 headline",
    title: "The Aliasing Crossover",
    kickerColor: C.red,
    notes: NOTES.crossover,
  });
  H.img(s, "Falpha_curve", { x: 0.4, y: 1.42, maxW: 5.85, maxH: 3.6, frame: true, vAlign: "top" });
  // right rail: the three reads
  const reads = [
    ["Tie at \u03b1 = 0", "PPO " + N.a0ppo + " vs RF " + N.a0rf + " \u2014 overlapping CIs. No built-in RL favoritism.", C.green],
    ["RF collapses", "monotone fall to " + N.a10rf + " at \u03b1 = 1 \u2014 net-harmful under full ambiguity.", C.red],
    ["PPO stays flat", "disjoint CIs from \u03b1 = " + N.headlineAlpha + " on; gap " + N.a4gap + " widening to " + N.a10gap + ".", C.blue],
  ];
  let y = 1.46;
  for (const [t, d, color] of reads) {
    H.accentCard(s, 6.6, y, 2.9, 1.02, color);
    s.addText([
      { text: t, options: { bold: true, fontSize: 12, color: C.ink, breakLine: true } },
      { text: d, options: { fontSize: 9.5, color: C.inkSoft } },
    ], { x: 6.78, y: y + 0.08, w: 2.58, h: 0.88, fontFace: F.body, margin: 0 });
    y += 1.14;
  }
  s.addText("Oracle ceiling " + N.oracle + " \u2014 the price of perfect perception.", {
    x: 6.6, y: 4.92, w: 2.9, h: 0.28, fontFace: F.body, fontSize: 9.5, italic: true, color: C.muted, margin: 0,
  });
}

// Slide 22 — reward-coupling ablation
function addCoupling(pres) {
  const s = H.newContent(pres, {
    kicker: "4 \u00b7 Results",
    title: "The Advantage Survives Removing the Shaping",
    titleSize: 24,
    notes: NOTES.coupling,
  });
  H.img(s, "Fcoupling_reward_gap", { x: 0.4, y: 1.42, maxW: 5.35, maxH: 3.2, frame: true, vAlign: "top" });
  H.accentCard(s, 6.35, 1.55, 3.15, 1.32, C.gold);
  s.addText([
    { text: "Coupled (shaped)", options: { bold: true, fontSize: 12, color: C.ink, breakLine: true } },
    { text: "best RL: DQN " + N.cplDQN, options: { fontSize: 11, color: C.inkSoft, breakLine: true } },
    { text: "RF \u2212 best-RL = " + N.cplGap, options: { bold: true, fontSize: 11.5, color: C.blue } },
  ], { x: 6.55, y: 1.68, w: 2.8, h: 1.05, fontFace: F.body, margin: 0 });
  H.accentCard(s, 6.35, 3.05, 3.15, 1.32, C.blue);
  s.addText([
    { text: "Outcome (sparse)", options: { bold: true, fontSize: 12, color: C.ink, breakLine: true } },
    { text: "best RL: A2C " + N.outA2C, options: { fontSize: 11, color: C.inkSoft, breakLine: true } },
    { text: "RF \u2212 best-RL = " + N.outGap, options: { bold: true, fontSize: 11.5, color: C.blue } },
  ], { x: 6.55, y: 3.18, w: 2.8, h: 1.05, fontFace: F.body, margin: 0 });
  H.card(s, 0.55, 4.72, 8.95, 0.52, { fill: C.cardAlt });
  s.addText([
    { text: "Shaping changes which algorithm wins \u2014 not whether learned control leads.  ", options: { bold: true, color: C.ink } },
    { text: "(DQN: " + N.cplDQN + " coupled \u2192 " + N.outDQN + " sparse.)", options: { color: C.inkSoft } },
  ], { x: 0.8, y: 4.72, w: 8.5, h: 0.52, fontFace: F.body, fontSize: 11, valign: "middle", margin: 0 });
}

// Slide 23 — robustness sweeps (F10 + F17)
function addSweeps(pres) {
  const s = H.newContent(pres, {
    kicker: "4 \u00b7 Results",
    title: "Robustness: Harsher Environments, Evasive Attackers",
    titleSize: 25,
    notes: NOTES.sweeps,
  });
  s.addText("Environment difficulty (de-escalation prob.)", {
    x: 0.4, y: 1.26, w: 4.55, h: 0.24, fontFace: F.body, fontSize: 10.5, bold: true, color: C.inkSoft, align: "center", margin: 0,
  });
  s.addText("Evasive persistence (post-detection hardening)", {
    x: 5.1, y: 1.26, w: 4.55, h: 0.24, fontFace: F.body, fontSize: 10.5, bold: true, color: C.inkSoft, align: "center", margin: 0,
  });
  const im1 = H.img(s, "F10_aggressiveness", { x: 0.4, y: 1.68, maxW: 4.55, maxH: 2.72, frame: true, vAlign: "top" });
  const im2 = H.img(s, "F17_evasion_sweep", { x: 5.1, y: 1.68, maxW: 4.55, maxH: 2.72, frame: true, vAlign: "top" });
  H.accentCard(s, 0.55, 4.62, 4.3, 0.6, C.blue);
  s.addText("A2C leads at every difficulty; widest on-policy margin at the harshest setting.", {
    x: 0.75, y: 4.62, w: 4.0, h: 0.6, fontFace: F.body, fontSize: 10, color: C.inkSoft, valign: "middle", margin: 0,
  });
  H.accentCard(s, 5.15, 4.62, 4.3, 0.6, C.blue);
  s.addText("A2C clears the pre-registered 25% criterion (" + N.f17a2c0 + " \u2192 " + N.f17a2c75 + "); PPO narrowly misses.", {
    x: 5.35, y: 4.62, w: 4.0, h: 0.6, fontFace: F.body, fontSize: 10, color: C.inkSoft, valign: "middle", margin: 0 });
}

// Slide 24 — OOD grid
function addOOD(pres) {
  const s = H.newContent(pres, {
    kicker: "4 \u00b7 Results",
    title: "Ten Attack Classes the System Never Saw",
    notes: NOTES.ood,
  });
  H.img(s, "F15_ood_robustness", { x: 0.35, y: 1.32, maxW: 4.9, maxH: 3.72, frame: true, vAlign: "top" });

  // Feedback [25]: "10 classes? Aumentou?" — no: these are the SAME ten
  // reserved back on the projection slide. Say so explicitly.
  s.addText("The same 10 classes reserved earlier \u2014 never trained on, by detector or agent.", {
    x: 5.6, y: 1.24, w: 3.85, h: 0.24, fontFace: F.body, fontSize: 9.5, italic: true, color: C.muted, margin: 0,
  });

  // Feedback [25]: "Metrica Prevention Rate nao ficou muito claro." Define it.
  H.card(s, 5.6, 1.54, 3.85, 0.74, { fill: C.cardAlt });
  s.addText([
    { text: "Prevention rate  P", options: { bold: true, color: C.ink } },
    { text: "prev", options: { bold: true, color: C.ink, fontSize: 7 } },
    { text: "  =  Pr( attacker never reaches IMPACT )\n", options: { bold: true, color: C.ink } },
    { text: "fraction of episodes held below IMPACT for all 100 steps \u2014 not accuracy, not detection.", options: { color: C.inkSoft } },
  ], { x: 5.74, y: 1.61, w: 3.6, h: 0.6, fontFace: F.body, fontSize: 9, margin: 0, valign: "top" });

  H.stat(s, { x: 5.6, y: 2.42, w: 3.85, value: N.oodRLlo + " \u2013 " + N.oodRLhi, label: "best windowed RL (A2C): prevention on every held-out class", color: C.blue, valueSize: 25 });
  H.stat(s, { x: 5.6, y: 3.32, w: 3.85, value: N.oodRFlo + " \u2013 " + N.oodRFhi, label: "RF-Acting: prevention on the same classes", color: C.red, valueSize: 25 });
  H.stat(s, { x: 5.6, y: 4.16, w: 3.85, value: N.oodAdvLo + " to " + N.oodAdvHi, label: "advantage on every single class \u2014 none negative", color: C.green, valueSize: 25 });
  s.addText("(always-BLOCK reaches 1.0 only by disrupting 100% of benign traffic \u2014 inadmissible; learned agents stay under 1%.)", {
    x: 5.6, y: 5.0, w: 3.85, h: 0.4, fontFace: F.body, fontSize: 8.5, italic: true, color: C.muted, margin: 0,
  });
}

// Slide 25 — recall independence + mechanism
function addRecallIndependence(pres) {
  const s = H.newContent(pres, {
    kicker: "4 \u00b7 Results",
    title: "No Dependence on Detector Recall",
    notes: NOTES.recall,
  });
  H.img(s, "F15b_recall_vs_advantage", { x: 0.4, y: 1.38, maxW: 5.35, maxH: 3.7, frame: true, vAlign: "top" });
  H.bullets(s, [
    { text: "Spearman \u03c1 = " + N.spearman + " (p = " + N.spearmanP + ") \u00b7 Pearson r = " + N.pearson + " (p = " + N.pearsonP + ")", bold: true },
    { text: "OLS slope 95% CI [" + N.olsLo + ", " + N.olsHi + "] \u2014 spans zero" },
    { text: "Advantage as large where the detector sees almost perfectly as where it is nearly blind" },
    { text: "Mechanism: RF acts passively on ~2/3 of steps \u2192 attacker reaches impact \u2192 late block = mitigated, not prevented", sub: false },
  ], { x: 6.0, y: 1.6, w: 3.5, h: 2.9, size: 11, gap: 12 });
  s.addText("Caveat stated plainly: n = 10 classes \u2014 no detectable trend, not proof of independence.", {
    x: 6.0, y: 4.75, w: 3.45, h: 0.55, fontFace: F.body, fontSize: 9.5, italic: true, color: C.muted, margin: 0,
  });
}

// Slide 26 — limitations
function addLimitations(pres) {
  const s = H.newContent(pres, {
    kicker: "5 \u00b7 Closing",
    title: "Limitations \u2014 Stated, Not Hidden",
    kickerColor: C.ink,
    notes: NOTES.limitations,
  });
  const rows = [
    ["Modeling abstractions", "Session coherence + aliasing are environment-layer constructs; \u03b1 is a knob, not a measured deployment property"],
    ["Conditional advantage", "At \u03b1 = 0 the claim is a tie \u2014 the win emerges only as ambiguity grows"],
    ["Designed attacker", "Reactive but not co-trained; crossover is conditional on this attacker class \u2014 self-play deferred"],
    ["Memoryless baseline", "Isolates windowed vs per-flow control \u2014 a windowed supervised baseline is named follow-up work"],
    ["Single dataset", "CICIoT2023 only; replication on Bot-IoT required before claiming dataset independence"],
    ["OOD scope", "Feature-injection stress test at n = 10 classes \u2014 not deployed zero-day generalization"],
  ];
  let y = 1.42; let x = 0.55;
  for (let i = 0; i < rows.length; i++) {
    const [t, d] = rows[i];
    H.card(s, x, y, 4.35, 1.14);
    s.addShape("rect", { x, y, w: 4.35, h: 0.055, fill: { color: C.ink }, line: { type: "none" } });
    s.addText(t, { x: x + 0.16, y: y + 0.12, w: 4.05, h: 0.27, fontFace: F.body, fontSize: 11.5, bold: true, color: C.ink, margin: 0 });
    s.addText(d, { x: x + 0.16, y: y + 0.41, w: 4.05, h: 0.68, fontFace: F.body, fontSize: 9.5, color: C.inkSoft, margin: 0 });
    x += 4.55;
    if (i % 2 === 1) { x = 0.55; y += 1.3; }
  }
}

// Slide 27 — conclusions
function addConclusions(pres) {
  const s = H.newContent(pres, {
    kicker: "5 \u00b7 Closing",
    title: "What This Dissertation Establishes",
    kickerColor: C.ink,
    notes: NOTES.conclusions,
  });
  const rows = [
    ["1", "The crossover is real and controlled", "Tie at \u03b1 = 0 \u2192 disjoint CIs from \u03b1 = " + N.headlineAlpha + " \u2192 RF net-harmful at \u03b1 = 1 while windowed PPO holds flat", C.red],
    ["2", "It is not a reward artifact", "Best learned agent leads under both reward functions (" + N.cplGap + " coupled, " + N.outGap + " sparse)", C.gold],
    ["3", "It extends to unseen attack classes", "Prevention advantage " + N.oodAdvLo + "\u2013" + N.oodAdvHi + " on all 10 held-out classes \u2014 independent of detector recall", C.green],
  ];
  let y = 1.45;
  for (const [n, t, d, color] of rows) {
    H.accentCard(s, 0.55, y, 8.9, 0.85, color);
    s.addText(n, { x: 0.75, y: y + 0.1, w: 0.45, h: 0.65, fontFace: F.head, fontSize: 24, bold: true, color, valign: "middle", margin: 0 });
    s.addText([
      { text: t, options: { bold: true, fontSize: 13, color: C.ink, breakLine: true } },
      { text: d, options: { fontSize: 10.5, color: C.inkSoft } },
    ], { x: 1.35, y: y + 0.1, w: 7.95, h: 0.68, fontFace: F.body, valign: "middle", margin: 0 });
    y += 0.98;
  }
  // Feedback [28]: the footprint detail was a whole talking point at the end
  // of a long talk. Compress to a clause; the numbers live in backup B7.
  H.card(s, 0.55, 4.5, 8.9, 0.72, { fill: C.cardAlt });
  s.addText([
    { text: "And it is operationally deployable: ", options: { bold: true, color: C.ink } },
    { text: "under 1% benign disruption, a policy small enough for an edge gateway, and reproducible end-to-end (" + N.tests + " tests, hash-pinned figures).", options: { color: C.inkSoft } },
  ], { x: 0.8, y: 4.58, w: 8.5, h: 0.56, fontFace: F.body, fontSize: 11.5, valign: "middle", margin: 0 });
}

// Slide 28 — future work + publication
function addFuture(pres) {
  const s = H.newContent(pres, {
    kicker: "5 \u00b7 Closing",
    title: "Where This Goes Next",
    kickerColor: C.ink,
    notes: NOTES.future,
  });
  const cols = [
    ["Strengthen the claim", ["Windowed supervised baseline (named follow-up)", "Recurrent belief-state policies", "Second dataset: Bot-IoT replication"], C.blue],
    ["Harder adversaries", ["Co-adaptive self-play attacker", "Stage-skipping / non-monotonic chains", "Multi-agent cooperative defense"], C.red],
    ["Toward deployment", ["Edge-hardware cost quantification", "Federated multi-site training", "Constrained-MDP FPR guarantees"], C.green],
  ];
  let x = 0.55;
  for (const [t, items, color] of cols) {
    H.card(s, x, 1.45, 2.87, 2.6);
    s.addShape("rect", { x, y: 1.45, w: 2.87, h: 0.06, fill: { color }, line: { type: "none" } });
    s.addText(t, { x: x + 0.16, y: 1.6, w: 2.55, h: 0.3, fontFace: F.body, fontSize: 12.5, bold: true, color: C.ink, margin: 0 });
    H.bullets(s, items, { x: x + 0.16, y: 2.0, w: 2.6, h: 1.95, size: 10, gap: 9, bulletColor: color });
    x += 3.02;
  }
  H.accentCard(s, 0.55, 4.35, 8.9, 0.85, C.gold);
  s.addText([
    { text: "Dissemination: ", options: { bold: true, color: C.ink, fontSize: 12 } },
    { text: "condensed journal article submitted to Elsevier Internet of Things (2026) \u00b7 code + hash-chain manifests public on GitHub (feli-santos/rl-iot-defense-system)", options: { color: C.inkSoft, fontSize: 11.5 } },
  ], { x: 0.8, y: 4.45, w: 8.5, h: 0.65, fontFace: F.body, valign: "middle", margin: 0 });
}

// Slide 29 — thanks
function addThanks(pres) {
  const s = H.newDark(pres, {
    notes: NOTES.thanks,
  });
  const bandW = H.PAGE.w / 5;
  for (let i = 0; i < 5; i++) {
    s.addShape("rect", { x: i * bandW, y: H.PAGE.h - 0.07, w: bandW, h: 0.07, fill: { color: C.stage[i] }, line: { type: "none" } });
  }
  s.addText("Thank you.", {
    x: 0.75, y: 1.7, w: 8.5, h: 0.9, fontFace: F.head, fontSize: 44, bold: true, color: C.textOnDark, margin: 0,
  });
  s.addText("Committee \u00b7 colleagues \u00b7 family \u2014 obrigado.", {
    x: 0.75, y: 2.75, w: 8.5, h: 0.4, fontFace: F.body, fontSize: 15, italic: true, color: C.mutedOnDark, margin: 0,
  });
  s.addText([
    { text: "Felipe Augusto Oliveira dos Santos \u00b7 FEEC / UNICAMP", options: { breakLine: true, color: C.textOnDark, fontSize: 12 } },
    { text: "Advisor: Prof. Dr. Denis Fantinato", options: { breakLine: true, color: C.mutedOnDark, fontSize: 11 } },
    { text: "Code & reproducibility: github.com/feli-santos/rl-iot-defense-system", options: { color: C.mutedOnDark, fontSize: 11 } },
  ], { x: 0.75, y: 3.55, w: 6.5, h: 1.0, fontFace: F.body, margin: 0, lineSpacingMultiple: 1.3 });
  H.img(s, "unicamp_logo_white", { x: 8.3, y: 3.9, maxW: 1.0, maxH: 1.1 });
  s.addText("Backup slides follow \u2192", {
    x: 7.4, y: 5.2, w: 2.0, h: 0.26, align: "right", fontFace: F.body, fontSize: 9, color: C.mutedOnDark, margin: 0,
  });
}

module.exports = {
  addLearning, addDoctrines, addCrossover, addCoupling, addSweeps,
  addOOD, addRecallIndependence, addLimitations, addConclusions,
  addFuture, addThanks,
};
