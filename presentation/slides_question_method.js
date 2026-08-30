"use strict";
const { C, F, N } = require("./theme");
const { NOTES } = require("./notes");
const H = require("./helpers");

// Slide 9 — the central question
function addQuestion(pres) {
  const s = H.newDark(pres, { notes: NOTES.question });
  s.addText("2 \u00b7 RESEARCH QUESTION", {
    x: 0.75, y: 0.6, w: 5, h: 0.3, fontFace: F.body, fontSize: 11, bold: true, charSpacing: 3, color: C.gold, margin: 0,
  });
  s.addText("When does a learned sequential policy\nbeat a memoryless per-flow classifier?", {
    x: 0.75, y: 1.02, w: 8.5, h: 1.0, fontFace: F.head, fontSize: 25, bold: true, color: C.textOnDark, margin: 0, lineSpacingMultiple: 1.06,
  });

  // How a supervised classifier is actually wired into a defense loop.
  // Feedback [9]: "explicar como um classificador supervisionado pode ser usado".
  s.addText("How a classifier defends \u2014 the baseline we must beat (RF-Acting):", {
    x: 0.75, y: 2.24, w: 8.5, h: 0.24, fontFace: F.body, fontSize: 11, bold: true,
    color: C.mutedOnDark, margin: 0,
  });
  const chain = [
    ["one flow", "a single 29-feature row", C.mutedOnDark],
    ["RandomForest", "predicts the stage", C.blueSoft],
    ["lookup table", "stage \u2192 recommended action", C.blueSoft],
    ["act", "no memory of the past", C.redOnDark],
  ];
  let cx = 0.75;
  const cw = 1.95, cgap = 0.24;
  chain.forEach(([t, d, col], i) => {
    s.addShape("rect", { x: cx, y: 2.54, w: cw, h: 0.66, fill: { color: "2E2E33" }, line: { color: C.borderDark, width: 1 } });
    s.addText(t, {
      x: cx, y: 2.60, w: cw, h: 0.26, fontFace: F.body, fontSize: 11, bold: true,
      color: col, align: "center", margin: 0,
    });
    s.addText(d, {
      x: cx + 0.06, y: 2.86, w: cw - 0.12, h: 0.3, fontFace: F.body, fontSize: 8.5,
      color: C.mutedOnDark, align: "center", margin: 0,
    });
    if (i < chain.length - 1) {
      s.addText("\u25B8", {
        x: cx + cw, y: 2.74, w: cgap, h: 0.26, fontFace: F.body, fontSize: 13,
        color: C.mutedOnDark, align: "center", valign: "middle", margin: 0,
      });
    }
    cx += cw + cgap;
  });

  // the objection card
  s.addShape("rect", { x: 0.75, y: 3.52, w: 8.5, h: 0.86, fill: { color: "2E2E33" }, line: { color: C.borderDark, width: 1 } });
  s.addShape("rect", { x: 0.75, y: 3.52, w: 0.07, h: 0.86, fill: { color: C.redOnDark }, line: { type: "none" } });
  s.addText([
    { text: "The skeptic's objection this thesis answers:  ", options: { bold: true, color: C.redOnDark, fontSize: 12 } },
    { text: "\u201cIf each flow reveals the attack stage, your RL agent is just an expensive classifier \u2014 a supervised model should match it.\u201d", options: { italic: true, color: C.textOnDark, fontSize: 12 } },
  ], { x: 1.02, y: 3.62, w: 8.0, h: 0.68, fontFace: F.body, margin: 0 });

  s.addText([
    { text: "Answer strategy: ", options: { bold: true, color: C.textOnDark } },
    { text: "make stage ambiguity a controlled dial (aliasing rate \u03b1), keep everything else fixed, and measure where classification stops being enough.", options: { color: C.mutedOnDark } },
  ], { x: 0.75, y: 4.6, w: 8.5, h: 0.6, fontFace: F.body, fontSize: 12, margin: 0 });
}

// Slide 10 — objectives (general + specific). Added after the first dry run:
// the committee asked for an explicit objectives slide, and the dissertation
// states its aims as a general goal + five enumerated contributions
// (tex/introduction.tex:18-30) rather than an ABNT-style objectives section.
function addObjectives(pres) {
  const s = H.newContent(pres, {
    kicker: "2 \u00b7 Question & Objectives",
    title: "Objectives",
    kickerColor: C.gold,
    notes: NOTES.objectives,
  });

  // --- general objective ----------------------------------------------------
  H.accentCard(s, 0.55, 1.32, 8.9, 1.06, C.gold);
  s.addText("General objective", {
    x: 0.78, y: 1.42, w: 8.4, h: 0.24, fontFace: F.body, fontSize: 10.5, bold: true,
    charSpacing: 1, color: C.gold, margin: 0,
  });
  s.addText("Design, implement, and evaluate a closed-loop adaptive defense framework for IoT networks in which a kill-chain-aware RL defender faces a reactive adversary over real CICIoT2023 traffic \u2014 posed as a genuine partial-observability problem, not a disguised classification problem.", {
    x: 0.78, y: 1.68, w: 8.45, h: 0.64, fontFace: F.body, fontSize: 11.5,
    color: C.ink, margin: 0, valign: "top",
  });

  // --- specific objectives --------------------------------------------------
  s.addText("Specific objectives", {
    x: 0.55, y: 2.55, w: 8.9, h: 0.24, fontFace: F.body, fontSize: 10.5, bold: true,
    charSpacing: 1, color: C.inkSoft, margin: 0,
  });

  const objs = [
    ["1", "Build a partially observable kill-chain environment", "real traffic \u00b7 reactive attacker \u00b7 tunable ambiguity \u03b1", C.blue],
    ["2", "Locate where windowed control overtakes per-flow control", "sweep \u03b1, hold everything else fixed", C.red],
    ["3", "Test whether the advantage depends on a privileged reward", "retrain under shaped and sparse reward alike", C.gold],
    ["4", "Measure generalization to attack classes never trained on", "ten held-out classes, prevention rate", C.green],
    ["5", "Report algorithm reliability and make it all reproducible", "10 seeds \u00b7 hash-pinned artifacts \u00b7 " + N.tests + " tests", C.ink],
  ];
  let y = 2.84;
  for (const [n, t, d, color] of objs) {
    s.addShape("ellipse", { x: 0.6, y: y + 0.04, w: 0.28, h: 0.28, fill: { color }, line: { type: "none" } });
    s.addText(n, {
      x: 0.6, y: y + 0.04, w: 0.28, h: 0.28, fontFace: F.body, fontSize: 11, bold: true,
      color: "FFFFFF", align: "center", valign: "middle", margin: 0,
    });
    s.addText([
      { text: t + "   ", options: { bold: true, fontSize: 12, color: C.ink } },
      { text: "\u2014  " + d, options: { fontSize: 10, color: C.inkSoft } },
    ], { x: 1.02, y: y, w: 8.4, h: 0.36, fontFace: F.body, valign: "middle", margin: 0 });
    y += 0.42;
  }

  s.addText("Each specific objective maps one-to-one onto a contribution \u2014 and onto a results slide later in this talk.", {
    x: 0.55, y: 4.98, w: 8.9, h: 0.28, fontFace: F.body, fontSize: 10.5, italic: true,
    color: C.muted, margin: 0,
  });
}

// Slide 11 — contributions map
function addContributions(pres) {
  const s = H.newContent(pres, {
    kicker: "2 \u00b7 Question & Objectives",
    title: "Five Contributions \u2014 the Map of This Talk",
    kickerColor: C.gold,
    notes: NOTES.contributions,
  });
  const cards = [
    ["1", "A genuinely partially observable kill-chain environment", "Reactive escalation attacker + real CICIoT2023 features + controllable aliasing \u03b1", C.blue],
    ["2", "A controlled crossover: windowed RL vs per-flow control", "Tie at \u03b1 = 0 by design; disjoint CIs from \u03b1 = " + N.headlineAlpha + " onward", C.red],
    ["3", "A reward ablation that answers the privileged-reward objection", "Best learned agent beats RF-Acting under both reward functions", C.gold],
    ["4", "A held-out attack-class prevention advantage", "Higher prevention on all 10 unseen classes \u2014 no dependence on detector recall", C.green],
    ["5", "Algorithm reliability + reproducibility by construction", "On-policy stability finding \u00b7 hash-pinned manifests \u00b7 " + N.tests + " tests", C.ink],
  ];
  let y = 1.38;
  for (const [n, t, d, color] of cards) {
    H.accentCard(s, 0.55, y, 8.9, 0.64, color);
    s.addText(n, { x: 0.72, y: y + 0.05, w: 0.42, h: 0.54, fontFace: F.head, fontSize: 20, bold: true, color, valign: "middle", margin: 0 });
    s.addText([
      { text: t, options: { bold: true, fontSize: 12.5, color: C.ink, breakLine: true } },
      { text: d, options: { fontSize: 10, color: C.inkSoft } },
    ], { x: 1.25, y: y + 0.04, w: 8.05, h: 0.56, fontFace: F.body, valign: "middle", margin: 0 });
    y += 0.74;
  }
}

// Slide 10 — architecture
function addArchitecture(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "Framework at a Glance",
    notes: NOTES.architecture,
  });
  H.img(s, "architecture_diagram", { x: 0.4, y: 1.3, maxW: 5.7, maxH: 3.72, frame: true, vAlign: "top" });
  const pts = [
    { text: "Offline: kill-chain projection \u00b7 closed-form attacker \u00b7 supervised stage detector", bold: false },
    { text: "Online: attacker \u2192 realization engine \u2192 windowed obs \u2192 defender \u2192 escalation kernel" },
    { text: "Evaluation: held-out benchmark vs baselines + oracle, then ablations" },
  ];
  H.bullets(s, pts, { x: 6.35, y: 1.6, w: 3.15, h: 2.2, size: 11, gap: 12 });
  H.accentCard(s, 6.35, 3.85, 3.1, 1.15, C.blue);
  s.addText([
    { text: "Closed loop, not trace replay.  ", options: { bold: true, color: C.blue, fontSize: 11.5, breakLine: true } },
    { text: "Every defender action changes the attacker's next move \u2014 evaluation included.", options: { color: C.inkSoft, fontSize: 10.5 } },
  ], { x: 6.55, y: 3.98, w: 2.75, h: 0.9, fontFace: F.body, margin: 0 });
}

// Slide 11 — CICIoT2023
function addDataset(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "Real Traffic: the CICIoT2023 Dataset",
    notes: NOTES.dataset,
  });
  H.img(s, "feature_selection_funnel", { x: 0.4, y: 1.35, maxW: 5.35, maxH: 3.85, frame: true });
  H.stat(s, { x: 6.15, y: 1.36, w: 3.3, value: N.devices + " devices", label: "physical IoT testbed \u2014 cameras, sensors, speakers, hubs", color: C.blue, valueSize: 23 });
  // Feedback [13]: the deck said "33 attack types" here and "34 labels" two
  // slides later without ever reconciling them. State the arithmetic on-slide.
  H.stat(s, { x: 6.15, y: 2.30, w: 3.3, value: "33 attack types", label: "7 categories: DDoS, DoS, Recon, Web, Brute force, Spoofing, Mirai", color: C.red, valueSize: 23 });
  s.addText("+ benign  =  34 labels in total", {
    x: 6.15, y: 3.16, w: 3.3, h: 0.24, fontFace: F.body, fontSize: 10.5, bold: true, color: C.red, margin: 0,
  });
  H.stat(s, { x: 6.15, y: 3.52, w: 3.3, value: "46 \u2192 29 features", label: "leakage-safe selection, fit on the training partition only", color: C.ink, valueSize: 23 });
  s.addText(N.trainRows + " training rows after reserving held-out classes", {
    x: 6.15, y: 4.6, w: 3.3, h: 0.5, fontFace: F.body, fontSize: 10.5, italic: true, color: C.muted, margin: 0,
  });
}

// Slide 12 — kill-chain projection + splits
function addProjection(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "34 Labels = 33 Attacks + Benign \u2192 5 Stages, 10 Held Out",
    titleSize: 22,
    notes: NOTES.projection,
  });
  H.img(s, "projection_pipeline", { x: 0.4, y: 1.32, maxW: 9.2, maxH: 2.35, frame: true });
  // bottom row: two cards
  H.accentCard(s, 0.55, 3.95, 4.3, 1.2, C.blue);
  s.addText([
    { text: "Deterministic projection \u03c8", options: { bold: true, fontSize: 12, color: C.ink, breakLine: true } },
    { text: "33 attack labels + benign = 34 \u2192 5 stages. Each stage becomes an empirical feature distribution p(x\u2009|\u2009s) of real rows.", options: { fontSize: 10.5, color: C.inkSoft } },
  ], { x: 0.75, y: 4.08, w: 3.95, h: 0.95, fontFace: F.body, margin: 0 });
  H.accentCard(s, 5.15, 3.95, 4.3, 1.2, C.red);
  s.addText([
    { text: "10 classes reserved, never trained on", options: { bold: true, fontSize: 12, color: C.ink, breakLine: true } },
    { text: "\u2265 2 per non-benign stage (e.g. VulnerabilityScan, SqlInjection, DNS_Spoofing, DoS-SYN_Flood) \u2014 the zero-day probe.", options: { fontSize: 10.5, color: C.inkSoft } },
  ], { x: 5.35, y: 4.08, w: 3.95, h: 0.95, fontFace: F.body, margin: 0 });
}

// Slide 13 — stages overlap (PCA)
function addOverlap(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "Adjacent Stages Genuinely Overlap in Feature Space",
    notes: NOTES.overlap,
  });
  H.img(s, "dataset_raw_traffic_a", { x: 0.4, y: 1.35, maxW: 5.15, maxH: 3.7, frame: true, vAlign: "top" });
  // Feedback [14]: "O que e PC? Nao ficou muito claro." Say it on the slide.
  H.card(s, 5.95, 1.5, 3.5, 0.78, { fill: C.cardAlt });
  s.addText([
    { text: "PC1 / PC2 = principal components.  ", options: { bold: true, color: C.ink } },
    { text: "The two directions that carry the most variance \u2014 29 features squeezed onto a readable plane.", options: { color: C.inkSoft } },
  ], { x: 6.1, y: 1.58, w: 3.2, h: 0.62, fontFace: F.body, fontSize: 9.5, margin: 0, valign: "top" });

  H.bullets(s, [
    { text: "Each dot is one real flow, colored by its kill-chain stage", bold: true },
    { text: "IMPACT floods separate; BENIGN \u00b7 RECON \u00b7 ACCESS interleave heavily" },
    { text: "No single flow row reveals the stage \u2014 ambiguity is a property of the data, not an assumption we added" },
  ], { x: 5.95, y: 2.45, w: 3.5, h: 1.9, size: 11, gap: 11 });
  s.addText("Session coherence is imposed at the environment layer (no session key in the dataset) \u2014 declared as a modeling abstraction.", {
    x: 5.95, y: 4.45, w: 3.5, h: 0.7, fontFace: F.body, fontSize: 9.5, italic: true, color: C.muted, margin: 0,
  });
}

// Slide 14 — the reactive attacker (tug of war)
function addAttacker(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "A Reactive Attacker: the Escalation Tug-of-War",
    notes: NOTES.attacker,
  });
  // three regime cards
  const regimes = [
    ["d = 0", "Proportionate", "Attacker pushed back one stage with p = 0.90 (ISOLATE: 0.98)", C.green, "\u25BC push-back"],
    ["d \u2264 \u22121", "Under-force", "Attacker advances with p_up scaled by proximity to IMPACT", C.red, "\u25B2 escalate"],
    ["d \u2265 +1", "Over-force", "Attacker merely holds \u2014 and availability cost is still paid", C.gold, "\u25A0 hold"],
  ];
  let x = 0.55;
  for (const [gap, name, d, color, tag] of regimes) {
    H.card(s, x, 1.42, 2.87, 1.62);
    s.addShape("rect", { x, y: 1.42, w: 2.87, h: 0.06, fill: { color }, line: { type: "none" } });
    s.addText([
      { text: name + "  ", options: { bold: true, fontSize: 13, color: C.ink } },
      { text: gap, options: { fontSize: 11, color, bold: true } },
    ], { x: x + 0.15, y: 1.58, w: 2.6, h: 0.3, fontFace: F.body, margin: 0 });
    s.addText(tag, { x: x + 0.15, y: 1.88, w: 2.6, h: 0.26, fontFace: F.body, fontSize: 10.5, bold: true, color, margin: 0 });
    s.addText(d, { x: x + 0.15, y: 2.18, w: 2.6, h: 0.8, fontFace: F.body, fontSize: 10.5, color: C.inkSoft, margin: 0 });
    x += 3.02;
  }
  s.addText("d  =  chosen action  \u2212  recommended action for the true stage      (the defender never sees that stage \u2014 it must infer it)", {
    x: 0.55, y: 3.2, w: 8.9, h: 0.28, fontFace: F.body, fontSize: 10.5, italic: true, color: C.inkSoft, align: "center", margin: 0,
  });
  // proximity + prevention row
  H.accentCard(s, 0.55, 3.62, 4.3, 1.35, C.red);
  s.addText([
    { text: "Proximity-coupled escalation", options: { bold: true, fontSize: 12, color: C.ink, breakLine: true } },
    { text: "escalation prob. scales with proximity \u03bb = s/4 (floor 0.4) \u2014 the deeper the foothold, the harder it pushes. No fixed intrusion budget.", options: { fontSize: 10.5, color: C.inkSoft } },
  ], { x: 0.75, y: 3.74, w: 3.95, h: 1.12, fontFace: F.body, margin: 0, valign: "top" });
  H.accentCard(s, 5.15, 3.62, 4.3, 1.35, C.green);
  s.addText([
    { text: "Prevention is the goal", options: { bold: true, fontSize: 12, color: C.ink, breakLine: true } },
    { text: "Hold the attacker below IMPACT for all 100 steps \u2192 episode prevented (+50). One good block is not enough \u2014 sustained pressure is.", options: { fontSize: 10.5, color: C.inkSoft } },
  ], { x: 5.35, y: 3.74, w: 3.95, h: 1.12, fontFace: F.body, margin: 0, valign: "top" });
}

// Slide 15 — aliasing + windowed observation
function addAliasing(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "The \u03b1 Dial: Aliasing and the Windowed Observer",
    titleSize: 25,
    notes: NOTES.aliasing,
  });
  H.img(s, "state_machine_aliasing", { x: 0.4, y: 1.38, maxW: 9.2, maxH: 1.75, frame: true, vAlign: "top" });
  const im2 = H.img(s, "obs_tensor_schematic", { x: 0.4, y: 3.55, maxW: 4.5, maxH: 1.52, frame: true, vAlign: "top" });
  H.bullets(s, [
    { text: "Z = (1\u2212\u03b1)\u00b7own-stage  +  \u03b1\u00b7adjacent-stage rows", bold: true },
    { text: "Observation = last w = 5 rows + temporal deltas \u2192 290-dim" },
    { text: "Identical aliased stream for every policy \u2014 no one is privileged" },
  ], { x: 5.25, y: 3.58, w: 4.2, h: 1.1, size: 10.5, gap: 8 });

  // Feedback [13]: "ilustrar o alfa com um exemplo?" — make the dial concrete.
  H.card(s, 5.25, 4.5, 4.2, 0.72, { fill: C.cardAlt });
  s.addText([
    { text: "Concretely, at \u03b1 = " + N.headlineAlpha + ": ", options: { bold: true, color: C.ink } },
    { text: "out of every 10 rows the defender sees, about 4 were emitted by a neighbouring stage. It is watching a scan and being shown a break-in \u2014 and vice versa.", options: { color: C.inkSoft } },
  ], { x: 5.4, y: 4.57, w: 3.9, h: 0.6, fontFace: F.body, fontSize: 9, margin: 0, valign: "top" });
}

// Slide 18 — the two reward functions
function addReward(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "Two Reward Functions: the Sparse One Is Primary",
    titleSize: 24,
    notes: NOTES.reward,
  });
  // outcome card (primary)
  H.accentCard(s, 0.55, 1.45, 4.35, 2.6, C.blue);
  s.addText("OUTCOME  \u00b7  sparse  \u00b7  PRIMARY", { x: 0.78, y: 1.6, w: 3.9, h: 0.28, fontFace: F.body, fontSize: 12.5, bold: true, color: C.blue, margin: 0 });
  H.bullets(s, [
    { text: "Sparse: action cost + terminal accounting only", bold: true },
    { text: "Prevented episode \u2192 +50 \u00b7 terminal defense \u2192 +250" },
    { text: "Compromise \u2192 \u2212200 (\u2212150 more if passive at impact)" },
    { text: "No per-step stage hints \u2014 measures defense, not imitation" },
  ], { x: 0.78, y: 1.98, w: 3.9, h: 1.9, size: 10.5, gap: 8, bulletColor: C.blue });
  // coupled card (ablation)
  H.accentCard(s, 5.15, 1.45, 4.35, 2.6, C.gold);
  s.addText("COUPLED  \u00b7  shaped  \u00b7  ABLATION ONLY", { x: 5.38, y: 1.6, w: 3.9, h: 0.28, fontFace: F.body, fontSize: 12.5, bold: true, color: C.gold, margin: 0 });
  H.bullets(s, [
    { text: "Adds per-step shaping keyed to the true stage", bold: true },
    { text: "+5 proportionality bonus \u00b7 \u22125 disproportion penalty" },
    { text: "Benign guardrails: overreact \u221250 \u00b7 block-on-benign \u2212100" },
    { text: "Rewards exactly what a classifier predicts \u2014 a privileged signal" },
  ], { x: 5.38, y: 1.98, w: 3.9, h: 1.9, size: 10.5, gap: 8, bulletColor: C.gold });

  H.card(s, 0.55, 4.3, 8.95, 0.85, { fill: C.cardAlt });
  s.addText([
    { text: "Design principle:  ", options: { bold: true, color: C.ink } },
    { text: "if the reward hands the agent the stage label, a classifier wins by construction. Train under the sparse OUTCOME reward; keep the shaped COUPLED one only to test that objection.", options: { color: C.inkSoft } },
  ], { x: 0.8, y: 4.42, w: 8.5, h: 0.62, fontFace: F.body, fontSize: 11.5, margin: 0 });
}

// Slide 17 — contenders
function addContenders(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "The Contenders",
    notes: NOTES.contenders,
  });
  H.img(s, "stage_detector_position", { x: 0.4, y: 1.32, maxW: 9.2, maxH: 1.85, frame: true, vAlign: "top" });
  const rows = [
    ["Windowed RL (ours)", "PPO \u00b7 A2C \u00b7 DQN read the 290-dim window; infer the stage implicitly", C.blue],
    ["RF-Acting (deployable)", "Tuned RandomForest (macro-F1 " + N.rfF1 + ") + recommended-action rule \u2014 memoryless per-flow control", C.red],
    ["Trivials", "always-OBSERVE \u00b7 always-BLOCK \u00b7 random \u2014 bracket the reward scale", C.muted],
    ["Oracle (instrument)", "Reads the true stage \u2192 prices perfect perception at " + N.oracle + " \u2014 a ceiling, not a competitor", C.gold],
  ];
  let y = 3.35;
  for (const [t, d, color] of rows) {
    H.accentCard(s, 0.55, y, 8.9, 0.38, color);
    s.addText([
      { text: t + "   ", options: { bold: true, fontSize: 11, color: C.ink } },
      { text: d, options: { fontSize: 10, color: C.inkSoft } },
    ], { x: 0.78, y, w: 8.55, h: 0.38, fontFace: F.body, valign: "middle", margin: 0 });
    y += 0.46;
  }
}

// Slide 18 — evaluation protocol
function addProtocol(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "Evaluation Protocol: Built to Be Believed",
    notes: NOTES.protocol,
  });
  const cards = [
    ["10 \u00d7 5M", "seeds \u00d7 training steps per algorithm \u2014 no early stopping, best-on-validation checkpoint", C.blue],
    ["n = 300", "held-out episodes per policy \u00b7 95% bootstrap CIs \u00b7 separation = disjoint intervals", C.ink],
    ["CPU-only", "commodity workstation \u2014 no GPU anywhere in the pipeline", C.green],
    [N.tests + " tests", "hash-pinned manifests per figure \u00b7 end-to-end re-verification on a fresh checkout", C.gold],
  ];
  let x = 0.55, y = 1.5;
  for (let i = 0; i < cards.length; i++) {
    const [v, d, color] = cards[i];
    H.card(s, x, y, 4.35, 1.28);
    s.addShape("rect", { x, y, w: 4.35, h: 0.06, fill: { color }, line: { type: "none" } });
    s.addText(v, { x: x + 0.2, y: y + 0.16, w: 3.95, h: 0.42, fontFace: F.head, fontSize: 21, bold: true, color, margin: 0, valign: "top" });
    s.addText(d, { x: x + 0.2, y: y + 0.62, w: 3.95, h: 0.58, fontFace: F.body, fontSize: 10.5, color: C.inkSoft, margin: 0, valign: "top" });
    x += 4.55;
    if (i === 1) { x = 0.55; y += 1.48; }
  }
  H.card(s, 0.55, 4.55, 8.95, 0.6, { fill: C.cardAlt });
  s.addText([
    { text: "Same aliased observation stream, same splits, same reward for every contender \u2014 ", options: { color: C.inkSoft } },
    { text: "the only variable is the controller.", options: { bold: true, color: C.ink } },
  ], { x: 0.8, y: 4.63, w: 8.5, h: 0.44, fontFace: F.body, fontSize: 11.5, valign: "middle", margin: 0 });
}

module.exports = {
  addQuestion, addObjectives, addContributions, addArchitecture, addDataset,
  addProjection, addOverlap, addAttacker, addAliasing, addReward,
  addContenders, addProtocol,
};
