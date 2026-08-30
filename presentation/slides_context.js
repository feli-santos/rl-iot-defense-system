"use strict";
const { C, F, N } = require("./theme");
const { NOTES } = require("./notes");
const H = require("./helpers");

// Small numbered badge used to anchor the audience to a region of the slide.
// Feedback [4]: "nao foi possivel perceber onde esta o que esta sendo falado".
// The speaker notes reference these numbers verbally ("bloco um, a esquerda").
function badge(s, x, y, n, color) {
  s.addShape("ellipse", {
    x, y, w: 0.26, h: 0.26, fill: { color }, line: { type: "none" },
  });
  s.addText(String(n), {
    x, y, w: 0.26, h: 0.26, fontFace: F.body, fontSize: 10.5, bold: true,
    color: "FFFFFF", align: "center", valign: "middle", margin: 0,
  });
}

// Slide 4 — what an IoT network actually is (bridge slide, native shapes only)
function addIoTNetwork(pres) {
  const s = H.newContent(pres, {
    kicker: "1 \u00b7 Context",
    title: "What Is an IoT Network?",
    kickerColor: C.green,
    notes: NOTES.iotNetwork,
  });

  s.addText("Everyday objects that sense, decide, and act \u2014 connected through a gateway to cloud services.", {
    x: 0.55, y: 1.24, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 13, italic: true,
    color: C.inkSoft, margin: 0,
  });

  // ---- Zone 1: the "things" ------------------------------------------------
  H.card(s, 0.55, 1.7, 2.95, 2.1, { fill: C.cardAlt });
  s.addShape("rect", { x: 0.55, y: 1.7, w: 2.95, h: 0.06, fill: { color: C.green }, line: { type: "none" } });
  badge(s, 0.63, 1.84, 1, C.green);
  s.addText("Things \u2014 sensors & actuators", {
    x: 0.95, y: 1.82, w: 2.45, h: 0.24, fontFace: F.body, fontSize: 11, bold: true, color: C.ink, margin: 0,
  });
  s.addText("constrained \u00b7 cheap \u00b7 wireless", {
    x: 0.95, y: 2.06, w: 2.45, h: 0.2, fontFace: F.body, fontSize: 8.5, italic: true, color: C.muted, margin: 0,
  });
  const devices = ["Camera", "Motion sensor", "Smart lock", "Thermostat", "Speaker", "Wearable"];
  const chipW = 1.3, chipH = 0.38, gx = 0.14, gy = 0.1;
  devices.forEach((name, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const cx = 0.67 + col * (chipW + gx);
    const cy = 2.34 + row * (chipH + gy);
    s.addShape("roundRect", {
      x: cx, y: cy, w: chipW, h: chipH, rectRadius: 0.05,
      fill: { color: C.card }, line: { color: C.border, width: 0.75 },
    });
    s.addText(name, {
      x: cx, y: cy, w: chipW, h: chipH, fontFace: F.body, fontSize: 9,
      color: C.ink, align: "center", valign: "middle", margin: 0,
    });
  });

  s.addText("Zigbee \u00b7 BLE\nWi-Fi", {
    x: 3.5, y: 2.28, w: 0.8, h: 0.36, fontFace: F.body, fontSize: 7.5,
    color: C.muted, align: "center", margin: 0,
  });
  s.addShape("line", {
    x: 3.5, y: 2.75, w: 0.8, h: 0,
    line: { color: C.inkSoft, width: 1.5, beginArrowType: "triangle", endArrowType: "triangle" },
  });

  // ---- Zone 2: gateway / hub ----------------------------------------------
  s.addShape("roundRect", {
    x: 4.3, y: 2.3, w: 1.85, h: 0.9, rectRadius: 0.07,
    fill: { color: C.card }, line: { color: C.blue, width: 1.25 },
  });
  badge(s, 4.38, 2.36, 2, C.blue);
  s.addText("Gateway / Hub", {
    x: 4.64, y: 2.42, w: 1.45, h: 0.28, fontFace: F.body, fontSize: 11.5, bold: true,
    color: C.blue, align: "center", margin: 0,
  });
  s.addText("aggregates traffic \u00b7 bridges protocols", {
    x: 4.38, y: 2.72, w: 1.69, h: 0.4, fontFace: F.body, fontSize: 8.5,
    color: C.inkSoft, align: "center", margin: 0,
  });

  s.addText("MQTT \u00b7 HTTP", {
    x: 6.11, y: 2.46, w: 0.84, h: 0.2, fontFace: F.body, fontSize: 7.5,
    color: C.muted, align: "center", margin: 0,
  });
  s.addShape("line", {
    x: 6.15, y: 2.75, w: 0.8, h: 0,
    line: { color: C.inkSoft, width: 1.5, beginArrowType: "triangle", endArrowType: "triangle" },
  });

  // ---- Zone 3: cloud -------------------------------------------------------
  s.addShape("cloud", {
    x: 6.95, y: 1.9, w: 2.5, h: 1.65,
    fill: { color: C.card }, line: { color: C.blueSoft, width: 1 },
  });
  badge(s, 7.05, 2.14, 3, C.blueSoft);
  s.addText("Cloud services", {
    x: 7.05, y: 2.44, w: 2.3, h: 0.28, fontFace: F.body, fontSize: 12, bold: true,
    color: C.ink, align: "center", margin: 0,
  });
  s.addText("analytics \u00b7 dashboards \u00b7 remote control", {
    x: 7.1, y: 2.72, w: 2.2, h: 0.36, fontFace: F.body, fontSize: 8.5,
    color: C.inkSoft, align: "center", margin: 0,
  });

  s.addText("telemetry flows up \u00b7 commands flow down \u2014 every hop is network traffic", {
    x: 3.6, y: 3.6, w: 5.85, h: 0.24, fontFace: F.body, fontSize: 9.5, italic: true,
    color: C.muted, align: "center", margin: 0,
  });

  // ---- three defining characteristics --------------------------------------
  const traits = [
    ["Sense \u2192 decide \u2192 act", "The physical world sits inside the control loop \u2014 a hijacked device has physical consequences.", C.green],
    ["MCU-class hardware", "Kilobytes of RAM, battery budgets \u2014 no room for antivirus or endpoint agents.", C.gold],
    ["Heterogeneous, always on", "Dozens of vendors and protocols \u2014 and every one of them has a path to the Internet.", C.blue],
  ];
  let tx = 0.55;
  traits.forEach(([t, d, color], i) => {
    H.card(s, tx, 3.95, 2.86, 0.94);
    s.addShape("rect", { x: tx, y: 3.95, w: 2.86, h: 0.055, fill: { color }, line: { type: "none" } });
    badge(s, tx + 0.13, 4.08, i + 4, color);
    s.addText(t, { x: tx + 0.45, y: 4.06, w: 2.28, h: 0.26, fontFace: F.body, fontSize: 11.5, bold: true, color: C.ink, margin: 0, valign: "top" });
    s.addText(d, { x: tx + 0.15, y: 4.36, w: 2.58, h: 0.52, fontFace: F.body, fontSize: 9, color: C.inkSoft, margin: 0, valign: "top" });
    tx += 3.02;
  });

  // Feedback [4c]: answer "como a defesa deve acontecer?" before leaving.
  s.addText([
    { text: "\u21d2  Defense must therefore live at the gateway \u2014 ", options: { bold: true, color: C.ink } },
    { text: "the one place that sees every flow. That is where this work acts.", options: { color: C.inkSoft } },
  ], { x: 0.55, y: 4.99, w: 8.9, h: 0.28, fontFace: F.body, fontSize: 11.5, margin: 0 });
}

// Slide 5 — why IoT security is hard
function addWhyIoT(pres) {
  const s = H.newContent(pres, {
    kicker: "1 \u00b7 Context",
    title: "IoT Turned Every Network Into an Attack Surface",
    kickerColor: C.green,
    notes: NOTES.whyIoT,
  });
  // stat callouts row
  H.stat(s, { x: 0.55, y: 1.42, w: 2.8, value: "19.8B \u2192 40.6B", label: "IoT devices worldwide, 2025 \u2192 2034 (forecast)", color: C.blue, valueSize: 26 });
  H.stat(s, { x: 3.6, y: 1.42, w: 2.6, value: "USD 12.2T", label: "projected annual cybercrime damages by 2031", color: C.red, valueSize: 26 });
  H.stat(s, { x: 6.5, y: 1.42, w: 3.0, value: "33 attacks", label: "attack types executed device-to-device in CICIoT2023", color: C.ink, valueSize: 26 });

  s.addShape("line", { x: 0.55, y: 2.62, w: 8.9, h: 0, line: { color: C.border, width: 1 } });

  s.addText("Why traditional security fails here", {
    x: 0.55, y: 2.78, w: 6.0, h: 0.3, fontFace: F.body, fontSize: 12, bold: true, color: C.inkSoft, margin: 0,
  });
  const fails = [
    ["Constrained devices", "No compute, memory, or energy budget for on-device security agents"],
    ["Extreme heterogeneity", "Disparate protocols and OSes \u2014 uniform policy enforcement is impractical"],
    ["Static defenses", "Signature IDS misses zero-days; anomaly IDS floods operators with false positives"],
  ];
  let x = 0.55;
  for (const [t, d] of fails) {
    H.card(s, x, 3.18, 2.86, 1.5);
    s.addShape("rect", { x, y: 3.18, w: 2.86, h: 0.06, fill: { color: C.red }, line: { type: "none" } });
    s.addText(t, { x: x + 0.15, y: 3.34, w: 2.56, h: 0.28, fontFace: F.body, fontSize: 12.5, bold: true, color: C.ink, margin: 0, valign: "top" });
    s.addText(d, { x: x + 0.15, y: 3.68, w: 2.56, h: 0.92, fontFace: F.body, fontSize: 10.5, color: C.inkSoft, margin: 0, valign: "top" });
    x += 3.02;
  }
  s.addText([
    { text: "\u21d2  The gap: ", options: { bold: true, color: C.ink } },
    { text: "defenses that adapt, learn from data, and act autonomously \u2014 the promise of Deep RL.", options: { color: C.inkSoft } },
  ], { x: 0.55, y: 4.86, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 12.5, margin: 0 });
}

// Slide 5 — the cyber kill chain
function addKillChain(pres) {
  const s = H.newContent(pres, {
    kicker: "1 \u00b7 Context",
    title: "The Cyber Kill Chain: a Decision Frame for Defense",
    kickerColor: C.green,
    notes: NOTES.killChain,
  });
  s.addText("An intrusion is a campaign in stages \u2014 and each stage has a proportionate response.", {
    x: 0.55, y: 1.28, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 13, italic: true, color: C.inkSoft, margin: 0,
  });

  // stage chips with descriptions
  const desc = [
    "Routine traffic. Business as usual.",
    "Scanning, fingerprinting, enumeration.",
    "Foothold: brute force, injection, hijack.",
    "Lateral movement, botnet staging.",
    "DDoS, exfiltration, detonation.",
  ];
  const gapW = 0.18; const chipW = (8.9 - 4 * gapW) / 5;
  H.kcStrip(s, 0.55, 1.75, 8.9, { labels: "stage", chipH: 0.5, fontSize: 11 });
  for (let i = 0; i < 5; i++) {
    const cx = 0.55 + i * (chipW + gapW);
    s.addText(desc[i], {
      x: cx, y: 2.32, w: chipW, h: 0.75, fontFace: F.body, fontSize: 9.5,
      color: C.inkSoft, align: "center", valign: "top", margin: 0,
    });
  }

  // defender mirror row
  s.addText("Defender's five-action menu (recommended-action mapping):", {
    x: 0.55, y: 3.18, w: 8.9, h: 0.28, fontFace: F.body, fontSize: 12, bold: true, color: C.blue, margin: 0,
  });
  const aDesc = ["monitor only", "record & watch", "throttle traffic", "drop flows", "quarantine segment"];
  for (let i = 0; i < 5; i++) {
    const cx = 0.55 + i * (chipW + gapW);
    s.addShape("roundRect", {
      x: cx, y: 3.52, w: chipW, h: 0.44, rectRadius: 0.05,
      fill: { color: C.card }, line: { color: C.blue, width: 1.25 },
    });
    s.addText(H.KC_ACTIONS[i], {
      x: cx, y: 3.52, w: chipW, h: 0.44, fontFace: F.body, fontSize: 10.5, bold: true,
      color: C.blue, align: "center", valign: "middle", margin: 0,
    });
    s.addText(aDesc[i], {
      x: cx, y: 4.0, w: chipW, h: 0.24, fontFace: F.body, fontSize: 9,
      color: C.muted, align: "center", margin: 0,
    });
  }

  // cost gradient annotation
  H.arrow(s, 0.55, 4.62, 8.9, { color: C.muted, width: 1.25 });
  s.addText("cost of interruption grows monotonically \u2192   \u00b7   over-forcing early = availability damage   \u00b7   under-forcing late = compromise", {
    x: 0.55, y: 4.74, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 10.5, color: C.inkSoft, align: "center", margin: 0,
  });
}

// Slide 7 — RL primer: what it is, WHY it fits this problem, and which three.
// Feedback [4d] "E caracteristicas do RL? Por que usar?" and [7a] "por que os
// modelos de RL foram escolhidos?" — both were only ever answered verbally.
function addRLPrimer(pres) {
  const s = H.newContent(pres, {
    kicker: "1 \u00b7 Context",
    title: "Reinforcement Learning \u2014 and Why It Fits This Problem",
    kickerColor: C.green,
    notes: NOTES.rlPrimer,
  });

  H.img(s, "rl_agent_loop", { x: 0.55, y: 1.34, maxW: 4.3, maxH: 1.72, frame: true, vAlign: "top" });
  s.addText("Learn a policy \u03c0(a\u2009|\u2009s) that maximizes expected cumulative reward \u2014 by trial and error, not labels.", {
    x: 0.55, y: 3.18, w: 4.3, h: 0.44, fontFace: F.body, fontSize: 10.5, italic: true,
    color: C.inkSoft, align: "center", margin: 0,
  });

  // --- WHY reinforcement learning, and not supervised learning --------------
  H.accentCard(s, 0.55, 3.62, 4.3, 1.32, C.green);
  s.addText("Why RL here?", {
    x: 0.75, y: 3.70, w: 3.9, h: 0.24, fontFace: F.body, fontSize: 11, bold: true, color: C.green, margin: 0,
  });
  H.bullets(s, [
    { text: "The response changes the threat \u2014 it is control, not labelling" },
    { text: "Success is an episode outcome, not a per-flow verdict" },
    { text: "No ground-truth \u201ccorrect action\u201d exists to supervise on" },
  ], { x: 0.75, y: 3.95, w: 3.95, h: 0.94, size: 9, gap: 3, bulletColor: C.green });

  // --- WHICH algorithms, and why exactly these three -----------------------
  const algos = [
    ["DQN", "off-policy \u00b7 value-based", "Reuses past experience from a replay buffer \u2014 sample-efficient, but the bootstrap can wobble.", C.red],
    ["PPO", "on-policy \u00b7 actor-critic", "Fresh rollouts with clipped, conservative updates. The de-facto stability standard.", C.blue],
    ["A2C", "on-policy \u00b7 actor-critic", "Synchronous advantage actor-critic \u2014 simpler and faster per update.", C.blue],
  ];
  let y = 1.34;
  for (const [name, fam, d, color] of algos) {
    H.accentCard(s, 5.15, y, 4.3, 0.86, color);
    s.addText([
      { text: name + "  ", options: { bold: true, fontSize: 13.5, color: C.ink } },
      { text: fam, options: { fontSize: 9, italic: true, color } },
    ], { x: 5.33, y: y + 0.07, w: 4.0, h: 0.26, fontFace: F.body, margin: 0 });
    s.addText(d, { x: 5.33, y: y + 0.34, w: 4.0, h: 0.46, fontFace: F.body, fontSize: 9.5, color: C.inkSoft, margin: 0 });
    y += 0.96;
  }

  H.card(s, 5.15, 4.22, 4.3, 0.9, { fill: C.cardAlt });
  s.addText("Why these three?", {
    x: 5.33, y: 4.29, w: 4.0, h: 0.22, fontFace: F.body, fontSize: 10.5, bold: true, color: C.ink, margin: 0,
  });
  s.addText("They span both model-free families for discrete actions, and are the SB3-recommended set for DRL-for-IDS. SAC / TD3 / Dreamer excluded: a 5-action discrete menu does not need them.", {
    x: 5.33, y: 4.52, w: 4.0, h: 0.56, fontFace: F.body, fontSize: 8.5, color: C.inkSoft, margin: 0, valign: "top",
  });

  // Feedback [7b]: "nem sempre chegam os mesmos dados; DQN usa replay buffer".
  // The honest control is the interaction budget, not the number of gradient
  // samples — replay reuse IS part of the learning rule under test.
  s.addText([
    { text: "Controlled: ", options: { bold: true, color: C.ink } },
    { text: "same 2\u00d764 MLP, same 5M environment-step budget, same 10 seeds. How each reuses that experience (DQN replays it, PPO/A2C do not) is precisely the learning rule under test.", options: { color: C.inkSoft } },
  ], { x: 0.55, y: 5.02, w: 8.3, h: 0.26, fontFace: F.body, fontSize: 8.5, margin: 0 });
}

// Slide 7 — POMDP: the perception gap
function addPOMDP(pres) {
  const s = H.newContent(pres, {
    kicker: "1 \u00b7 Context",
    title: "The Perception Gap: From MDP to POMDP",
    kickerColor: C.green,
    notes: NOTES.pomdp,
  });

  // left: MDP box
  H.card(s, 0.55, 1.55, 4.25, 1.5);
  s.addText("Full observability (MDP)", { x: 0.75, y: 1.68, w: 3.9, h: 0.28, fontFace: F.body, fontSize: 12.5, bold: true, color: C.ink, margin: 0 });
  s.addText([
    { text: "state ", options: { color: C.inkSoft } },
    { text: "s\u209c", options: { bold: true, color: C.green } },
    { text: " is visible \u2192 react to what you ", options: { color: C.inkSoft } },
    { text: "see", options: { italic: true, color: C.inkSoft } },
  ], { x: 0.75, y: 2.0, w: 3.9, h: 0.3, fontFace: F.body, fontSize: 11.5, margin: 0 });
  s.addText("A memoryless rule per state can be optimal \u2014 classification is enough.", {
    x: 0.75, y: 2.36, w: 3.9, h: 0.55, fontFace: F.body, fontSize: 10.5, color: C.muted, margin: 0,
  });

  // right: POMDP box
  H.accentCard(s, 5.2, 1.55, 4.25, 1.5, C.red);
  s.addText("Partial observability (POMDP) \u2014 this work", { x: 5.4, y: 1.68, w: 3.9, h: 0.28, fontFace: F.body, fontSize: 12.5, bold: true, color: C.ink, margin: 0 });
  s.addText([
    { text: "stage ", options: { color: C.inkSoft } },
    { text: "s\u209c", options: { bold: true, color: C.red } },
    { text: " is hidden \u2192 only feature rows ", options: { color: C.inkSoft } },
    { text: "o\u209c \u223c Z(\u00b7\u2009|\u2009s\u209c)", options: { bold: true, color: C.ink } },
  ], { x: 5.4, y: 2.0, w: 3.9, h: 0.3, fontFace: F.body, fontSize: 11.5, margin: 0 });
  s.addText("Adjacent stages emit overlapping observations \u2014 no single row identifies the stage.", {
    x: 5.4, y: 2.36, w: 3.9, h: 0.55, fontFace: F.body, fontSize: 10.5, color: C.muted, margin: 0,
  });

  // belief pipeline diagram (native shapes)
  const py = 3.55;
  s.addText("The only way through: accumulate evidence over time", {
    x: 0.55, y: 3.18, w: 8.9, h: 0.28, fontFace: F.body, fontSize: 12, bold: true, color: C.blue, margin: 0,
  });
  const steps = [
    ["o\u209c\u208b\u2084 \u2026 o\u209c", "window of recent\nobservations (w = 5)"],
    ["belief b\u209c(s)", "implicit posterior over\nhidden stages"],
    ["\u03c0(a\u2009|\u2009history)", "policy conditioned\non the window"],
    ["a\u209c", "proportionate\ndefensive action"],
  ];
  let x = 0.55;
  const bw = 1.95, gap = (8.9 - 4 * bw) / 3;
  for (let i = 0; i < steps.length; i++) {
    H.card(s, x, py, bw, 1.05, { fill: i === 3 ? C.blue : C.card });
    s.addText(steps[i][0], {
      x, y: py + 0.1, w: bw, h: 0.34, fontFace: F.head, fontSize: 13, bold: true,
      color: i === 3 ? "FFFFFF" : C.ink, align: "center", margin: 0,
    });
    s.addText(steps[i][1], {
      x: x + 0.06, y: py + 0.47, w: bw - 0.12, h: 0.52, fontFace: F.body, fontSize: 9,
      color: i === 3 ? "E8EDF2" : C.muted, align: "center", margin: 0,
    });
    if (i < 3) H.arrow(s, x + bw + 0.06, py + 0.52, gap - 0.12, { color: C.inkSoft });
    x += bw + gap;
  }
  s.addText("If one snapshot cannot diagnose, memory becomes the defense. This is where sequential control earns its keep.", {
    x: 0.55, y: 4.85, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 11.5, italic: true, color: C.inkSoft, align: "center", margin: 0,
  });
}

module.exports = { addIoTNetwork, addWhyIoT, addKillChain, addRLPrimer, addPOMDP };
