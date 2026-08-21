"use strict";
const { C, F, N } = require("./theme");
const H = require("./helpers");

// Slide 4 — why IoT security is hard
function addWhyIoT(pres) {
  const s = H.newContent(pres, {
    kicker: "1 \u00b7 Context",
    title: "IoT Turned Every Network Into an Attack Surface",
    kickerColor: C.green,
    notes:
      "Teach: scale first. 19.8 billion devices in 2025, forecast to more than double to 40.6 billion by 2034 (Statista). Cybercrime damages " +
      "projected at USD 12.2 trillion annually by 2031 (Cybersecurity Ventures / Morgan). Then the three structural reasons traditional " +
      "IT security fails on IoT: (1) devices are resource-constrained \u2014 no room for antivirus/firewall agents; (2) extreme heterogeneity " +
      "\u2014 no uniform policy enforcement; (3) signature IDS is reactive \u2014 zero-days have no signature, and anomaly IDS drowns in false " +
      "positives on heterogeneous traffic. Land the conclusion: we need defense that is adaptive, data-driven, autonomous.",
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
    notes:
      "This is the vocabulary slide \u2014 teach it well, everything later hangs on it. An intrusion is not one event; it is a campaign in " +
      "stages. Walk the five stages with the story: attacker scans (RECON), gets a foothold (ACCESS), moves laterally / positions " +
      "(MANEUVER), and detonates \u2014 DDoS, exfiltration (IMPACT). Two teaching points: (1) stopping an attack EARLY is cheap, stopping it " +
      "LATE is expensive; (2) each stage has a proportionate response \u2014 the recommended-action mapping (IoTWarden). Over-reacting has a " +
      "real availability cost: you cannot isolate the network on every port scan. This stage-to-action alignment is what our reward and " +
      "our attacker dynamics are built on.",
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

// Slide 6 — RL in one slide
function addRLPrimer(pres) {
  const s = H.newContent(pres, {
    kicker: "1 \u00b7 Context",
    title: "Reinforcement Learning in One Slide",
    kickerColor: C.green,
    notes:
      "Keep this brisk \u2014 the committee knows RL; this is calibration, not a lecture. Agent observes state, picks action, environment " +
      "returns reward and next state; the agent learns a policy maximizing cumulative discounted reward. No labels \u2014 learning from " +
      "interaction; credit assignment over time is the hard part. Three algorithms, spanning the two model-free families: DQN " +
      "(off-policy, value-based, replay buffer), PPO and A2C (on-policy actor-critic; PPO adds clipped updates). Foreshadow: the " +
      "on-policy/off-policy distinction becomes an empirical finding later \u2014 remember it.",
  });
  const im = H.img(s, "rl_agent_loop", { x: 0.55, y: 1.52, maxW: 4.6, maxH: 2.6, frame: true, vAlign: "top" });
  s.addText("Learn a policy \u03c0(a\u2009|\u2009s) that maximizes expected cumulative reward \u2014 by trial and error, not labels.", {
    x: 0.55, y: 4.35, w: 4.6, h: 0.6, fontFace: F.body, fontSize: 11, italic: true, color: C.inkSoft, align: "center", margin: 0,
  });

  const algos = [
    ["DQN", "off-policy \u00b7 value-based", "Learns Q-values; replays past experience from a buffer. Sample-efficient \u2014 but bootstrap can wobble.", C.red],
    ["PPO", "on-policy \u00b7 actor-critic", "Learns from fresh rollouts with clipped, conservative policy updates. The de-facto stability standard.", C.blue],
    ["A2C", "on-policy \u00b7 actor-critic", "Synchronous advantage actor-critic \u2014 simpler, faster updates from parallel workers.", C.blue],
  ];
  let y = 1.43;
  for (const [name, fam, d, color] of algos) {
    H.accentCard(s, 5.5, y, 3.95, 0.92, color);
    s.addText([
      { text: name + "  ", options: { bold: true, fontSize: 14, color: C.ink } },
      { text: fam, options: { fontSize: 9.5, italic: true, color } },
    ], { x: 5.68, y: y + 0.08, w: 3.6, h: 0.28, fontFace: F.body, margin: 0 });
    s.addText(d, { x: 5.68, y: y + 0.38, w: 3.62, h: 0.5, fontFace: F.body, fontSize: 10, color: C.inkSoft, margin: 0 });
    y += 1.04;
  }
  s.addText("Same MLP policy network (2\u00d764 units), same budget, same seeds \u2014 the comparison isolates the learning rule.", {
    x: 5.5, y: 4.72, w: 3.95, h: 0.55, fontFace: F.body, fontSize: 10, color: C.muted, margin: 0 },
  );
}

// Slide 7 — POMDP: the perception gap
function addPOMDP(pres) {
  const s = H.newContent(pres, {
    kicker: "1 \u00b7 Context",
    title: "The Perception Gap: From MDP to POMDP",
    kickerColor: C.green,
    notes:
      "THE central concept of the thesis \u2014 spend time here. In an MDP the agent sees the true state. Our defender never does: the true " +
      "kill-chain stage is latent. It sees only traffic features \u2014 and adjacent stages EMIT OVERLAPPING FEATURES. That is partial " +
      "observability (POMDP): observations are drawn from an observation kernel Z, and the optimal policy must act on a BELIEF built " +
      "from history, not on a single observation. Teaching metaphor: a doctor who never sees the disease, only symptoms \u2014 and adjacent " +
      "diseases share symptoms; one snapshot cannot diagnose, a case history can. Our agent approximates belief with a sliding window " +
      "of the last 5 observations. Foreshadow: we make ambiguity a CONTROLLED DIAL (aliasing rate alpha) and study defense as a " +
      "function of it.",
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

module.exports = { addWhyIoT, addKillChain, addRLPrimer, addPOMDP };
