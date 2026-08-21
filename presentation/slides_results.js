"use strict";
const { C, F, N } = require("./theme");
const H = require("./helpers");

// Slide 19 — learning curves + reliability
function addLearning(pres) {
  const s = H.newContent(pres, {
    kicker: "4 \u00b7 Results",
    title: "Sparse-Reward Training: a Reliability Story",
    titleSize: 25,
    notes:
      "First result. Learning curves over 5M steps, 10 seeds, sparse outcome reward, headline alpha 0.4. Dashed grey band = RF-Acting " +
      "reference; dash-dot black = oracle ceiling. Read: both ON-POLICY algorithms climb out of the negative regime and plateau \u2014 " +
      "best-checkpoint PPO +121.3 (sd ~15), A2C +138.7 (sd ~9, the tightest). OFF-POLICY DQN destabilizes without per-step shaping: " +
      "best-checkpoint +72.5, across-seed sd ~52, seeds ranging roughly \u221215 to +132 \u2014 replay does not bootstrap credit across a " +
      "100-step sparse episode. Finding: the on-policy advantage is TRAINING RELIABILITY, not peak return. We report all three \u2014 " +
      "including the unstable one.",
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
    notes:
      "My favorite teaching slide: the algorithms did not just score differently \u2014 they learned different DEFENSIVE PHILOSOPHIES. " +
      "Per-stage action distributions (rows: A2C, PPO, DQN; columns: BENIGN\u2192IMPACT). A2C: prevent-at-maneuver \u2014 blocks 84.4% of " +
      "MANEUVER steps, suppresses the mid-chain advance, never isolates. PPO: contain-at-impact \u2014 tolerates deeper penetration, then " +
      "acts at the final stage. Both benign-safe on the left column (aggressive action on <1% of benign flows). Neither doctrine was " +
      "programmed \u2014 both emerged from the same sparse reward. This interpretability matters for a security operator: strategy, not " +
      "black-box scores.",
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
    notes:
      "THE slide of the defense \u2014 take your time. X-axis: aliasing rate alpha (ambiguity dial). Y: mean episodic reward on the held-out " +
      "split, 95% bootstrap CIs. Walk it in order. (1) At \u03b1=0, honesty anchor: PPO +138.6 vs RF +136.5 \u2014 STATISTICAL TIE, overlapping " +
      "CIs: the environment does NOT favor RL by construction; where the task is per-flow classification, the classifier ties. " +
      "(2) As \u03b1 rises, RF-Acting degrades MONOTONICALLY: +113.2, +94.4, +64.0, +20.5 \u2026 \u221229.3 at \u03b1=1 \u2014 NET-HARMFUL, worse than doing " +
      "nothing's baseline of always-block at ~0. (3) Windowed PPO stays FLAT \u2014 no monotonic trend; the window absorbs the ambiguity. " +
      "(4) From \u03b1=0.4 (headline): DISJOINT CIs \u2014 gap +26.9, widening to +161.2 at \u03b1=1. Oracle flat at +194.8 prices perfect " +
      "perception. One sentence: when observation gets ambiguous, memoryless classification collapses; windowed control does not.",
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
    notes:
      "Answering the privileged-reward objection head-on. Retrain EVERYTHING under both contracts; score RF under the same contract " +
      "each time. Coupled: best learned agent is DQN +226.2 \u2014 thrives on dense shaping \u2014 RF-minus-best-RL = \u221263.1 (RL leads). " +
      "Outcome: best learned agent is A2C +146.1 \u2014 gap \u221263.0, virtually identical. Two lessons: (1) the RL-vs-RF separation does NOT " +
      "depend on the privileged shaping \u2014 objection closed; (2) the reward contract changes WHICH algorithm wins: DQN's coupled win " +
      "evaporates under sparsity (\u22128.6 pooled mean, variance triples) \u2014 the same replay machinery that exploits dense signals fails " +
      "sparse credit assignment. Off-policy value bootstrapping needs shaping; on-policy tolerates its absence.",
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
    notes:
      "Two off-distribution stress tests on FIXED policies \u2014 trained once, never retrained per condition. Left (difficulty sweep): " +
      "lower p_down = proportionate actions less likely to push the attacker back; monotone response, no rank inversions; A2C leads at " +
      "EVERY difficulty (\u22127.2 at the harshest \u2192 +147.3 at the easiest); the prevent-at-maneuver doctrine loses the least ground " +
      "exactly where it is harshest. DQN's fragility reappears as the widest bands. Right (evasion sweep): attacker hardens against " +
      "eviction after sensing force (post-detection hardening, up to 0.75). Pre-registered criterion: lower CI bound must not fall " +
      "more than 25% of evasion-free mean. A2C +142.6 \u2192 +112.7, compromise 0.23 \u2192 0.41 \u2014 CLEARS it; PPO narrowly misses; DQN " +
      "within band but from a low base. Graceful degradation, not collapse \u2014 and a real A2C-over-PPO separation.",
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
    notes:
      "The zero-day-like probe. The 10 reserved classes are injected at their true stage into an otherwise in-distribution episode " +
      "(single-stage feature injection) at headline alpha. Metric: PREVENTION RATE \u2014 attacker held below IMPACT for the whole episode. " +
      "Grid: rows = policies, columns = classes. Best windowed RL (A2C) prevents 0.71\u20130.85 on EVERY class; RF-Acting 0.00\u20130.15. " +
      "Advantage +0.70 to +0.78 per class. Note honestly: always-block 'prevents' 1.0 everywhere \u2014 by quarantining 100% of benign " +
      "traffic; reward 0.0; operationally inadmissible. The meaningful frontier is prevention WHILE benign-safe (<1% FPR) \u2014 only the " +
      "learned agents live there. Also honest: moderate absolute rates \u2014 the oracle ceiling is far above; the claim is the relative " +
      "advantage, not near-perfect security.",
  });
  H.img(s, "F15_ood_robustness", { x: 0.35, y: 1.32, maxW: 4.9, maxH: 3.72, frame: true, vAlign: "top" });
  H.stat(s, { x: 5.75, y: 1.45, w: 3.7, value: N.oodRLlo + " \u2013 " + N.oodRLhi, label: "best windowed RL (A2C): prevention on every held-out class", color: C.blue, valueSize: 27 });
  H.stat(s, { x: 5.75, y: 2.55, w: 3.7, value: N.oodRFlo + " \u2013 " + N.oodRFhi, label: "RF-Acting: prevention on the same classes", color: C.red, valueSize: 27 });
  H.stat(s, { x: 5.75, y: 3.65, w: 3.7, value: N.oodAdvLo + " to " + N.oodAdvHi, label: "advantage on every single class \u2014 none negative", color: C.green, valueSize: 27 });
  s.addText("(always-BLOCK reaches 1.0 only by disrupting 100% of benign traffic \u2014 inadmissible; learned agents stay under 1%.)", {
    x: 5.75, y: 4.72, w: 3.7, h: 0.5, fontFace: F.body, fontSize: 9.5, italic: true, color: C.muted, margin: 0,
  });
}

// Slide 25 — recall independence + mechanism
function addRecallIndependence(pres) {
  const s = H.newContent(pres, {
    kicker: "4 \u00b7 Results",
    title: "No Dependence on Detector Recall",
    notes:
      "Why the OOD result matters scientifically \u2014 the last version of the skeptic's objection: 'RL only wins where the detector is " +
      "blind.' If true, advantage should SHRINK as detector recall rises. Scatter: per-class RF recall (x) vs RL prevention advantage " +
      "(y). The held-out classes span recall 0.20\u20130.998 by construction \u2014 near-blind to near-perfect. Result: NO detectable trend. " +
      "Spearman rho 0.22 (p=0.54), Pearson r \u22120.02 (p=0.95), bootstrap OLS slope CI [\u22120.08, +0.04] spans zero. Honest caveat: n=10 \u2014 " +
      "absence of a detectable trend, not proof of independence; but the trend the objection REQUIRES (negative) is absent. Mechanism " +
      "(structural, not perceptual): prevention needs SUSTAINED proportionate pressure; RF \u2014 even when it classifies correctly \u2014 acts " +
      "passively ~2/3 of steps, lets the attacker reach IMPACT, then blocks late: 'mitigated', never 'prevented'. A one-shot classifier " +
      "structurally cannot express temporal control.",
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
    notes:
      "Owning the boundaries pre-empts half the committee's questions. (1) Session coherence + aliasing are modeling abstractions \u2014 " +
      "the dataset has no session key; alpha is a controlled knob, and the contribution is the SHAPE of the response, anchored by the " +
      "alpha=0 tie. (2) The advantage is conditional on partial observability \u2014 at alpha=0 we claim a tie, nothing more. (3) The " +
      "attacker is designed, reactive, NOT co-trained; the crossover is conditional on this attacker class; evasive-persistence is the " +
      "closest tested relaxation; self-play is future work. (4) RF-Acting is memoryless by construction \u2014 the benchmark isolates " +
      "windowed vs memoryless CONTROL; a windowed supervised baseline is the most direct strengthening and is named follow-up work. " +
      "(5) Single dataset \u2014 replication on Bot-IoT is the natural next step. (6) OOD study is a controlled feature-injection stress " +
      "test at n=10 \u2014 not deployed zero-day evidence.",
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
    notes:
      "The three findings, one breath each. F1: windowed temporal control beats memoryless per-flow control PRECISELY under partial " +
      "observability \u2014 tie at alpha=0, RF monotone collapse to net-harmful, PPO flat, disjoint CIs from alpha=0.4. F2: the advantage " +
      "is not a reward artifact \u2014 best RL leads under both contracts (\u221263.1 / \u221263.0); shaping changes which algorithm wins. F3: on ten " +
      "held-out classes, prevention advantage on every class, no detectable recall dependence \u2014 temporal control, not detector blind " +
      "spots. Plus the operational bonus: benign-safe (<1% FPR) and a 90 KB / 23K-param policy vs 181 MB tuned RF \u2014 edge-deployable. " +
      "Closing line: the contribution is not 'RL wins' \u2014 it is a controlled account of WHEN and WHY it wins, reproducible end-to-end.",
  });
  const rows = [
    ["1", "The crossover is real and controlled", "Tie at \u03b1 = 0 \u2192 disjoint CIs from \u03b1 = " + N.headlineAlpha + " \u2192 RF net-harmful at \u03b1 = 1 while windowed PPO holds flat", C.red],
    ["2", "It is not a reward artifact", "Best learned agent leads under both reward contracts (" + N.cplGap + " coupled, " + N.outGap + " sparse)", C.gold],
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
  H.card(s, 0.55, 4.5, 8.9, 0.72, { fill: C.cardAlt });
  s.addText([
    { text: "Operationally deployable: ", options: { bold: true, color: C.ink } },
    { text: "benign disruption < 1% \u00b7 policy footprint " + N.policyKB + " KB / " + N.policyParams + " params (vs " + N.rfMB + " MB tuned RF) \u00b7 fully reproducible (" + N.tests + " tests, hash-pinned figures)", options: { color: C.inkSoft } },
  ], { x: 0.8, y: 4.58, w: 8.5, h: 0.56, fontFace: F.body, fontSize: 11.5, valign: "middle", margin: 0 });
}

// Slide 28 — future work + publication
function addFuture(pres) {
  const s = H.newContent(pres, {
    kicker: "5 \u00b7 Closing",
    title: "Where This Goes Next",
    kickerColor: C.ink,
    notes:
      "Grouped future work. Nearest to the thesis: windowed supervised baseline (the most direct strengthening); recurrent belief-state " +
      "policies \u2014 a preliminary recurrent trial did NOT beat the window under this budget, worth characterizing; second dataset " +
      "(Bot-IoT). Threat model: co-adaptive self-play attacker; multi-agent defense (RESTRAIN-style coordination). Deployment: edge " +
      "hardware quantification (the 90 KB policy invites it); federated multi-site training; constrained-MDP false-positive guarantees. " +
      "Dissemination: a condensed journal article of this work is submitted to Elsevier Internet of Things (2026). Code, manifests, and " +
      "the full artifact chain are public on GitHub.",
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
    notes:
      "Thank the committee by name; thank the audience. Then: 'I am at your disposal for questions.' Backup slides follow the thank-you " +
      "slide: reward constants, hyperparameters, kill-chain mapping, detector performance, benign safety, reproducibility chain.",
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
