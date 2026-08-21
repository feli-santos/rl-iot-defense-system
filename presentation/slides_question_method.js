"use strict";
const { C, F, N } = require("./theme");
const H = require("./helpers");

// Slide 8 — the central question
function addQuestion(pres) {
  const s = H.newDark(pres, {
    notes:
      "Slow down; this is the thesis in one slide. The naive pitch \u2014 'RL beats a classifier' \u2014 is not a scientific claim until you ask " +
      "WHEN and WHY. If every flow reveals the attacker's intent, defense collapses to per-flow classification, and a well-tuned " +
      "classifier is the right tool. RL earns its keep only when the problem is genuinely sequential AND partially observed. So the " +
      "dissertation poses the skeptic's objection to itself: 'your agent is solving a disguised classification problem' \u2014 and designs " +
      "experiments to answer it. Everything that follows is that answer.",
  });
  s.addText("2 \u00b7 RESEARCH QUESTION", {
    x: 0.75, y: 0.6, w: 5, h: 0.3, fontFace: F.body, fontSize: 11, bold: true, charSpacing: 3, color: C.gold, margin: 0,
  });
  s.addText("When does a learned sequential policy\ngenuinely beat a memoryless\nper-flow classifier?", {
    x: 0.75, y: 1.18, w: 8.5, h: 1.7, fontFace: F.head, fontSize: 29, bold: true, color: C.textOnDark, margin: 0, lineSpacingMultiple: 1.08,
  });
  // the objection card
  s.addShape("rect", { x: 0.75, y: 3.0, w: 8.5, h: 1.05, fill: { color: "2E2E33" }, line: { color: C.borderDark, width: 1 } });
  s.addShape("rect", { x: 0.75, y: 3.0, w: 0.07, h: 1.05, fill: { color: C.redOnDark }, line: { type: "none" } });
  s.addText([
    { text: "The skeptic's objection this thesis answers:  ", options: { bold: true, color: C.redOnDark, fontSize: 12.5 } },
    { text: "\u201cIf each flow reveals the attack stage, your RL agent is just an expensive classifier \u2014 a supervised model should match it.\u201d", options: { italic: true, color: C.textOnDark, fontSize: 12.5 } },
  ], { x: 1.02, y: 3.12, w: 8.0, h: 0.8, fontFace: F.body, margin: 0 });

  s.addText([
    { text: "Answer strategy: ", options: { bold: true, color: C.textOnDark } },
    { text: "make stage ambiguity a controlled dial (aliasing rate \u03b1), keep everything else fixed, and measure where classification stops being enough.", options: { color: C.mutedOnDark } },
  ], { x: 0.75, y: 4.45, w: 8.5, h: 0.65, fontFace: F.body, fontSize: 13, margin: 0 });
}

// Slide 9 — contributions map
function addContributions(pres) {
  const s = H.newContent(pres, {
    kicker: "2 \u00b7 Research Question",
    title: "Five Contributions \u2014 the Map of This Talk",
    kickerColor: C.gold,
    notes:
      "One breath per card; each returns later with evidence. 1: a genuinely partially observable kill-chain environment on real " +
      "CICIoT2023 traffic with a reactive attacker \u2014 the instrument. 2: the controlled crossover \u2014 tie at alpha=0, separation as " +
      "ambiguity grows. 3: reward ablation \u2014 the advantage survives removing the shaped reward (kills the privileged-reward objection). " +
      "4: on ten held-out attack classes the RL defender prevents more on every class, independent of detector recall. 5: honest " +
      "algorithm-reliability reporting + a fully hash-pinned reproducible artifact chain (462 tests).",
  });
  const cards = [
    ["1", "A genuinely partially observable kill-chain environment", "Reactive escalation attacker + real CICIoT2023 features + controllable aliasing \u03b1", C.blue],
    ["2", "A controlled crossover: windowed RL vs per-flow control", "Tie at \u03b1 = 0 by design; disjoint CIs from \u03b1 = " + N.headlineAlpha + " onward", C.red],
    ["3", "A reward ablation that answers the privileged-reward objection", "Best learned agent beats RF-Acting under both reward contracts", C.gold],
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
    notes:
      "Orient with the figure (thesis Fig. 3.1): three blocks. Offline preparation \u2014 project CICIoT2023 onto the 5-stage kill chain, " +
      "specify the closed-form attacker, train the supervised stage detector (it powers the baseline, not our agent). Online loop \u2014 " +
      "attacker emits a stage; realization engine samples a real feature row for that stage (session-coherent, aliased); Gymnasium env " +
      "builds the windowed observation; blue-team agent acts; escalation kernel moves the attacker. Held-out evaluation \u2014 benchmark vs " +
      "baselines + oracle, then ablations. Key sentence: the loop is genuinely CLOSED \u2014 the environment responds to the defender's " +
      "action; it does not replay a fixed trace.",
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
    notes:
      "Why this dataset: contemporary, large, real. 105 physical IoT devices at the Canadian Institute for Cybersecurity; 33 attack " +
      "types in 7 categories, attacks launched BY compromised IoT devices against others \u2014 realistic botnet behavior. Each row is a " +
      "pre-aggregated flow record: 46 statistical features. We apply a leakage-safe funnel \u2014 fit on the training partition only \u2014 " +
      "removing zero-variance, low-variance, and highly correlated columns: 29 features survive (timing/rate, header/size, TCP flags, " +
      "protocol indicators, distribution moments). Training pool after reserving OOD classes: 235,324 rows. Point to the funnel figure.",
  });
  H.img(s, "feature_selection_funnel", { x: 0.4, y: 1.35, maxW: 5.35, maxH: 3.85, frame: true });
  H.stat(s, { x: 6.15, y: 1.42, w: 3.3, value: N.devices + " devices", label: "physical IoT testbed \u2014 cameras, sensors, speakers, hubs", color: C.blue, valueSize: 23 });
  H.stat(s, { x: 6.15, y: 2.42, w: 3.3, value: "33 attack types", label: "7 categories: DDoS, DoS, Recon, Web, Brute force, Spoofing, Mirai", color: C.red, valueSize: 23 });
  H.stat(s, { x: 6.15, y: 3.42, w: 3.3, value: "46 \u2192 29 features", label: "leakage-safe selection, fit on the training partition only", color: C.ink, valueSize: 23 });
  s.addText(N.trainRows + " training rows after reserving held-out classes", {
    x: 6.15, y: 4.5, w: 3.3, h: 0.5, fontFace: F.body, fontSize: 10.5, italic: true, color: C.muted, margin: 0,
  });
}

// Slide 12 — kill-chain projection + splits
function addProjection(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "From 34 Labels to 5 Stages, 10 Held Out",
    titleSize: 24,
    notes:
      "Two moves. First, a deterministic map psi projects each of the 34 CICIoT2023 labels onto exactly one kill-chain stage: scanners " +
      "to RECON, brute-force/injection to ACCESS, spoofing + Mirai staging to MANEUVER, DoS/DDoS floods to IMPACT. Each stage now has " +
      "an empirical feature distribution \u2014 the environment samples real rows per stage. Second, the splits protocol: TEN attack " +
      "classes (at least two per non-benign stage) are RESERVED \u2014 never seen in training by either the detector or the agents. They " +
      "become our zero-day-like stress test later. Disjointness is asserted by automated tests.",
  });
  H.img(s, "projection_pipeline", { x: 0.4, y: 1.32, maxW: 9.2, maxH: 2.35, frame: true });
  // bottom row: two cards
  H.accentCard(s, 0.55, 3.95, 4.3, 1.2, C.blue);
  s.addText([
    { text: "Deterministic projection \u03c8", options: { bold: true, fontSize: 12, color: C.ink, breakLine: true } },
    { text: "34 labels \u2192 5 stages; each stage becomes an empirical feature distribution p(x\u2009|\u2009s) of real rows.", options: { fontSize: 10.5, color: C.inkSoft } },
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
    notes:
      "Anticipates the objection 'you fabricated the ambiguity'. PCA projection of the 29-dim feature space, colored by stage: " +
      "IMPACT floods separate sharply (flag counts, rates), but BENIGN/RECON \u2014 and the middle of the chain \u2014 overlap substantially. " +
      "Even with full feature information, neighboring stages are not linearly separable. So the aliasing dial alpha AMPLIFIES a real " +
      "property of the data; it does not invent one. Mention session coherence honestly: CICIoT2023 has no session key, so within-stage " +
      "draws are made contiguous at the environment layer \u2014 a modeling abstraction, declared as such (limitations).",
  });
  H.img(s, "dataset_raw_traffic_a", { x: 0.4, y: 1.35, maxW: 5.15, maxH: 3.7, frame: true, vAlign: "top" });
  H.bullets(s, [
    { text: "2-D PCA of the 29 features, colored by kill-chain stage", bold: true },
    { text: "IMPACT floods separate; BENIGN \u00b7 RECON \u00b7 ACCESS interleave heavily" },
    { text: "No single flow row reveals the stage \u2014 ambiguity is a property of the data" },
    { text: "The aliasing rate \u03b1 turns this real overlap into a controlled experimental dial" },
  ], { x: 5.95, y: 1.75, w: 3.5, h: 2.6, size: 11.5, gap: 12 });
  s.addText("Session coherence is imposed at the environment layer (no session key in the dataset) \u2014 declared as a modeling abstraction.", {
    x: 5.95, y: 4.45, w: 3.5, h: 0.7, fontFace: F.body, fontSize: 9.5, italic: true, color: C.muted, margin: 0,
  });
}

// Slide 14 — the reactive attacker (tug of war)
function addAttacker(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "A Reactive Attacker: the Escalation Tug-of-War",
    notes:
      "The attacker is not a script \u2014 it reacts to the defender's force. Signed force gap d = action minus recommended action for the " +
      "true stage. Three regimes: proportionate (d=0) pushes the attacker DOWN one stage with p=0.90 (0.98 for ISOLATE); under-force " +
      "(d\u2264\u22121) lets it ADVANCE with p_up scaled by proximity \u2014 momentum compounds as it nears IMPACT (sigma_min=0.4); over-force " +
      "(d\u2265+1) merely HOLDS \u2014 and the reward still charges the availability cost. Onset is autonomous from BENIGN (p=0.35 to RECON, " +
      "0.10 straight to ACCESS \u2014 stolen credential). PREVENTION = holding the attacker below IMPACT for the whole 100-step horizon. " +
      "Closed form, no learned attacker: fully specified, reproducible threat model \u2014 a declared scope boundary.",
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
    notes:
      "How partial observability is instrumented. Top figure: the state machine \u2014 solid arrows are attacker transitions; dashed arrows " +
      "are ALIASING: with probability alpha, the emitted feature row comes from an ADJACENT stage, not the true one. Z is a two-component " +
      "mixture: (1\u2212\u03b1) own-stage + \u03b1 adjacent-stage. At \u03b1=0, single rows are informative; at \u03b1=1 every row misleads. Same aliased " +
      "stream feeds every policy \u2014 nobody is privileged. Bottom: what the agent actually sees \u2014 the last 5 rows + their temporal deltas, " +
      "stacked into 290 dims. The window is the agent's only belief machinery; the RF baseline reads one row.",
  });
  H.img(s, "state_machine_aliasing", { x: 0.4, y: 1.38, maxW: 9.2, maxH: 1.75, frame: true, vAlign: "top" });
  const im2 = H.img(s, "obs_tensor_schematic", { x: 0.4, y: 3.55, maxW: 4.5, maxH: 1.52, frame: true, vAlign: "top" });
  H.bullets(s, [
    { text: "Z = (1\u2212\u03b1)\u00b7own-stage  +  \u03b1\u00b7adjacent-stage rows", bold: true },
    { text: "\u03b1 = 0: rows informative \u00b7 \u03b1 = 1: every row misleads" },
    { text: "Observation = last w = 5 rows + temporal deltas \u2192 290-dim" },
    { text: "Identical aliased stream for every policy \u2014 no one is privileged" },
  ], { x: 5.25, y: 3.62, w: 4.2, h: 1.6, size: 10.5, gap: 8 });
}

// Slide 16 — reward contracts
function addReward(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "Two Reward Contracts: Sparse Is Primary",
    titleSize: 24,
    notes:
      "Critical design decision. The COUPLED reward pays per step for choosing the stage-recommended action \u2014 but that rewards exactly " +
      "what a supervised classifier predicts: training under it alone would conflate 'learned to defend' with 'imitated a lookup table'. " +
      "So the PRIMARY contract is the sparse OUTCOME reward: action costs + terminal accounting + prevention bonus \u2014 no per-step stage " +
      "hints. Much harder credit assignment (reward arrives ~100 steps late), but it measures defense, not imitation. The coupled variant " +
      "is retained ONLY as the ablation that answers the privileged-reward objection. Benign guardrails and caps prevent degenerate " +
      "block-everything policies; availability cost rises with action severity.",
  });
  // outcome card (primary)
  H.accentCard(s, 0.55, 1.45, 4.35, 2.6, C.blue);
  s.addText("OUTCOME  \u00b7  primary", { x: 0.78, y: 1.6, w: 3.9, h: 0.28, fontFace: F.body, fontSize: 12.5, bold: true, color: C.blue, margin: 0 });
  H.bullets(s, [
    { text: "Sparse: action cost + terminal accounting only", bold: true },
    { text: "Prevented episode \u2192 +50 \u00b7 terminal defense \u2192 +250" },
    { text: "Compromise \u2192 \u2212200 (\u2212150 more if passive at impact)" },
    { text: "No per-step stage hints \u2014 measures defense, not imitation" },
  ], { x: 0.78, y: 1.98, w: 3.9, h: 1.9, size: 10.5, gap: 8, bulletColor: C.blue });
  // coupled card (ablation)
  H.accentCard(s, 5.15, 1.45, 4.35, 2.6, C.gold);
  s.addText("COUPLED  \u00b7  ablation only", { x: 5.38, y: 1.6, w: 3.9, h: 0.28, fontFace: F.body, fontSize: 12.5, bold: true, color: C.gold, margin: 0 });
  H.bullets(s, [
    { text: "Adds per-step shaping keyed to the true stage", bold: true },
    { text: "+5 proportionality bonus \u00b7 \u22125 disproportion penalty" },
    { text: "Benign guardrails: overreact \u221250 \u00b7 block-on-benign \u2212100" },
    { text: "Rewards exactly what a classifier predicts \u2014 a privileged signal" },
  ], { x: 5.38, y: 1.98, w: 3.9, h: 1.9, size: 10.5, gap: 8, bulletColor: C.gold });

  H.card(s, 0.55, 4.3, 8.95, 0.85, { fill: C.cardAlt });
  s.addText([
    { text: "Design principle:  ", options: { bold: true, color: C.ink } },
    { text: "if the reward hands the agent the stage label, a classifier wins by construction. Train under the sparse contract; keep the shaped one only to test that objection.", options: { color: C.inkSoft } },
  ], { x: 0.8, y: 4.42, w: 8.5, h: 0.62, fontFace: F.body, fontSize: 11.5, margin: 0 });
}

// Slide 17 — contenders
function addContenders(pres) {
  const s = H.newContent(pres, {
    kicker: "3 \u00b7 Framework",
    title: "The Contenders",
    notes:
      "Who competes. Learned agents: PPO, A2C, DQN \u2014 windowed observation, never see the true stage, never consume the detector. " +
      "Deployable baseline: RF-Acting \u2014 a hyperparameter-TUNED RandomForest stage detector (macro-F1 0.924; the strongest classifier a " +
      "practitioner would ship, not a straw man) + the recommended-action rule; memoryless: one row, one action. Trivial baselines: " +
      "always-observe, always-block, random \u2014 they bracket the reward scale. And the ORACLE: same action rule but reading the TRUE " +
      "stage \u2014 not a competitor, a measuring instrument: it prices perfect perception (+194.8 ceiling). Figure: detector position \u2014 " +
      "RF-Acting consumes the detector; our agents bypass it. That architectural contrast is the experiment.",
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
    notes:
      "Rigor slide \u2014 deliver with quiet confidence. 10 seeds per algorithm; FIXED 5M-step budget, NO early stopping (an early-stop rule " +
      "tuned to one algorithm truncates another \u2014 and full budgets are what exposed DQN's instability). Best-on-validation checkpoint " +
      "carried forward. Evaluation: n=300 episodes per policy on the held-out balanced-test split; 95% bootstrap CIs; separation = " +
      "disjoint intervals \u2014 a conservative criterion. Everything CPU-only. Reproducibility: every figure ships a manifest pinning input " +
      "hashes + git commit; 462 automated tests; the whole chain re-verifies on a fresh checkout. All numbers ahead carry these " +
      "guarantees.",
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
  addQuestion, addContributions, addArchitecture, addDataset,
  addProjection, addOverlap, addAttacker, addAliasing, addReward,
  addContenders, addProtocol,
};
