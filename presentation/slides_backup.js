"use strict";
const { C, F, N } = require("./theme");
const H = require("./helpers");

// Darker stage-color variants for table chips (white text stays >=4.5:1)
const STAGE_DARK = ["4A6350", "84662B", "94512A", "833B2A", "6E211B"];

function addBackupDivider(pres) {
  const s = H.newDark(pres, { notes: "Divider \u2014 backup material for Q&A. Not part of the timed talk." });
  s.addText("BACKUP", {
    x: 0.75, y: 2.3, w: 8.5, h: 0.7, fontFace: F.head, fontSize: 40, bold: true, color: C.textOnDark, margin: 0,
  });
  s.addText("Supporting detail for questions \u2014 reward constants \u00b7 hyperparameters \u00b7 kill-chain mapping \u00b7 detector performance \u00b7 benign safety \u00b7 reproducibility", {
    x: 0.75, y: 3.15, w: 8.0, h: 0.6, fontFace: F.body, fontSize: 13, color: C.mutedOnDark, margin: 0,
  });
}

// B1 — reward constants table
function addRewardTable(pres) {
  const s = H.newContent(pres, {
    kicker: "Backup B1",
    title: "Reward Constants (Environment Design Defaults)",
    kickerColor: C.muted,
    titleSize: 24,
    notes: "Full constant table from methodology Table 3.5. Calibration: terminal outcomes set the scale; per-step shaping an order of magnitude smaller; caps kill reward farming; guardrails make over-blocking on benign never profitable.",
  });
  const rows = [
    [["Constant", true], ["Value", true], ["Constant", true], ["Value", true]],
    ["impact terminal penalty", "\u2212200", "de-escalation reward (capped 150)", "+15"],
    ["defense-success bonus (terminal)", "+250", "proportionality bonus (capped 100)", "+5"],
    ["missed-impact penalty", "\u2212150", "disproportion penalty", "\u22125"],
    ["prevention bonus (truncation)", "+50", "benign-passive bonus", "+10"],
    ["overreact-on-benign", "\u221250", "block-on-benign / on-recon", "\u2212100 / \u221250"],
    ["p_down (ISOLATE)", "0.90 (0.98)", "p_up base / proximity floor", "0.90 / 0.4"],
    ["onset recon / access", "0.35 / 0.10", "action costs (obs\u2192iso)", "0, .1, .3, .5, .8"],
  ];
  const tableRows = rows.map((r, ri) =>
    r.map((c) => {
      const [txt, hdr] = Array.isArray(c) ? c : [c, false];
      return {
        text: txt,
        options: {
          fontFace: F.body, fontSize: hdr ? 10.5 : 10, bold: !!hdr,
          color: hdr ? "FFFFFF" : C.ink,
          fill: { color: hdr ? C.ink : ri % 2 === 0 ? C.card : "F1EEE8" },
          align: "left", valign: "middle",
        },
      };
    })
  );
  s.addTable(tableRows, {
    x: 0.55, y: 1.5, w: 8.9, colW: [3.0, 1.45, 3.0, 1.45],
    border: { pt: 0.5, color: C.border }, rowH: 0.38, margin: 0.06,
  });
  s.addText("Sparse outcome contract strips the five stage-conditioned shaping components; action cost + terminal accounting + prevention remain.", {
    x: 0.55, y: 4.85, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 10, italic: true, color: C.muted, margin: 0,
  });
}

// B2 — hyperparameters
function addHparams(pres) {
  const s = H.newContent(pres, {
    kicker: "Backup B2",
    title: "Training Hyperparameters (Grid-Searched per Algorithm)",
    kickerColor: C.muted,
    titleSize: 24,
    notes: "Appendix C. Notables: A2C n_steps=256 (vs default 5) \u2014 long rollouts for the sparse credit delay; DQN 200k replay buffer + slow target updates; same 2\u00d764 MLP everywhere isolates the learning rule.",
  });
  const rows = [
    ["Hyperparameter", "PPO", "A2C", "DQN"],
    ["Learning rate", "3 \u00d7 10\u207b\u2074", "7 \u00d7 10\u207b\u2074", "5 \u00d7 10\u207b\u2074"],
    ["Discount \u03b3", "0.99", "0.99", "0.99"],
    ["Rollout n_steps", "2048", "256", "\u2014"],
    ["GAE \u03bb / entropy / value coef", "0.95 / 0.01 / 0.5", "0.95 / 0.01 / 0.5", "\u2014"],
    ["Epochs \u00d7 batch", "10 \u00d7 64", "\u2014", "batch 64"],
    ["Replay buffer / learning starts", "\u2014", "\u2014", "200k / 5000"],
    ["Target update / exploration", "\u2014", "\u2014", "5000 / 1.0\u21920.05 (20%)"],
    ["Network (all)", "MLP 2\u00d764 ReLU", "MLP 2\u00d764 ReLU", "MLP 2\u00d764 ReLU"],
  ];
  const tableRows = rows.map((r, ri) =>
    r.map((c, ci) => ({
      text: c,
      options: {
        fontFace: F.body, fontSize: ri === 0 ? 11 : 10, bold: ri === 0 || ci === 0,
        color: ri === 0 ? "FFFFFF" : C.ink,
        fill: { color: ri === 0 ? C.ink : ri % 2 === 1 ? C.card : "F1EEE8" },
        align: ci === 0 ? "left" : "center", valign: "middle",
      },
    }))
  );
  s.addTable(tableRows, {
    x: 0.55, y: 1.5, w: 8.9, colW: [3.4, 1.83, 1.83, 1.84],
    border: { pt: 0.5, color: C.border }, rowH: 0.36, margin: 0.06,
  });
  s.addText("10 seeds \u00d7 5,000,000 steps, no early stopping; best-on-validation checkpoint carried to the benchmark.", {
    x: 0.55, y: 5.0, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 10, italic: true, color: C.muted, margin: 0,
  });
}

// B3 — kill-chain mapping
function addMapping(pres) {
  const s = H.newContent(pres, {
    kicker: "Backup B3",
    title: "CICIoT2023 \u2192 Kill-Chain Mapping (34 Labels, 10 Held Out)",
    kickerColor: C.muted,
    titleSize: 23,
    notes: "Appendix B. Stars = the 10 reserved OOD classes, at least two per non-benign stage. Unknown labels raise a hard error \u2014 no silent leakage.",
  });
  const rows = [
    ["Stage", "CICIoT2023 labels (\u2605 = held out)"],
    ["BENIGN", "BenignTraffic"],
    ["RECON", "PortScan \u00b7 OSScan\u2605 \u00b7 HostDiscovery \u00b7 PingSweep \u00b7 VulnerabilityScan\u2605"],
    ["ACCESS", "BrowserHijacking \u00b7 CommandInjection \u00b7 SqlInjection\u2605 \u00b7 Backdoor_Malware \u00b7 Uploading_Attack \u00b7 XSS\u2605 \u00b7 DictionaryBruteForce"],
    ["MANEUVER", "MITM-ArpSpoofing \u00b7 DNS_Spoofing\u2605 \u00b7 Mirai-greeth_flood \u00b7 Mirai-greip_flood \u00b7 Mirai-udpplain\u2605"],
    ["IMPACT", "4 DoS + 12 DDoS flood variants, incl. DoS-SYN_Flood\u2605 \u00b7 DDoS-ACK_Fragmentation\u2605 \u00b7 DDoS-SlowLoris\u2605 \u00b7 DDoS-HTTP_Flood\u2605"],
  ];
  const tableRows = rows.map((r, ri) =>
    r.map((c, ci) => ({
      text: c,
      options: {
        fontFace: F.body, fontSize: ri === 0 ? 11 : 10, bold: ri === 0 || ci === 0,
        color: ri === 0 ? "FFFFFF" : ci === 0 && ri > 0 ? "FFFFFF" : C.ink,
        fill: { color: ri === 0 ? C.ink : ci === 0 ? STAGE_DARK[ri - 1] : ri % 2 === 1 ? C.card : "F1EEE8" },
        align: "left", valign: "middle",
      },
    }))
  );
  s.addTable(tableRows, {
    x: 0.55, y: 1.5, w: 8.9, colW: [1.5, 7.4],
    border: { pt: 0.5, color: C.border }, rowH: 0.52, margin: 0.07,
  });
  s.addText("Deterministic map \u03c8 implemented in code; \u2265 2 held-out classes per non-benign stage enable the OOD probe across the whole chain.", {
    x: 0.55, y: 4.95, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 10, italic: true, color: C.muted, margin: 0,
  });
}

// B4 — detector performance
function addDetector(pres) {
  const s = H.newContent(pres, {
    kicker: "Backup B4",
    title: "The Tuned RandomForest Detector Is Not a Straw Man",
    kickerColor: C.muted,
    titleSize: 24,
    notes: "Fairness evidence: 54-cell grid search on validation macro-F1; interior optimum (200 trees, depth 20, balanced weights); validation F1 flat at 0.927\u00b10.005 across tree count \u2014 not under-tuned. Balanced-test macro-F1 0.924; worst class 0.87 (recon/access ambiguity); impact 0.999. Only material confusion: RECON\u2192ACCESS 13.2% \u2014 inherent overlap in the middle of the chain.",
  });
  // left figure AR 1.736 (w=maxW 4.55 -> h 2.62); right AR 1.219 (h 3.3 -> w 4.02)
  H.img(s, "tuned_rf_per_class_f1", { x: 0.45, y: 1.42, maxW: 4.55, maxH: 3.3, frame: true, align: "left", vAlign: "top" });
  H.img(s, "tuned_rf_confusion", { x: 5.45, y: 1.42, maxW: 4.05, maxH: 3.3, frame: true, align: "left", vAlign: "top" });
  s.addText([
    { text: "Macro-F1 " + N.rfF1 + " (held-out balanced test) \u00b7 worst class " + N.rfWorstF1 + " on the ambiguous middle stages \u00b7 grid-flat validation \u2014 ", options: { color: C.inkSoft } },
    { text: "the strongest classifier a practitioner would deploy.", options: { bold: true, color: C.ink } },
  ], { x: 0.55, y: 4.95, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 10.5, margin: 0 });
}

// B5 — benign safety
function addBenignSafety(pres) {
  const s = H.newContent(pres, {
    kicker: "Backup B5",
    title: "Benign Safety: Prevention Without Collateral Damage",
    kickerColor: C.muted,
    titleSize: 24,
    notes: "The availability axis. Benign FPR = fraction of true-benign flows hit with block/isolate. Learned agents: PPO 0.89%, A2C 0.66%, DQN 0.46% \u2014 all under the 1% operational threshold. Random 41.3%; always-block 100% (that is why its perfect prevention is inadmissible); always-observe 0% but loses every episode (~\u2212350). Trivials bracket the trade-off; learned agents are the only policies strong AND safe.",
  });
  const bars = [
    ["always-OBSERVE", 0, "0%", C.muted, "never acts \u2014 every attack lands"],
    ["DQN", 0.46, N.fprDQN + "%", C.blue, "under threshold"],
    ["A2C", 0.66, N.fprA2C + "%", C.blue, "under threshold \u2014 strongest defender"],
    ["PPO", 0.89, N.fprPPO + "%", C.blue, "under threshold"],
    ["Random", 41.3, N.fprRandom + "%", C.gold, "disrupts 4 in 10 benign flows"],
    ["always-BLOCK", 100, "100%", C.red, "perfect prevention, inadmissible"],
  ];
  const bx = 3.1, bw = 4.0, y0 = 1.62, rh = 0.54;
  const xThresh = bx + (Math.sqrt(1) / 10) * bw;
  bars.forEach(([name, v, lab, color, note], i) => {
    const y = y0 + i * rh;
    s.addText(name, { x: 0.55, y, w: 1.6, h: 0.34, fontFace: F.body, fontSize: 10, bold: true, color: C.ink, align: "right", valign: "middle", margin: 0 });
    s.addText(lab, { x: 2.2, y, w: 0.75, h: 0.34, fontFace: F.body, fontSize: 10.5, bold: true, color, align: "right", valign: "middle", margin: 0 });
    s.addShape("rect", { x: bx, y: y + 0.03, w: bw, h: 0.28, fill: { color: "ECE8E1" }, line: { type: "none" } });
    const wv = Math.max(0.03, (Math.sqrt(v) / 10) * bw); // sqrt scale for visibility
    s.addShape("rect", { x: bx, y: y + 0.03, w: wv, h: 0.28, fill: { color }, line: { type: "none" } });
    const noteX = Math.max(bx + wv, xThresh) + 0.14;
    s.addText(note, { x: noteX, y: y + 0.01, w: 9.75 - noteX, h: 0.32, fontFace: F.body, fontSize: 9, color: C.inkSoft, valign: "middle", margin: 0 });
  });
  // threshold marker at 1%
  s.addShape("line", { x: xThresh, y: y0 - 0.1, w: 0, h: bars.length * rh + 0.06, line: { color: C.red, width: 1.5, dashType: "dash" } });
  s.addText("1% operational threshold", { x: xThresh - 0.5, y: y0 - 0.38, w: 2.2, h: 0.24, fontFace: F.body, fontSize: 9, bold: true, color: C.red, margin: 0 });
  s.addText("Aggressive action (block / isolate) on true-benign flows \u00b7 square-root scale for visibility \u00b7 always-OBSERVE scores \u2248 \u2212350 reward.", {
    x: 0.55, y: 5.0, w: 8.9, h: 0.28, fontFace: F.body, fontSize: 9.5, italic: true, color: C.muted, margin: 0,
  });
}

// B6 — reproducibility chain
function addRepro(pres) {
  const s = H.newContent(pres, {
    kicker: "Backup B6",
    title: "Reproducibility: Every Number Has a Chain of Custody",
    kickerColor: C.muted,
    titleSize: 24,
    notes: "Every figure ships a manifest.json: SHA-256 of every input artifact + the producing git commit + the exact command. A harness re-walks the chain on a fresh checkout. Thesis numbers are macro-generated from canonical JSONs \u2014 never hand-typed. 462 tests. Clone \u2192 pytest \u2192 reproducibility_smoke \u2192 PASS, no retraining needed.",
  });
  const steps = [
    ["experiment run", "canonical summary JSON + manifest (input hashes + git SHA)"],
    ["render layer", "macros + tables generated from JSONs \u2014 no hand-typed numbers"],
    ["thesis build", "LaTeX consumes generated macros; figures pinned by hash"],
    ["verification", "harness re-walks every manifest on a fresh checkout \u2192 PASS"],
  ];
  let x = 0.55;
  const bw2 = 2.12, gap = (8.9 - 4 * bw2) / 3;
  for (let i = 0; i < steps.length; i++) {
    H.card(s, x, 1.75, bw2, 1.55, { fill: i === 3 ? C.ink : C.card });
    s.addText(String(i + 1), {
      x, y: 1.88, w: bw2, h: 0.3, fontFace: F.head, fontSize: 16, bold: true,
      color: i === 3 ? C.gold : C.blue, align: "center", margin: 0,
    });
    s.addText(steps[i][0], {
      x, y: 2.2, w: bw2, h: 0.28, fontFace: F.body, fontSize: 11, bold: true,
      color: i === 3 ? "FFFFFF" : C.ink, align: "center", margin: 0,
    });
    s.addText(steps[i][1], {
      x: x + 0.1, y: 2.5, w: bw2 - 0.2, h: 0.72, fontFace: F.body, fontSize: 9,
      color: i === 3 ? "D8D5D0" : C.inkSoft, align: "center", margin: 0,
    });
    if (i < 3) H.arrow(s, x + bw2 + 0.04, 2.52, gap - 0.08, { color: C.inkSoft });
    x += bw2 + gap;
  }
  H.stat(s, { x: 0.9, y: 3.85, w: 2.6, value: N.tests, label: "tests pass on a fresh checkout", color: C.blue, valueSize: 26 });
  H.stat(s, { x: 3.9, y: 3.85, w: 2.6, value: "SHA-256", label: "hash chain, splits to figures", color: C.ink, valueSize: 26 });
  H.stat(s, { x: 6.9, y: 3.85, w: 2.6, value: "0 retrain", label: "verification without retraining", color: C.green, valueSize: 26 });
}

module.exports = {
  addBackupDivider, addRewardTable, addHparams, addMapping,
  addDetector, addBenignSafety, addRepro,
};
