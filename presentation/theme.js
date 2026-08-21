"use strict";
// ---------------------------------------------------------------------------
// Theme: sober academic, charcoal ink on warm off-white, deep-red threat
// accent, slate-blue defender accent. Dark charcoal for title/divider slides.
// Fonts: Georgia (headers) + Arial (body) — both native in Google Slides.
// ---------------------------------------------------------------------------

const C = {
  // canvas
  bgLight: "F7F5F2", // warm off-white content background
  bgDark: "232327", // charcoal title/divider background
  card: "FFFFFF",
  cardAlt: "EFECE6",
  border: "DDD8CF",
  borderDark: "3A3A40",

  // ink
  ink: "2A2A2E", // main charcoal text
  inkSoft: "55555C",
  muted: "76767E",
  mutedOnDark: "A5A3A0",
  textOnDark: "F1EEE8",

  // semantic accents
  red: "9E2B25", // attacker / threat / RF degradation
  redSoft: "C4B5B0", // hairlines in red zones
  redOnDark: "C05048",
  blue: "3E5C76", // defender / RL agents
  blueSoft: "8FA3B5",
  green: "5F7D62", // benign / positive outcome
  gold: "B08A3E", // highlight sparingly (oracle, key stats)

  // kill-chain stage ramp (safe -> danger, muted)
  stage: ["5F7D62", "B08A3E", "C0703C", "A94E38", "8C2B24"],
};

const F = {
  head: "Georgia",
  body: "Arial",
};

// canonical numbers (from tex/generated/numbers.tex — macro layer, never
// hand-typed from memory)
const N = {
  numSeeds: "10",
  headlineAlpha: "0.4",
  oracle: "+194.8",
  // alpha curve (PPO / RF / A2C / DQN)
  a0ppo: "+138.6", a0rf: "+136.5", a0gap: "+2.1", a0a2c: "+147.1", a0dqn: "+116.6",
  a2ppo: "+121.6", a2rf: "+113.2", a2gap: "+8.4",
  a4ppo: "+121.3", a4rf: "+94.4", a4gap: "+26.9", a4a2c: "+138.7", a4dqn: "+72.5",
  a6ppo: "+113.3", a6rf: "+64.0", a6gap: "+49.3", a6a2c: "+151.4",
  a8ppo: "+135.2", a8rf: "+20.5", a8gap: "+114.7",
  a10ppo: "+131.9", a10rf: "-29.3", a10gap: "+161.2", a10a2c: "+142.0",
  // coupling ablation
  cplDQN: "+226.2", cplPPO: "+162.4", cplA2C: "+144.8", cplGap: "-63.1",
  outA2C: "+146.1", outPPO: "+126.2", outDQN: "-8.6", outGap: "-63.0",
  // reliability
  sdPPO: "~15", sdA2C: "~9", sdDQN: "~52",
  // F10
  f10a2c0: "-7.2", f10a2c1: "+147.3", f10ppo0: "-46.2", f10ppo1: "+142.6",
  f10dqn0: "-79.6", f10dqn1: "+78.3",
  // F17
  f17a2c0: "+142.6", f17a2c75: "+112.7", f17a2cComp0: "0.233", f17a2cComp75: "0.412",
  f17ppo0: "+123.2", f17ppo75: "+91.9",
  // F15 OOD
  oodRLlo: "0.71", oodRLhi: "0.85", oodRFlo: "0.00", oodRFhi: "0.15",
  oodAdvLo: "+0.70", oodAdvHi: "+0.78",
  spearman: "0.22", spearmanP: "0.54", pearson: "-0.02", pearsonP: "0.95",
  olsLo: "-0.08", olsHi: "+0.04",
  // detector
  rfF1: "0.924", rfWorstF1: "0.87", rfImpactF1: "0.999", rfConf: "13.2",
  // benign safety
  fprPPO: "0.89", fprDQN: "0.46", fprA2C: "0.66", fprRandom: "41.3",
  // action shares
  a2cManeuverBlock: "84.4",
  // footprint
  policyKB: "90", policyParams: "23K", rfMB: "181", rfNodes: "1.7M", ratio: "1956",
  // misc
  tests: "462", trainRows: "235,324", devices: "105",
};

module.exports = { C, F, N };
