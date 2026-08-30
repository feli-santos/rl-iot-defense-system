"use strict";
// ---------------------------------------------------------------------------
// MSc defense deck builder — Felipe A. O. dos Santos, FEEC/UNICAMP.
// Usage:  node build.js   (from presentation/)
// Output: defense.pptx
// ---------------------------------------------------------------------------
const pptxgen = require("pptxgenjs");
const H = require("./helpers");
const opening = require("./slides_opening");
const context = require("./slides_context");
const qm = require("./slides_question_method");
const results = require("./slides_results");
const backup = require("./slides_backup");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Felipe Augusto Oliveira dos Santos";
pres.company = "FEEC / UNICAMP";
pres.title = "A Deep Reinforcement Learning Framework for Autonomous Cybersecurity Defense in IoT Networks";
pres.subject = "Master's Defense";

H.resetCounter();

// --- Opening ---------------------------------------------------------------
opening.addTitle(pres); //           1
opening.addSpeaker(pres); //         2
opening.addAgenda(pres); //          3

// --- Act I: Context ---------------------------------------------------------
context.addIoTNetwork(pres); //      4
context.addWhyIoT(pres); //          5
context.addKillChain(pres); //       6
context.addRLPrimer(pres); //        7
context.addPOMDP(pres); //           8

// --- Act II: Research question + objectives ---------------------------------
qm.addQuestion(pres); //             9
qm.addObjectives(pres); //          10
qm.addContributions(pres); //       11

// --- Act III: Framework -------------------------------------------------------
qm.addArchitecture(pres); //        12
qm.addDataset(pres); //             13
qm.addProjection(pres); //          14
qm.addOverlap(pres); //             15
qm.addAttacker(pres); //            16
qm.addAliasing(pres); //            17
qm.addReward(pres); //              18
qm.addContenders(pres); //          19
qm.addProtocol(pres); //            20

// --- Act IV: Results ----------------------------------------------------------
results.addLearning(pres); //       21
results.addDoctrines(pres); //      22
results.addCrossover(pres); //      23
results.addCoupling(pres); //       24
results.addSweeps(pres); //         25
results.addOOD(pres); //            26
results.addRecallIndependence(pres); // 27

// --- Act V: Closing -----------------------------------------------------------
results.addLimitations(pres); //    28
results.addConclusions(pres); //    29
results.addFuture(pres); //         30
results.addThanks(pres); //         31

// --- Backup -------------------------------------------------------------------
backup.addBackupDivider(pres); //   32
backup.addRewardTable(pres); //     33
backup.addHparams(pres); //         34
backup.addMapping(pres); //         35
backup.addDetector(pres); //        36
backup.addBenignSafety(pres); //    37
backup.addRepro(pres); //           38
backup.addFootprint(pres); //       39

const OUT = "MASTER'S DEFENSE.pptx";
pres
  .writeFile({ fileName: OUT })
  .then(() => console.log(`OK ${OUT} written`))
  .catch((e) => {
    console.error("BUILD FAILED:", e);
    process.exit(1);
  });
