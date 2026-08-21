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
context.addWhyIoT(pres); //          4
context.addKillChain(pres); //       5
context.addRLPrimer(pres); //        6
context.addPOMDP(pres); //           7

// --- Act II: Research question ----------------------------------------------
qm.addQuestion(pres); //             8
qm.addContributions(pres); //        9

// --- Act III: Framework -------------------------------------------------------
qm.addArchitecture(pres); //        10
qm.addDataset(pres); //             11
qm.addProjection(pres); //          12
qm.addOverlap(pres); //             13
qm.addAttacker(pres); //            14
qm.addAliasing(pres); //            15
qm.addReward(pres); //              16
qm.addContenders(pres); //          17
qm.addProtocol(pres); //            18

// --- Act IV: Results ----------------------------------------------------------
results.addLearning(pres); //       19
results.addDoctrines(pres); //      20
results.addCrossover(pres); //      21
results.addCoupling(pres); //       22
results.addSweeps(pres); //         23
results.addOOD(pres); //            24
results.addRecallIndependence(pres); // 25

// --- Act V: Closing -----------------------------------------------------------
results.addLimitations(pres); //    26
results.addConclusions(pres); //    27
results.addFuture(pres); //         28
results.addThanks(pres); //         29

// --- Backup -------------------------------------------------------------------
backup.addBackupDivider(pres); //   30
backup.addRewardTable(pres); //     31
backup.addHparams(pres); //         32
backup.addMapping(pres); //         33
backup.addDetector(pres); //        34
backup.addBenignSafety(pres); //    35
backup.addRepro(pres); //           36

pres
  .writeFile({ fileName: "defense.pptx" })
  .then(() => console.log("OK defense.pptx written"))
  .catch((e) => {
    console.error("BUILD FAILED:", e);
    process.exit(1);
  });
