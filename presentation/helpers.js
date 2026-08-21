"use strict";
const { C, F } = require("./theme");

// Slide canvas: LAYOUT_16x9 -> 10.0 x 5.625 in
const PAGE = { w: 10.0, h: 5.625 };
const MARGIN = 0.55;

// Measured pixel dimensions of every asset (pdftoppm output), used for
// aspect-ratio-correct placement.
const IMG_DIMS = {
  "F10_aggressiveness": [1482, 946],
  "F15_ood_robustness": [2362, 2341],
  "F15b_recall_vs_advantage": [1701, 1151],
  "F17_evasion_sweep": [1333, 935],
  "F3_learning_curves": [1483, 911],
  "F4_action_distribution": [2724, 2279],
  "Falpha_curve": [1482, 947],
  "Fcoupling_reward_gap": [1483, 861],
  "architecture_diagram": [1525, 1394],
  "class_distribution": [2837, 1187],
  "dataset_raw_traffic_a": [1527, 1130],
  "dataset_raw_traffic_b": [2349, 779],
  "feature_selection_funnel": [1525, 1039],
  "obs_tensor_schematic": [1233, 862],
  "projection_pipeline": [1621, 433],
  "red_team_env": [1217, 480],
  "rl_agent_loop": [973, 541],
  "stage_detector_position": [1472, 673],
  "stage_distribution": [1847, 1073],
  "state_machine_aliasing": [1595, 334],
  "tuned_rf_confusion": [1644, 1349],
  "tuned_rf_per_class_f1": [2033, 1171],
  "unicamp_logo": [692, 730],
  "unicamp_logo_white": [692, 730],
  "unicamp_logo_dark": [692, 730],
};

const KC_STAGES = ["BENIGN", "RECON", "ACCESS", "MANEUVER", "IMPACT"];
const KC_ACTIONS = ["OBSERVE", "LOG", "RESTRICT", "BLOCK", "ISOLATE"];

let pageCounter = 0;
function resetCounter() { pageCounter = 0; }

function footer(slide, label) {
  slide.addText(label, {
    x: PAGE.w - 1.05, y: PAGE.h - 0.34, w: 0.75, h: 0.24,
    fontFace: F.body, fontSize: 8, color: C.muted, align: "right", margin: 0,
  });
  slide.addText("FEEC \u00b7 UNICAMP", {
    x: 0.3, y: PAGE.h - 0.34, w: 2.0, h: 0.24,
    fontFace: F.body, fontSize: 8, color: C.muted, align: "left", margin: 0,
  });
}

// Standard light content slide with kicker + title. Returns slide.
function newContent(pres, { kicker, title, notes, kickerColor, titleSize = 27 }) {
  pageCounter += 1;
  const slide = pres.addSlide();
  slide.background = { color: C.bgLight };
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: MARGIN, y: 0.28, w: PAGE.w - 2 * MARGIN, h: 0.24,
      fontFace: F.body, fontSize: 10.5, bold: true, charSpacing: 2,
      color: kickerColor || C.blue, margin: 0, valign: "top",
    });
  }
  if (title) {
    // Auto-shrink so the title always fits ONE line (Georgia bold avg char
    // width ~0.0078 in/pt): pt_max ~= 1080 / len. Prevents kicker collisions.
    const fitted = Math.max(17, Math.min(titleSize, Math.floor(1080 / title.length)));
    slide.addText(title, {
      x: MARGIN, y: 0.54, w: PAGE.w - 2 * MARGIN, h: 0.6,
      fontFace: F.head, fontSize: fitted, bold: true, color: C.ink,
      margin: 0, valign: "top",
    });
  }
  footer(slide, String(pageCounter));
  if (notes) slide.addNotes(notes);
  return slide;
}

// Dark slide (title, thanks, backup divider)
function newDark(pres, { notes, count = true } = {}) {
  if (count) pageCounter += 1;
  const slide = pres.addSlide();
  slide.background = { color: C.bgDark };
  if (notes) slide.addNotes(notes);
  return slide;
}

// Aspect-ratio-correct image placement inside a bounding box (contain).
// Optional white card frame behind the image.
function img(slide, name, { x, y, maxW, maxH, frame = false, align = "center", vAlign = "center" }) {
  const [pw, ph] = IMG_DIMS[name];
  const ar = pw / ph;
  let w = maxW, h = maxW / ar;
  if (h > maxH) { h = maxH; w = maxH * ar; }
  let ix = x, iy = y + (maxH - h) / 2;
  if (vAlign === "top") iy = y;
  if (vAlign === "bottom") iy = y + (maxH - h);
  if (align === "center") ix = x + (maxW - w) / 2;
  if (align === "left") ix = x;
  if (align === "right") ix = x + (maxW - w);
  if (frame) {
    slide.addShape("rect", {
      x: ix - 0.09, y: iy - 0.09, w: w + 0.18, h: h + 0.18,
      fill: { color: C.card }, line: { color: C.border, width: 0.75 },
    });
  }
  slide.addImage({ path: `assets/${name}.png`, x: ix, y: iy, w, h });
  return { x: ix, y: iy, w, h };
}

function card(slide, x, y, w, h, { fill = C.card, line = C.border, lineW = 0.75 } = {}) {
  slide.addShape("rect", {
    x, y, w, h, fill: { color: fill }, line: { color: line, width: lineW },
  });
}

// Left accent-edge card (the deck's repeated motif for key claims)
function accentCard(slide, x, y, w, h, color, opts = {}) {
  card(slide, x, y, w, h, opts);
  slide.addShape("rect", { x, y, w: 0.065, h, fill: { color }, line: { type: "none" } });
}

// Big stat callout
function stat(slide, { x, y, w, value, label, color = C.ink, valueSize = 30, labelColor = C.inkSoft, labelSize = 10, align = "left" }) {
  slide.addText(value, {
    x, y, w, h: valueSize / 60, fontFace: F.head, fontSize: valueSize,
    bold: true, color, align, margin: 0,
  });
  slide.addText(label, {
    x, y: y + valueSize / 60 + 0.04, w, h: 0.55, fontFace: F.body,
    fontSize: labelSize, color: labelColor, align, margin: 0,
  });
}

// Bullet list from [text | {text, bold, sub, color, size}] entries
function bullets(slide, items, { x, y, w, h, size = 12.5, color = C.ink, gap = 8, bulletColor } = {}) {
  const runs = items.map((it, i) => {
    const o = typeof it === "string" ? { text: it } : it;
    return {
      text: o.text,
      options: {
        bullet: o.sub
          ? { code: "2013", indent: 12 }
          : { code: "25AA", indent: 14, color: bulletColor || C.blue },
        indentLevel: o.sub ? 1 : 0,
        bold: !!o.bold,
        color: o.color || color,
        fontSize: o.size || (o.sub ? size - 1 : size),
        paraSpaceAfter: i === items.length - 1 ? 0 : gap,
        breakLine: true,
      },
    };
  });
  slide.addText(runs, { x, y, w, h, fontFace: F.body, valign: "top", margin: 0 });
}

// Kill-chain chip strip (the deck's visual motif).
// opts: { labels: 'stage'|'action'|'both', focus: [idx], y2label }
function kcStrip(slide, x, y, w, { labels = "stage", focus = null, chipH = 0.42, fontSize = 9.5 } = {}) {
  const gapW = 0.18;
  const chipW = (w - 4 * gapW) / 5;
  for (let i = 0; i < 5; i++) {
    const cx = x + i * (chipW + gapW);
    const dim = focus && !focus.includes(i);
    slide.addShape("roundRect", {
      x: cx, y, w: chipW, h: chipH, rectRadius: 0.05,
      fill: { color: C.stage[i], transparency: dim ? 72 : 0 },
      line: { type: "none" },
    });
    const mainLabel = labels === "action" ? KC_ACTIONS[i] : KC_STAGES[i];
    slide.addText(mainLabel, {
      x: cx, y: y + (labels === "both" ? -0.02 : 0), w: chipW, h: chipH * (labels === "both" ? 0.62 : 1),
      fontFace: F.body, fontSize, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    if (labels === "both") {
      slide.addText(KC_ACTIONS[i], {
        x: cx, y: y + chipH * 0.5, w: chipW, h: chipH * 0.48,
        fontFace: F.body, fontSize: fontSize - 2, color: "FFFFFF",
        align: "center", valign: "middle", margin: 0,
      });
    }
    if (i < 4) {
      slide.addText("\u25B8", {
        x: cx + chipW - 0.015, y: y + chipH / 2 - 0.13, w: gapW + 0.03, h: 0.26,
        fontFace: F.body, fontSize: 12, color: C.muted, align: "center", valign: "middle", margin: 0,
      });
    }
  }
}

// Simple horizontal arrow
function arrow(slide, x, y, len, { color = C.inkSoft, width = 1.75, dash } = {}) {
  slide.addShape("line", {
    x, y, w: len, h: 0,
    line: { color, width, endArrowType: "triangle", dashType: dash },
  });
}

// Section tag chip used on content slides (small, top-right)
function sectionTag(slide, text, color) {
  slide.addText(text.toUpperCase(), {
    x: PAGE.w - 2.6, y: 0.3, w: 2.05, h: 0.26, align: "right",
    fontFace: F.body, fontSize: 9, bold: true, charSpacing: 1.5,
    color: color || C.muted, margin: 0,
  });
}

module.exports = {
  PAGE, MARGIN, IMG_DIMS, KC_STAGES, KC_ACTIONS,
  newContent, newDark, img, card, accentCard, stat, bullets, kcStrip,
  arrow, footer, sectionTag, resetCounter,
};
