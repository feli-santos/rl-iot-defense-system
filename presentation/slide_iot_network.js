"use strict";
// ---------------------------------------------------------------------------
// Standalone one-slide deck: "What Is an IoT Network?"
// Bridge slide to be inserted between Agenda (3) and the attack-surface
// slide (4). Native shapes only — no external figures.
// Build: node slide_iot_network.js  ->  slide_iot_network.pptx
// ---------------------------------------------------------------------------
const pptxgen = require("pptxgenjs");
const { C, F } = require("./theme");
const H = require("./helpers");

const pres = new pptxgen();
pres.defineLayout({ name: "W16x9", width: 10, height: 5.625 });
pres.layout = "W16x9";

const s = pres.addSlide();
s.background = { color: C.bgLight };

// --- kicker + title (same geometry as newContent, no page number) ----------
s.addText("1 \u00b7 CONTEXT", {
  x: H.MARGIN, y: 0.28, w: 8.9, h: 0.24,
  fontFace: F.body, fontSize: 10.5, bold: true, charSpacing: 2,
  color: C.green, margin: 0, valign: "top",
});
s.addText("What Is an IoT Network?", {
  x: H.MARGIN, y: 0.54, w: 8.9, h: 0.6,
  fontFace: F.head, fontSize: 27, bold: true, color: C.ink, margin: 0, valign: "top",
});
s.addText("FEEC \u00b7 UNICAMP", {
  x: 0.3, y: H.PAGE.h - 0.34, w: 2.0, h: 0.24,
  fontFace: F.body, fontSize: 8, color: C.muted, align: "left", margin: 0,
});

// --- opener line ------------------------------------------------------------
s.addText("Everyday objects that sense, decide, and act \u2014 connected through a gateway to cloud services.", {
  x: 0.55, y: 1.24, w: 8.9, h: 0.3, fontFace: F.body, fontSize: 13, italic: true,
  color: C.inkSoft, margin: 0,
});

// ============================ DIAGRAM (3 layers) ============================
// Zone 1 — the "things"
H.card(s, 0.55, 1.7, 2.95, 2.1, { fill: C.cardAlt });
s.addShape("rect", { x: 0.55, y: 1.7, w: 2.95, h: 0.06, fill: { color: C.green }, line: { type: "none" } });
s.addText("Things \u2014 sensors & actuators", {
  x: 0.68, y: 1.8, w: 2.7, h: 0.24, fontFace: F.body, fontSize: 11, bold: true, color: C.ink, margin: 0,
});
s.addText("constrained \u00b7 cheap \u00b7 wireless", {
  x: 0.68, y: 2.04, w: 2.7, h: 0.2, fontFace: F.body, fontSize: 8.5, italic: true, color: C.muted, margin: 0,
});
const devices = ["Camera", "Motion sensor", "Smart lock", "Thermostat", "Speaker", "Wearable"];
const chipW = 1.3, chipH = 0.38, gx = 0.14, gy = 0.1;
devices.forEach((name, i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const cx = 0.67 + col * (chipW + gx);
  const cy = 2.32 + row * (chipH + gy);
  s.addShape("roundRect", {
    x: cx, y: cy, w: chipW, h: chipH, rectRadius: 0.05,
    fill: { color: C.card }, line: { color: C.border, width: 0.75 },
  });
  s.addText(name, {
    x: cx, y: cy, w: chipW, h: chipH, fontFace: F.body, fontSize: 9,
    color: C.ink, align: "center", valign: "middle", margin: 0,
  });
});

// Link 1: things -> gateway (bidirectional)
s.addText("Zigbee \u00b7 BLE\nWi-Fi", {
  x: 3.5, y: 2.28, w: 0.8, h: 0.36, fontFace: F.body, fontSize: 7.5,
  color: C.muted, align: "center", margin: 0,
});
s.addShape("line", {
  x: 3.5, y: 2.75, w: 0.8, h: 0,
  line: { color: C.inkSoft, width: 1.5, beginArrowType: "triangle", endArrowType: "triangle" },
});

// Zone 2 — gateway / hub
s.addShape("roundRect", {
  x: 4.3, y: 2.3, w: 1.85, h: 0.9, rectRadius: 0.07,
  fill: { color: C.card }, line: { color: C.blue, width: 1.25 },
});
s.addText("Gateway / Hub", {
  x: 4.3, y: 2.42, w: 1.85, h: 0.28, fontFace: F.body, fontSize: 11.5, bold: true,
  color: C.blue, align: "center", margin: 0,
});
s.addText("aggregates traffic \u00b7 bridges protocols", {
  x: 4.38, y: 2.72, w: 1.69, h: 0.4, fontFace: F.body, fontSize: 8.5,
  color: C.inkSoft, align: "center", margin: 0,
});

// Link 2: gateway -> cloud (bidirectional)
s.addText("MQTT \u00b7 HTTP", {
  x: 6.11, y: 2.46, w: 0.84, h: 0.2, fontFace: F.body, fontSize: 7.5,
  color: C.muted, align: "center", margin: 0,
});
s.addShape("line", {
  x: 6.15, y: 2.75, w: 0.8, h: 0,
  line: { color: C.inkSoft, width: 1.5, beginArrowType: "triangle", endArrowType: "triangle" },
});

// Zone 3 — cloud
s.addShape("cloud", {
  x: 6.95, y: 1.9, w: 2.5, h: 1.65,
  fill: { color: C.card }, line: { color: C.blueSoft, width: 1 },
});
s.addText("Cloud services", {
  x: 7.05, y: 2.42, w: 2.3, h: 0.28, fontFace: F.body, fontSize: 12, bold: true,
  color: C.ink, align: "center", margin: 0,
});
s.addText("analytics \u00b7 dashboards \u00b7 remote control", {
  x: 7.1, y: 2.7, w: 2.2, h: 0.36, fontFace: F.body, fontSize: 8.5,
  color: C.inkSoft, align: "center", margin: 0,
});

// flow annotation under the diagram
s.addText("telemetry flows up \u00b7 commands flow down \u2014 every hop is network traffic", {
  x: 3.6, y: 3.6, w: 5.85, h: 0.24, fontFace: F.body, fontSize: 9.5, italic: true,
  color: C.muted, align: "center", margin: 0,
});

// ==================== three defining characteristics ========================
const traits = [
  ["Sense \u2192 decide \u2192 act", "The physical world sits inside the control loop \u2014 a hijacked device has physical consequences.", C.green],
  ["MCU-class hardware", "Kilobytes of RAM, battery budgets \u2014 no room for antivirus or endpoint agents.", C.gold],
  ["Heterogeneous, always on", "Dozens of vendors and protocols \u2014 and every one of them has a path to the Internet.", C.blue],
];
let x = 0.55;
for (const [t, d, color] of traits) {
  H.card(s, x, 3.95, 2.86, 0.94);
  s.addShape("rect", { x, y: 3.95, w: 2.86, h: 0.055, fill: { color }, line: { type: "none" } });
  s.addText(t, { x: x + 0.15, y: 4.06, w: 2.58, h: 0.26, fontFace: F.body, fontSize: 11.5, bold: true, color: C.ink, margin: 0, valign: "top" });
  s.addText(d, { x: x + 0.15, y: 4.34, w: 2.58, h: 0.52, fontFace: F.body, fontSize: 9, color: C.inkSoft, margin: 0, valign: "top" });
  x += 3.02;
}

// foreshadow line (tees up the attack-surface slide)
s.addText([
  { text: "\u21d2  Keep this picture in mind: ", options: { bold: true, color: C.ink } },
  { text: "everything is reachable over the network \u2014 almost nothing defends itself.", options: { color: C.inkSoft } },
], { x: 0.55, y: 4.99, w: 8.9, h: 0.28, fontFace: F.body, fontSize: 11.5, margin: 0 });

// --- speaker notes (pt-BR) --------------------------------------------------
s.addNotes(
  "SLIDE-PONTE (~70 s). Antes de falarmos de vulnerabilidades, vamos concordar sobre o que \u00e9, concretamente, uma rede IoT. " +
  "S\u00e3o objetos do cotidiano \u2014 c\u00e2meras, sensores de presen\u00e7a, fechaduras inteligentes, termostatos, caixas de som, vest\u00edveis \u2014 " +
  "equipados com sensores, um pouco de computa\u00e7\u00e3o e uma interface de rede. A organiza\u00e7\u00e3o t\u00edpica tem tr\u00eas camadas. " +
  "Na borda, as \u201ccoisas\u201d: dispositivos restritos, baratos e sem fio, que sentem e agem sobre o mundo f\u00edsico. " +
  "No meio, um gateway ou hub, que agrega o tr\u00e1fego e traduz protocolos \u2014 Zigbee, BLE e Wi-Fi do lado dos dispositivos; " +
  "MQTT e HTTP sobre a Internet do outro lado. E na ponta, os servi\u00e7os de nuvem: an\u00e1lise, pain\u00e9is e controle remoto. " +
  "A telemetria sobe, os comandos descem \u2014 e cada salto desse caminho \u00e9 tr\u00e1fego de rede. " +
  "Tr\u00eas caracter\u00edsticas definem essas redes e importam para seguran\u00e7a. Primeira: o mundo f\u00edsico est\u00e1 dentro do la\u00e7o de " +
  "controle \u2014 um dispositivo comprometido tem consequ\u00eancia f\u00edsica, n\u00e3o apenas digital. Segunda: o hardware \u00e9 classe " +
  "microcontrolador \u2014 kilobytes de RAM e or\u00e7amento de bateria; n\u00e3o h\u00e1 espa\u00e7o para antiv\u00edrus nem agente de endpoint. " +
  "Terceira: o conjunto \u00e9 heterog\u00eaneo e est\u00e1 sempre conectado \u2014 dezenas de fabricantes e protocolos, todos com um caminho " +
  "at\u00e9 a Internet. Guardem esta imagem: cada elemento aqui \u00e9 alcan\u00e7\u00e1vel pela rede \u2014 e quase nenhum consegue se defender " +
  "sozinho. \u00c9 exatamente isso que o pr\u00f3ximo slide quantifica."
);

pres.writeFile({ fileName: "slide_iot_network.pptx" }).then(() => {
  console.log("OK slide_iot_network.pptx written");
});
