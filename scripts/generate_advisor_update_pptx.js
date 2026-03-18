const pptxgen = require("pptxgenjs");

async function main() {
const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "OpenAI Codex";
pptx.company = "OpenAI";
pptx.subject = "LCTSCap advisor update";
pptx.title = "LCTSCap Advisor Update";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  lang: "zh-CN",
};

const C = {
  midnight: "21295C",
  ocean: "065A82",
  teal: "1C7293",
  mint: "6FD3C1",
  sand: "F5F1E8",
  cream: "FBF9F4",
  ink: "1F2937",
  slate: "506071",
  coral: "E76F51",
  moss: "5F8B4C",
  gold: "C89B3C",
  white: "FFFFFF",
  pale: "E8F2F7",
  soft: "DCE8EE",
};

function addFooter(slide, text, dark = false) {
  slide.addText(text, {
    x: 0.55,
    y: 5.12,
    w: 8.6,
    h: 0.2,
    fontFace: "Microsoft YaHei",
    fontSize: 8.5,
    color: dark ? "D7E7F0" : "687684",
    margin: 0,
    align: "left",
  });
}

function addTitle(slide, title, subtitle = "") {
  slide.addText(title, {
    x: 0.55,
    y: 0.34,
    w: 5.9,
    h: 0.5,
    fontFace: "Microsoft YaHei",
    fontSize: 24,
    bold: true,
    color: C.midnight,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.55,
      y: 0.86,
      w: 7.9,
      h: 0.3,
      fontFace: "Microsoft YaHei",
      fontSize: 10.5,
      color: C.slate,
      margin: 0,
    });
  }
}

function pill(slide, text, x, y, w, fill, color = C.white) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h: 0.3,
    rectRadius: 0.06,
    line: { color: fill, transparency: 100 },
    fill: { color: fill },
  });
  slide.addText(text, {
    x, y: y + 0.03, w, h: 0.18,
    fontFace: "Microsoft YaHei",
    fontSize: 9,
    bold: true,
    color,
    align: "center",
    margin: 0,
  });
}

function statCard(slide, { x, y, w, h, num, label, fill, accent }) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.08,
    line: { color: fill, transparency: 100 },
    fill: { color: fill },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 0.18, y: y + 0.18, w: 0.09, h: h - 0.36,
    line: { color: accent, transparency: 100 },
    fill: { color: accent },
  });
  slide.addText(num, {
    x: x + 0.42, y: y + 0.22, w: w - 0.58, h: 0.38,
    fontFace: "Aptos",
    fontSize: 24,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  slide.addText(label, {
    x: x + 0.42, y: y + 0.68, w: w - 0.58, h: 0.34,
    fontFace: "Microsoft YaHei",
    fontSize: 10,
    color: C.slate,
    margin: 0,
  });
}

function bulletList(slide, items, x, y, w, h, color = C.ink, fontSize = 13) {
  const arr = [];
  items.forEach((item, idx) => {
    arr.push({
      text: item,
      options: {
        bullet: { indent: 12 },
        breakLine: idx < items.length - 1,
        color,
      },
    });
  });
  slide.addText(arr, {
    x, y, w, h,
    fontFace: "Microsoft YaHei",
    fontSize,
    margin: 0,
    breakLine: false,
    valign: "top",
  });
}

function sectionCard(slide, opts) {
  const { x, y, w, h, title, body, accent, bg = C.white, titleColor = C.ink } = opts;
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.08,
    line: { color: "D9E2E8", width: 1 },
    fill: { color: bg },
    shadow: { type: "outer", color: "9AA7B2", blur: 1, angle: 45, offset: 1, opacity: 0.12 },
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: x + 0.18, y: y + 0.17, w: 0.26, h: 0.26,
    rectRadius: 0.13,
    line: { color: accent, transparency: 100 },
    fill: { color: accent },
  });
  slide.addText(title, {
    x: x + 0.54, y: y + 0.12, w: w - 0.7, h: 0.26,
    fontFace: "Microsoft YaHei",
    fontSize: 13.5,
    bold: true,
    color: titleColor,
    margin: 0,
  });
  slide.addText(body, {
    x: x + 0.18, y: y + 0.5, w: w - 0.36, h: h - 0.62,
    fontFace: "Microsoft YaHei",
    fontSize: 10.5,
    color: C.slate,
    margin: 0,
    valign: "top",
  });
}

function flowBox(slide, x, y, w, h, num, title, body, fill) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: 0.08,
    line: { color: fill, width: 1.5 },
    fill: { color: C.white },
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: x + 0.16, y: y + 0.15, w: 0.34, h: 0.34,
    rectRadius: 0.17,
    line: { color: fill, transparency: 100 },
    fill: { color: fill },
  });
  slide.addText(String(num), {
    x: x + 0.16, y: y + 0.19, w: 0.34, h: 0.12,
    fontFace: "Aptos",
    fontSize: 12,
    bold: true,
    color: C.white,
    margin: 0,
    align: "center",
  });
  slide.addText(title, {
    x: x + 0.58, y: y + 0.14, w: w - 0.74, h: 0.18,
    fontFace: "Microsoft YaHei",
    fontSize: 12.5,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  slide.addText(body, {
    x: x + 0.16, y: y + 0.56, w: w - 0.28, h: h - 0.68,
    fontFace: "Microsoft YaHei",
    fontSize: 10,
    color: C.slate,
    margin: 0,
    valign: "top",
  });
}

function arrow(slide, x, y, w, color) {
  slide.addShape(pptx.ShapeType.chevron, {
    x, y, w, h: 0.24,
    line: { color, transparency: 100 },
    fill: { color },
  });
}

// Slide 1
{
  const slide = pptx.addSlide();
  slide.background = { color: C.midnight };

  slide.addShape(pptx.ShapeType.rect, {
    x: 0, y: 0, w: 10, h: 5.625,
    line: { color: C.midnight, transparency: 100 },
    fill: { color: C.midnight },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.95, y: 0, w: 3.05, h: 5.625,
    line: { color: C.ocean, transparency: 100 },
    fill: { color: C.ocean },
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 7.3, y: 0.7, w: 2.1, h: 1.2,
    rectRadius: 0.08,
    line: { color: C.mint, transparency: 100 },
    fill: { color: C.mint, transparency: 12 },
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 7.55, y: 2.2, w: 1.6, h: 1.6,
    rectRadius: 0.1,
    line: { color: C.white, transparency: 70, width: 1 },
    fill: { color: C.white, transparency: 82 },
  });

  slide.addText("LCTSCap", {
    x: 0.6, y: 0.7, w: 5.2, h: 0.5,
    fontFace: "Aptos Display",
    fontSize: 28,
    bold: true,
    color: C.white,
    margin: 0,
  });
  slide.addText("Hierarchical and Verifiable Long-Context Time Series Captioning", {
    x: 0.6, y: 1.35, w: 5.7, h: 1.05,
    fontFace: "Microsoft YaHei",
    fontSize: 21,
    bold: true,
    color: "EAF4FB",
    margin: 0,
    valign: "mid",
  });
  slide.addText("导师汇报版本：研究定位、方法框架、当前实验进度", {
    x: 0.62, y: 2.45, w: 4.8, h: 0.28,
    fontFace: "Microsoft YaHei",
    fontSize: 11,
    color: "C5D8E4",
    margin: 0,
  });

  pill(slide, "主数据集 CAPTURE-24", 0.6, 3.02, 1.65, C.teal);
  pill(slide, "外部验证 HARTH", 2.38, 3.02, 1.45, C.coral);
  pill(slide, "Phase 2 正在训练", 3.97, 3.02, 1.58, C.moss);

  statCard(slide, {
    x: 0.62, y: 3.62, w: 2.45, h: 1.12,
    num: "128/256/512",
    label: "semantic tokens\n≈ 21 / 43 / 85 分钟上下文",
    fill: "F1F8FA",
    accent: C.mint,
  });
  statCard(slide, {
    x: 3.28, y: 3.62, w: 2.45, h: 1.12,
    num: "30.54",
    label: "Phase 1 best val loss\nencoder + planner + aligner",
    fill: "F8F4EE",
    accent: C.gold,
  });
  addFooter(slide, "LCTSCap advisor update | 2026-03-18 | 目标：CCF-A 方向的长上下文时序解释生成", true);
}

// Slide 2
{
  const slide = pptx.addSlide();
  slide.background = { color: C.cream };
  addTitle(slide, "研究主题与论文核心问题", "这个项目要解决的不只是“生成一句 caption”，而是长上下文时序的结构化解释生成。");

  sectionCard(slide, {
    x: 0.55, y: 1.35, w: 4.15, h: 1.22,
    title: "Research Topic",
    body: "面向长时间范围行为/活动序列，生成层次化、可验证、可追溯的自然语言描述，使模型能从“识别”走向“解释”。",
    accent: C.ocean,
    bg: C.white,
  });
  sectionCard(slide, {
    x: 5.0, y: 1.35, w: 4.45, h: 1.22,
    title: "Paper Positioning",
    body: "定位在时序理解 + 文本生成 + 可信评测的交叉点，不把贡献停留在普通 caption，而强调 hierarchy / grounding / alignment。",
    accent: C.coral,
    bg: C.white,
  });

  sectionCard(slide, {
    x: 0.55, y: 2.82, w: 2.15, h: 1.5,
    title: "Q1 长上下文表示",
    body: "如何把 raw signals 转成可供语言生成使用的 semantic tokens，而不是直接塞给 LLM。",
    accent: C.teal,
    bg: "F2FAFB",
  });
  sectionCard(slide, {
    x: 2.9, y: 2.82, w: 2.15, h: 1.5,
    title: "Q2 层次化生成",
    body: "如何同时输出事件表、片段摘要、全局描述，而不是只有一段平面文本。",
    accent: C.moss,
    bg: "F5FAF1",
  });
  sectionCard(slide, {
    x: 5.25, y: 2.82, w: 2.15, h: 1.5,
    title: "Q3 可验证性",
    body: "如何确认文本内容被原始时序支持，尤其是活动、顺序、持续时间、跨度。",
    accent: C.gold,
    bg: "FBF7EF",
  });
  sectionCard(slide, {
    x: 7.6, y: 2.82, w: 1.85, h: 1.5,
    title: "Q4 模态对齐",
    body: "如何形成 TS-text 的共享空间，支持 retrieval 与 reranking。",
    accent: C.midnight,
    bg: "EEF1FA",
  });

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.55, y: 4.55, w: 8.9, h: 0.62,
    rectRadius: 0.06,
    line: { color: "D6E0E6", width: 1 },
    fill: { color: "F8FBFC" },
  });
  slide.addText("论文主卖点：long-context + hierarchical planning + verifiable caption + retrieval-aligned representation", {
    x: 0.75, y: 4.75, w: 8.5, h: 0.16,
    fontFace: "Aptos",
    fontSize: 12,
    bold: true,
    color: C.midnight,
    margin: 0,
    align: "center",
  });
  addFooter(slide, "关键词：Long context / Hierarchy / Verifiability / Alignment");
}

// Slide 3
{
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  addTitle(slide, "方法框架与整体流程", "从数据到输出的主链路：把长时间序列压缩成可解释、可检索、可生成的分层表示。");

  flowBox(slide, 0.55, 1.35, 1.72, 1.38, 1, "预处理与 token 化", "50Hz 信号 -> 10 秒窗口 -> 128/256/512 semantic tokens", C.ocean);
  arrow(slide, 2.35, 1.92, 0.28, C.ocean);
  flowBox(slide, 2.72, 1.35, 1.72, 1.38, 2, "Local Encoder", "对每个窗口做 patch 编码，输出窗口级表示", C.teal);
  arrow(slide, 4.52, 1.92, 0.28, C.teal);
  flowBox(slide, 4.89, 1.35, 1.72, 1.38, 3, "Hierarchical Planner", "token-level + segment-level 建模，得到 H_token / H_seg", C.moss);
  arrow(slide, 6.69, 1.92, 0.28, C.moss);
  flowBox(slide, 7.06, 1.35, 2.1, 1.38, 4, "Event + Decoder", "事件提议、检索对齐、caption 生成", C.coral);

  sectionCard(slide, {
    x: 0.55, y: 3.05, w: 2.75, h: 1.55,
    title: "输入",
    body: "CAPTURE-24 / HARTH\n单样本 = 多窗口连续长上下文\nsubject-level split 防止泄漏",
    accent: C.ocean,
    bg: "F3FAFC",
  });
  sectionCard(slide, {
    x: 3.45, y: 3.05, w: 2.1, h: 1.55,
    title: "中间层表示",
    body: "Event table\nSegment summaries\nGlobal caption supervision",
    accent: C.teal,
    bg: "F3FAFC",
  });
  sectionCard(slide, {
    x: 5.7, y: 3.05, w: 1.8, h: 1.55,
    title: "训练目标",
    body: "L_cap\nL_align\nL_event\nL_coverage",
    accent: C.gold,
    bg: "FBF8F1",
  });
  sectionCard(slide, {
    x: 7.65, y: 3.05, w: 1.8, h: 1.55,
    title: "输出",
    body: "全局描述\n段摘要\n证据可追溯",
    accent: C.midnight,
    bg: "F0F3FA",
  });

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.55, y: 4.86, w: 8.9, h: 0.32,
    rectRadius: 0.05,
    line: { color: C.soft, transparency: 100 },
    fill: { color: C.soft },
  });
  slide.addText("训练课程：Phase 0 模板基线 -> Phase 1 对齐与事件建模 -> Phase 2 caption decoder -> Phase 3 paraphrase + ctx512 -> Phase 4 LLM bridge", {
    x: 0.72, y: 4.94, w: 8.55, h: 0.14,
    fontFace: "Microsoft YaHei",
    fontSize: 10,
    color: C.ink,
    margin: 0,
    align: "center",
  });
  addFooter(slide, "方法关键词：Semantic tokenization / Event grounding / Cross-modal retrieval / Autoregressive decoding");
}

// Slide 4
{
  const slide = pptx.addSlide();
  slide.background = { color: C.cream };
  addTitle(slide, "当前工程进度与阶段性结果", "工程已经越过“规格设计”阶段，数据、模板基线、Phase 1 和 Phase 2 启动都已落地。");

  slide.addShape(pptx.ShapeType.line, {
    x: 0.95, y: 1.55, w: 8.1, h: 0,
    line: { color: "AFC6D2", width: 2.2 },
  });

  const phases = [
    ["P0", "模板基线", "已完成", C.moss, 0.95],
    ["P1", "Encoder/Planner/Aligner", "已完成", C.moss, 2.95],
    ["P2", "Caption Decoder", "进行中", C.coral, 5.05],
    ["P3", "Paraphrase + ctx512", "待开始", C.gold, 7.15],
  ];
  phases.forEach(([p, name, status, color, x]) => {
    slide.addShape(pptx.ShapeType.roundRect, {
      x, y: 1.25, w: 0.56, h: 0.56,
      rectRadius: 0.28,
      line: { color, transparency: 100 },
      fill: { color },
    });
    slide.addText(p, {
      x, y: 1.43, w: 0.56, h: 0.12,
      fontFace: "Aptos",
      fontSize: 12,
      bold: true,
      color: C.white,
      margin: 0,
      align: "center",
    });
    slide.addText(name, {
      x: x - 0.1, y: 1.92, w: 1.45, h: 0.18,
      fontFace: "Microsoft YaHei",
      fontSize: 10.5,
      bold: true,
      color: C.ink,
      margin: 0,
      align: "center",
    });
    slide.addText(status, {
      x: x - 0.02, y: 2.18, w: 1.2, h: 0.16,
      fontFace: "Microsoft YaHei",
      fontSize: 9.2,
      color,
      margin: 0,
      align: "center",
    });
  });

  statCard(slide, {
    x: 0.55, y: 2.8, w: 2.15, h: 1.16,
    num: "190,704",
    label: "CAPTURE-24 长上下文样本\n1,398,027 个基础窗口",
    fill: "F1F8FA",
    accent: C.ocean,
  });
  statCard(slide, {
    x: 2.95, y: 2.8, w: 2.15, h: 1.16,
    num: "30.54",
    label: "Phase 1 best val loss\nbest checkpoint 已保存",
    fill: "F8F4EE",
    accent: C.gold,
  });
  statCard(slide, {
    x: 5.35, y: 2.8, w: 1.85, h: 1.16,
    num: "0.83",
    label: "Phase 1 best align loss\nTS-text 对齐改善",
    fill: "F4F8F2",
    accent: C.moss,
  });
  statCard(slide, {
    x: 7.45, y: 2.8, w: 2.0, h: 1.16,
    num: "进行中",
    label: "Phase 2 运行状态\nGPU 2 上持续训练",
    fill: "F9F2EF",
    accent: C.coral,
  });

  sectionCard(slide, {
    x: 0.55, y: 4.2, w: 4.2, h: 0.92,
    title: "Phase 2 最新观测",
    body: "ctx=128 已完成 epoch 0，平均 total loss=19.79；caption loss 已明显下降，decoder 学习正常。",
    accent: C.teal,
    bg: C.white,
  });
  sectionCard(slide, {
    x: 5.0, y: 4.2, w: 4.45, h: 0.92,
    title: "当前判断",
    body: "长上下文下 caption 学得快，event head 仍是主要难点；目前未出现新的 NaN / OOM / device assert。",
    accent: C.coral,
    bg: C.white,
  });
  addFooter(slide, "结果摘要：工程已从“搭框架”进入“持续训练 + 准备主表”的阶段");
}

// Slide 5
{
  const slide = pptx.addSlide();
  slide.background = { color: C.midnight };

  slide.addText("风险判断与下一步计划", {
    x: 0.55, y: 0.36, w: 4.6, h: 0.42,
    fontFace: "Microsoft YaHei",
    fontSize: 23,
    bold: true,
    color: C.white,
    margin: 0,
  });
  slide.addText("目前最重要的是把 Phase 2 跑稳、拿到主表，再进入 paraphrase 与更长上下文。", {
    x: 0.55, y: 0.82, w: 6.1, h: 0.22,
    fontFace: "Microsoft YaHei",
    fontSize: 10.5,
    color: "C8DBE6",
    margin: 0,
  });

  sectionCard(slide, {
    x: 0.55, y: 1.45, w: 2.75, h: 2.2,
    title: "主要风险",
    body: "1. 长上下文下 event loss 仍高。\n2. Phase 3 的 mixed caption source 还未进入主实验闭环。\n3. 评测仍需更强的论文级验证报告与案例分析。",
    accent: C.coral,
    bg: "F6F2F1",
  });
  sectionCard(slide, {
    x: 3.55, y: 1.45, w: 2.75, h: 2.2,
    title: "短期目标",
    body: "1. 盯完 Phase 2。\n2. 抽取首个有效 checkpoint。\n3. 补 baseline / ablation 主表。\n4. 对 HARTH 做外部验证。",
    accent: C.mint,
    bg: "F1F8F7",
  });
  sectionCard(slide, {
    x: 6.55, y: 1.45, w: 2.9, h: 2.2,
    title: "汇报建议",
    body: "向导师强调：这不是普通 caption，而是面向长时程信号的层次化解释生成；当前方法已经具备清晰论文结构与可执行实验路径。",
    accent: C.gold,
    bg: "F8F5EE",
  });

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.55, y: 4.15, w: 8.9, h: 0.72,
    rectRadius: 0.08,
    line: { color: "58708A", transparency: 100 },
    fill: { color: "2C3C75" },
  });
  slide.addText("一句话总结：LCTSCap 已完成数据与 Phase 1，Phase 2 已实质启动；下一阶段重点是把“生成能力”和“事件 grounding”一起做强，形成可投稿的主实验闭环。", {
    x: 0.8, y: 4.36, w: 8.4, h: 0.22,
    fontFace: "Microsoft YaHei",
    fontSize: 12,
    bold: true,
    color: C.white,
    margin: 0,
    align: "center",
  });
  addFooter(slide, "Advisor update | LCTSCap | 2026-03-18", true);
}

await pptx.writeFile({ fileName: "advisor_update_deck_v2.pptx" });
console.log("Wrote advisor_update_deck_v2.pptx");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
