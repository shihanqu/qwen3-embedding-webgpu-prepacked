import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

interface Row {
  tokens: number;
  acceptance: boolean;
  webgpuSingleRps: number;
  lmStudioSingleRps: number;
  webgpuBatch16Rps: number;
  scaling: number;
}

interface BenchmarkData {
  date: string;
  hardware: { chip: string; gpu: string; memory: string };
  rows: Row[];
}

const inputPath = resolve('docs/benchmarks/2026-07-14-m3-max.json');
const outputPath = resolve('docs/lm-studio-comparison.svg');
const data = JSON.parse(readFileSync(inputPath, 'utf8')) as BenchmarkData;
const width = 1120;
const height = 610;
const left = { x: 80, y: 120, width: 430, height: 340 };
const right = { x: 650, y: 120, width: 390, height: 340 };
const singleMax = 60;
const scalingMax = 9;

const text = (x: number, y: number, value: string, className = 'label', anchor = 'start') =>
  `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}">${value}</text>`;

const singleGrid = [0, 15, 30, 45, 60].map((tick) => {
  const x = left.x + (tick / singleMax) * left.width;
  return `<line x1="${x}" y1="${left.y}" x2="${x}" y2="${left.y + left.height}" class="grid"/>${text(x, left.y + left.height + 28, String(tick), 'tick', 'middle')}`;
}).join('');

const scalingGrid = [0, 2, 4, 6, 8].map((tick) => {
  const x = right.x + (tick / scalingMax) * right.width;
  const target = tick === 4 ? ' target' : '';
  return `<line x1="${x}" y1="${right.y}" x2="${x}" y2="${right.y + right.height}" class="grid${target}"/>${text(x, right.y + right.height + 28, `${tick}×`, 'tick', 'middle')}`;
}).join('');

const rowHeight = 76;
const marks = data.rows.map((row, index) => {
  const y = left.y + index * rowHeight + 12;
  const lmWidth = (row.lmStudioSingleRps / singleMax) * left.width;
  const gpuWidth = (row.webgpuSingleRps / singleMax) * left.width;
  const scaleWidth = (row.scaling / scalingMax) * right.width;
  const badge = row.acceptance ? `<text x="${left.x - 16}" y="${y + 11}" class="acceptance" text-anchor="end">PASS</text>` : '';
  return `
    ${text(left.x - 16, y + 36, `${row.tokens} tok`, 'row-label', 'end')}
    ${badge}
    <rect x="${left.x}" y="${y}" width="${lmWidth}" height="22" rx="4" class="lm"/>
    <rect x="${left.x}" y="${y + 28}" width="${gpuWidth}" height="22" rx="4" class="gpu"/>
    ${text(left.x + lmWidth + 8, y + 16, row.lmStudioSingleRps.toFixed(2), 'value')}
    ${text(left.x + gpuWidth + 8, y + 44, row.webgpuSingleRps.toFixed(2), 'value')}
    <rect x="${right.x}" y="${y + 7}" width="${scaleWidth}" height="36" rx="5" class="scale"/>
    ${text(right.x + scaleWidth + 8, y + 31, `${row.scaling.toFixed(2)}×`, 'value')}
  `;
}).join('');

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">Qwen3 WebGPU versus LM Studio embedding throughput</title>
  <desc id="desc">Single-stream requests per second and WebGPU batch-16 scaling at 6, 17, 26, and 105 tokens on an Apple M3 Max. The 6-token acceptance workload is 1.82 times faster than LM Studio and scales 8.04 times.</desc>
  <style>
    .heading { font: 600 24px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #24292f; }
    .subheading { font: 600 17px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #24292f; }
    .label, .row-label, .value, .tick, .caption, .acceptance { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #57606a; }
    .row-label { font-size: 14px; font-weight: 600; }
    .value { font-size: 13px; font-weight: 600; }
    .tick, .caption { font-size: 13px; }
    .acceptance { font-size: 10px; font-weight: 700; fill: #1a7f37; }
    .grid { stroke: #d0d7de; stroke-width: 1; }
    .grid.target { stroke: #cf222e; stroke-width: 2; stroke-dasharray: 6 5; }
    .lm { fill: #6e7781; }
    .gpu { fill: #2f81f7; }
    .scale { fill: #a371f7; }
    @media (prefers-color-scheme: dark) {
      .heading, .subheading { fill: #f0f6fc; }
      .label, .row-label, .value, .tick, .caption { fill: #b1bac4; }
      .acceptance { fill: #3fb950; }
      .grid { stroke: #30363d; }
      .grid.target { stroke: #f85149; }
      .lm { fill: #8b949e; }
      .gpu { fill: #58a6ff; }
      .scale { fill: #bc8cff; }
    }
  </style>
  ${text(50, 48, 'Qwen3 Embedding 0.6B · WebGPU vs LM Studio', 'heading')}
  ${text(50, 78, `${data.hardware.chip}, ${data.hardware.gpu}, ${data.hardware.memory} · ${data.date}`, 'caption')}
  ${text(left.x, 105, 'Single-stream throughput (requests/second)', 'subheading')}
  ${text(right.x, 105, 'WebGPU aggregate scaling at 16 concurrent', 'subheading')}
  ${singleGrid}
  ${scalingGrid}
  ${marks}
  <rect x="80" y="510" width="16" height="16" rx="3" class="lm"/>${text(104, 523, 'LM Studio single', 'caption')}
  <rect x="245" y="510" width="16" height="16" rx="3" class="gpu"/>${text(269, 523, 'Custom WebGPU single', 'caption')}
  <rect x="445" y="510" width="16" height="16" rx="3" class="scale"/>${text(469, 523, 'WebGPU batch 16 ÷ single', 'caption')}
  <line x1="720" y1="518" x2="748" y2="518" class="grid target"/>${text(758, 523, '4× scaling target', 'caption')}
  ${text(50, 566, 'Acceptance: 6 exact tokenizer tokens. Warmed plans; identical text per implementation; LM Studio at 127.0.0.1:1234.', 'caption')}
</svg>\n`;

writeFileSync(outputPath, svg);
console.log(`Wrote ${outputPath}`);
