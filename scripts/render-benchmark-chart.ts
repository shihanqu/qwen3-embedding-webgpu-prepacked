import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

interface Row {
  tokens: number;
  webgpuSingleRps: number;
  webgpuBatch16Rps: number;
  webgpuScaling: number;
  lmStudioSingleRps: number;
  lmStudioConcurrent16Rps: number;
  lmStudioScaling: number;
}

interface BenchmarkData {
  webgpuDate: string;
  lmStudioConcurrencyDate: string;
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
const concurrentRatioMax = 16;

const text = (x: number, y: number, value: string, className = 'label', anchor = 'start') =>
  `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}">${value}</text>`;

const singleGrid = [0, 15, 30, 45, 60].map((tick) => {
  const x = left.x + (tick / singleMax) * left.width;
  return `<line x1="${x}" y1="${left.y}" x2="${x}" y2="${left.y + left.height}" class="grid"/>${text(x, left.y + left.height + 28, String(tick), 'tick', 'middle')}`;
}).join('');

const ratioGrid = [0, 5, 10, 15].map((tick) => {
  const x = right.x + (tick / concurrentRatioMax) * right.width;
  return `<line x1="${x}" y1="${right.y}" x2="${x}" y2="${right.y + right.height}" class="grid"/>${text(x, right.y + right.height + 28, `${tick}×`, 'tick', 'middle')}`;
}).join('');

const rowHeight = 76;
const marks = data.rows.map((row, index) => {
  const y = left.y + index * rowHeight + 12;
  const lmWidth = (row.lmStudioSingleRps / singleMax) * left.width;
  const gpuWidth = (row.webgpuSingleRps / singleMax) * left.width;
  const concurrentRatio = row.webgpuBatch16Rps / row.lmStudioConcurrent16Rps;
  const ratioWidth = (concurrentRatio / concurrentRatioMax) * right.width;
  return `
    ${text(left.x - 16, y + 36, `${row.tokens} tok`, 'row-label', 'end')}
    <rect x="${left.x}" y="${y}" width="${lmWidth}" height="22" rx="4" class="lm"/>
    <rect x="${left.x}" y="${y + 28}" width="${gpuWidth}" height="22" rx="4" class="gpu"/>
    ${text(left.x + lmWidth + 8, y + 16, row.lmStudioSingleRps.toFixed(2), 'value')}
    ${text(left.x + gpuWidth + 8, y + 44, row.webgpuSingleRps.toFixed(2), 'value')}
    <rect x="${right.x}" y="${y + 7}" width="${ratioWidth}" height="36" rx="5" class="ratio"/>
    ${text(right.x + ratioWidth + 8, y + 31, `${concurrentRatio.toFixed(2)}×`, 'value')}
  `;
}).join('');

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">Qwen3 WebGPU versus LM Studio embedding throughput</title>
  <desc id="desc">Single-stream requests per second and relative aggregate throughput with 16 simultaneous requests at 6, 17, 26, and 105 tokens on an Apple M3 Max.</desc>
  <style>
    .heading { font: 600 24px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #24292f; }
    .subheading { font: 600 17px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #24292f; }
    .label, .row-label, .value, .tick, .caption { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #57606a; }
    .row-label { font-size: 14px; font-weight: 600; }
    .value { font-size: 13px; font-weight: 600; }
    .tick, .caption { font-size: 13px; }
    .grid { stroke: #d0d7de; stroke-width: 1; }
    .lm { fill: #6e7781; }
    .gpu { fill: #2f81f7; }
    .ratio { fill: #a371f7; }
    @media (prefers-color-scheme: dark) {
      .heading, .subheading { fill: #f0f6fc; }
      .label, .row-label, .value, .tick, .caption { fill: #b1bac4; }
      .grid { stroke: #30363d; }
      .lm { fill: #8b949e; }
      .gpu { fill: #58a6ff; }
      .ratio { fill: #bc8cff; }
    }
  </style>
  ${text(50, 48, 'Qwen3 Embedding 0.6B · WebGPU vs LM Studio', 'heading')}
  ${text(50, 78, `${data.hardware.chip}, ${data.hardware.gpu}, ${data.hardware.memory}`, 'caption')}
  ${text(left.x, 105, 'Single-stream throughput (requests/second)', 'subheading')}
  ${text(right.x, 105, '16-concurrent throughput ratio (WebGPU / LM Studio)', 'subheading')}
  ${singleGrid}
  ${ratioGrid}
  ${marks}
  <rect x="80" y="510" width="16" height="16" rx="3" class="lm"/>${text(104, 523, 'LM Studio single', 'caption')}
  <rect x="245" y="510" width="16" height="16" rx="3" class="gpu"/>${text(269, 523, 'Custom WebGPU single', 'caption')}
  <rect x="445" y="510" width="16" height="16" rx="3" class="ratio"/>${text(469, 523, 'WebGPU aggregate ÷ LM Studio aggregate at 16 concurrent', 'caption')}
  ${text(50, 566, `WebGPU measured ${data.webgpuDate}; LM Studio 16-worker HTTP benchmark measured ${data.lmStudioConcurrencyDate}; no request errors.`, 'caption')}
</svg>\n`;

writeFileSync(outputPath, svg);
console.log(`Wrote ${outputPath}`);
