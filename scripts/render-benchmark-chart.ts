import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

interface Row {
  tokens: number;
  webgpuMedian: { singleRps: number; concurrency16AggregateRps: number };
  lmStudio: { singleRps: number; concurrency16AggregateRps: number };
}

interface BenchmarkData {
  date: string;
  hardware: { chip: string; gpu: string; memory: string };
  rows: Row[];
}

const inputPath = resolve('docs/benchmarks/2026-07-15-webgpu-vs-lm-studio-m3-max.json');
const outputPath = resolve('docs/lm-studio-comparison.svg');
const data = JSON.parse(readFileSync(inputPath, 'utf8')) as BenchmarkData;
const width = 1120;
const height = 610;
const left = { x: 80, y: 120, width: 430, height: 340 };
const right = { x: 650, y: 120, width: 390, height: 340 };
const singleMax = 70;
const concurrentMax = 280;

const text = (x: number, y: number, value: string, className = 'label', anchor = 'start') =>
  `<text x="${x}" y="${y}" class="${className}" text-anchor="${anchor}">${value}</text>`;

function grid(panel: typeof left, maximum: number, ticks: number[]): string {
  return ticks.map((tick) => {
    const x = panel.x + (tick / maximum) * panel.width;
    return `<line x1="${x}" y1="${panel.y}" x2="${x}" y2="${panel.y + panel.height}" class="grid"/>${text(x, panel.y + panel.height + 28, String(tick), 'tick', 'middle')}`;
  }).join('');
}

const rowHeight = 76;
const marks = data.rows.map((row, index) => {
  const y = left.y + index * rowHeight + 12;
  const lmSingleWidth = (row.lmStudio.singleRps / singleMax) * left.width;
  const gpuSingleWidth = (row.webgpuMedian.singleRps / singleMax) * left.width;
  const lmConcurrentWidth = (row.lmStudio.concurrency16AggregateRps / concurrentMax) * right.width;
  const gpuConcurrentWidth = (row.webgpuMedian.concurrency16AggregateRps / concurrentMax) * right.width;
  return `
    ${text(left.x - 16, y + 36, `${row.tokens} tok`, 'row-label', 'end')}
    <rect x="${left.x}" y="${y}" width="${lmSingleWidth}" height="22" rx="4" class="lm"/>
    <rect x="${left.x}" y="${y + 28}" width="${gpuSingleWidth}" height="22" rx="4" class="gpu"/>
    ${text(left.x + lmSingleWidth + 8, y + 16, row.lmStudio.singleRps.toFixed(2), 'value')}
    ${text(left.x + gpuSingleWidth + 8, y + 44, row.webgpuMedian.singleRps.toFixed(2), 'value')}
    <rect x="${right.x}" y="${y}" width="${lmConcurrentWidth}" height="22" rx="4" class="lm"/>
    <rect x="${right.x}" y="${y + 28}" width="${gpuConcurrentWidth}" height="22" rx="4" class="gpu"/>
    ${text(right.x + lmConcurrentWidth + 8, y + 16, row.lmStudio.concurrency16AggregateRps.toFixed(2), 'value')}
    ${text(right.x + gpuConcurrentWidth + 8, y + 44, row.webgpuMedian.concurrency16AggregateRps.toFixed(2), 'value')}
  `;
}).join('');

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">Qwen3 Embedding 0.6B WebGPU versus LM Studio</title>
  <desc id="desc">Two horizontal bar charts compare WebGPU and LM Studio at single stream and 16 simultaneous requests for exact 15, 50, 150, and 500 token inputs on an Apple M3 Max.</desc>
  <rect width="${width}" height="${height}" fill="#0d1117"/>
  <style>
    .heading { font: 600 24px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #f0f6fc; }
    .subheading { font: 600 17px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #f0f6fc; }
    .label, .row-label, .value, .tick, .caption { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #b1bac4; }
    .row-label { font-size: 14px; font-weight: 600; }
    .value { font-size: 13px; font-weight: 600; }
    .tick, .caption { font-size: 13px; }
    .grid { stroke: #30363d; stroke-width: 1; }
    .lm { fill: #8b949e; }
    .gpu { fill: #58a6ff; }
  </style>
  ${text(50, 48, 'Qwen3 Embedding 0.6B · WebGPU vs LM Studio', 'heading')}
  ${text(50, 78, `${data.hardware.chip}, ${data.hardware.gpu}, ${data.hardware.memory}`, 'caption')}
  ${text(left.x, 105, 'Single-stream throughput (requests/second)', 'subheading')}
  ${text(right.x, 105, '16-concurrent aggregate throughput (requests/second)', 'subheading')}
  ${grid(left, singleMax, [0, 20, 40, 60])}
  ${grid(right, concurrentMax, [0, 70, 140, 210, 280])}
  ${marks}
  <rect x="80" y="510" width="16" height="16" rx="3" class="lm"/>${text(104, 523, 'LM Studio', 'caption')}
  <rect x="220" y="510" width="16" height="16" rx="3" class="gpu"/>${text(244, 523, 'WebGPU', 'caption')}
  ${text(50, 566, `Measured ${data.date}; exact token counts include EOS; WebGPU is the median of three warmed trials; no request errors.`, 'caption')}
</svg>\n`;

writeFileSync(outputPath, svg);
console.log(`Wrote ${outputPath}`);
