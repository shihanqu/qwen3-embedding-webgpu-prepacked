export interface WebGPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
}

export async function requestWebGPUDevice(): Promise<WebGPUContext> {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter was found');
  if (!adapter.features.has('shader-f16')) throw new Error('The WebGPU adapter does not expose shader-f16');
  const requiredLimits: Record<string, number> = {};
  const desiredStorageBinding = Math.min(adapter.limits.maxStorageBufferBindingSize, 512 * 2 ** 20);
  requiredLimits.maxStorageBufferBindingSize = desiredStorageBinding;
  requiredLimits.maxBufferSize = Math.min(adapter.limits.maxBufferSize, 512 * 2 ** 20);
  requiredLimits.maxComputeWorkgroupStorageSize = adapter.limits.maxComputeWorkgroupStorageSize;
  const requiredFeatures: GPUFeatureName[] = ['shader-f16'];
  if (adapter.features.has('subgroups')) requiredFeatures.push('subgroups');
  if (adapter.features.has('timestamp-query')) requiredFeatures.push('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits,
  });
  return { adapter, device };
}

export function createBufferWithData(
  device: GPUDevice,
  data: ArrayBufferView,
  usage: GPUBufferUsageFlags,
  label?: string,
): GPUBuffer {
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buffer = device.createBuffer({ size, usage, mappedAtCreation: true, label });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}
