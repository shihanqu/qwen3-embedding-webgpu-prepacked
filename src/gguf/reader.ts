import {
  GGMLType,
  GGUFValueType,
  type GGUFMetadataValue,
  type GGUFModel,
  type GGUFScalar,
  type GGUFTensorInfo,
} from './types.ts';

const textDecoder = new TextDecoder();
const GGUF_MAGIC = 0x46554747;

function checkedNumber(value: bigint, label: string): number {
  const number = Number(value);
  if (!Number.isSafeInteger(number)) throw new Error(`${label} exceeds JavaScript's safe integer range`);
  return number;
}

export function tensorByteLength(type: GGMLType, elements: number): number {
  switch (type) {
    case GGMLType.F32: return elements * 4;
    case GGMLType.F16: return elements * 2;
    case GGMLType.Q4_0:
      if (elements % 32 !== 0) throw new Error(`Q4_0 tensor has ${elements} elements, not divisible by 32`);
      return elements / 32 * 18;
    case GGMLType.Q4_K:
      if (elements % 256 !== 0) throw new Error(`Q4_K tensor has ${elements} elements, not divisible by 256`);
      return elements / 256 * 144;
    case GGMLType.Q6_K:
      if (elements % 256 !== 0) throw new Error(`Q6_K tensor has ${elements} elements, not divisible by 256`);
      return elements / 256 * 210;
    default:
      throw new Error(`unsupported GGML tensor type ${type}`);
  }
}

export interface GGUFReaderOptions {
  /** Metadata to retain. All fields are still traversed so tensor offsets remain correct. */
  metadataKeys?: ReadonlySet<string>;
}

export class GGUFReader {
  readonly #view: DataView;
  #offset = 0;

  constructor(readonly buffer: ArrayBuffer) {
    this.#view = new DataView(buffer);
  }

  parse(options: GGUFReaderOptions = {}): GGUFModel {
    if (this.u32() !== GGUF_MAGIC) throw new Error('not a GGUF file');
    const version = this.u32();
    if (version < 2 || version > 3) throw new Error(`unsupported GGUF version ${version}`);
    const tensorCount = this.u64Number('tensor count');
    const metadataCount = this.u64Number('metadata count');
    const metadata = new Map<string, GGUFMetadataValue>();

    for (let index = 0; index < metadataCount; index += 1) {
      const key = this.string();
      const type = this.u32() as GGUFValueType;
      const retain = !options.metadataKeys || options.metadataKeys.has(key);
      const value = this.value(type, retain);
      if (retain) metadata.set(key, value as GGUFMetadataValue);
    }

    const tensorHeaders: Array<Omit<GGUFTensorInfo, 'byteOffset' | 'byteLength'>> = [];
    for (let index = 0; index < tensorCount; index += 1) {
      const name = this.string();
      const dimensionCount = this.u32();
      const dimensions = Array.from({ length: dimensionCount }, () => this.u64Number('tensor dimension'));
      const type = this.u32() as GGMLType;
      const offset = this.u64Number('tensor offset');
      tensorHeaders.push({
        name,
        dimensions,
        type,
        offset,
        elementCount: dimensions.reduce((product, dimension) => product * dimension, 1),
      });
    }

    const alignmentValue = metadata.get('general.alignment');
    const alignment = typeof alignmentValue === 'number' ? alignmentValue : 32;
    const dataOffset = Math.ceil(this.#offset / alignment) * alignment;
    const tensors = new Map<string, GGUFTensorInfo>();
    for (const tensor of tensorHeaders) {
      const byteLength = tensorByteLength(tensor.type, tensor.elementCount);
      const byteOffset = dataOffset + tensor.offset;
      if (byteOffset + byteLength > this.buffer.byteLength) {
        throw new Error(`${tensor.name} extends past end of GGUF file`);
      }
      tensors.set(tensor.name, { ...tensor, byteOffset, byteLength });
    }

    return { version, alignment, metadata, tensors, dataOffset, buffer: this.buffer };
  }

  private value(type: GGUFValueType, retain: boolean): GGUFMetadataValue | undefined {
    switch (type) {
      case GGUFValueType.UINT8: return this.scalar(1, retain, () => this.#view.getUint8(this.#offset));
      case GGUFValueType.INT8: return this.scalar(1, retain, () => this.#view.getInt8(this.#offset));
      case GGUFValueType.UINT16: return this.scalar(2, retain, () => this.#view.getUint16(this.#offset, true));
      case GGUFValueType.INT16: return this.scalar(2, retain, () => this.#view.getInt16(this.#offset, true));
      case GGUFValueType.UINT32: return this.scalar(4, retain, () => this.#view.getUint32(this.#offset, true));
      case GGUFValueType.INT32: return this.scalar(4, retain, () => this.#view.getInt32(this.#offset, true));
      case GGUFValueType.FLOAT32: return this.scalar(4, retain, () => this.#view.getFloat32(this.#offset, true));
      case GGUFValueType.BOOL: return this.scalar(1, retain, () => this.#view.getUint8(this.#offset) !== 0);
      case GGUFValueType.STRING: {
        const value = this.string();
        return retain ? value : undefined;
      }
      case GGUFValueType.UINT64: {
        const value = this.#view.getBigUint64(this.#offset, true);
        this.#offset += 8;
        return retain ? value : undefined;
      }
      case GGUFValueType.INT64: {
        const value = this.#view.getBigInt64(this.#offset, true);
        this.#offset += 8;
        return retain ? value : undefined;
      }
      case GGUFValueType.FLOAT64: return this.scalar(8, retain, () => this.#view.getFloat64(this.#offset, true));
      case GGUFValueType.ARRAY: {
        const elementType = this.u32() as GGUFValueType;
        const length = this.u64Number('metadata array length');
        const result = retain ? new Array<GGUFScalar>(length) : undefined;
        for (let index = 0; index < length; index += 1) {
          const item = this.value(elementType, retain);
          if (result) result[index] = item as GGUFScalar;
        }
        return result;
      }
      default: throw new Error(`unsupported GGUF metadata value type ${type}`);
    }
  }

  private scalar<T extends GGUFScalar>(bytes: number, retain: boolean, read: () => T): T | undefined {
    const value = retain ? read() : undefined;
    this.#offset += bytes;
    return value;
  }

  private string(): string {
    const length = this.u64Number('string length');
    const start = this.#offset;
    this.#offset += length;
    return textDecoder.decode(new Uint8Array(this.buffer, start, length));
  }

  private u32(): number {
    const value = this.#view.getUint32(this.#offset, true);
    this.#offset += 4;
    return value;
  }

  private u64Number(label: string): number {
    const value = this.#view.getBigUint64(this.#offset, true);
    this.#offset += 8;
    return checkedNumber(value, label);
  }
}

export const QWEN3_METADATA_KEYS = new Set([
  'general.architecture',
  'general.name',
  'general.alignment',
  'qwen3.block_count',
  'qwen3.context_length',
  'qwen3.embedding_length',
  'qwen3.feed_forward_length',
  'qwen3.attention.head_count',
  'qwen3.attention.head_count_kv',
  'qwen3.rope.freq_base',
  'qwen3.attention.layer_norm_rms_epsilon',
  'qwen3.pooling_type',
  'tokenizer.ggml.eos_token_id',
  'tokenizer.ggml.add_eos_token',
]);
