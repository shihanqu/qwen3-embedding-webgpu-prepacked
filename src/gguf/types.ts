export enum GGUFValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

export enum GGMLType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_K = 12,
  Q6_K = 14,
}

export type GGUFScalar = string | number | bigint | boolean;
export type GGUFMetadataValue = GGUFScalar | GGUFScalar[];

export interface GGUFTensorInfo {
  name: string;
  dimensions: number[];
  type: GGMLType;
  offset: number;
  byteOffset: number;
  byteLength: number;
  elementCount: number;
}

export interface GGUFModel {
  version: number;
  alignment: number;
  metadata: Map<string, GGUFMetadataValue>;
  tensors: Map<string, GGUFTensorInfo>;
  dataOffset: number;
  buffer: ArrayBuffer;
}
