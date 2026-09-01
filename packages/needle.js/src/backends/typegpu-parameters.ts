import type { TgpuRoot } from "typegpu";
import * as d from "typegpu/data";
import type {
  AttentionParameters,
  KvParameters,
  QueryParameters,
  ResidentParameter,
  ResidentParameterFactory,
} from "../resident/parameters.js";

const QuerySchema = d.struct({
  position: d.u32,
  thetaBits: d.u32,
});

const KvSchema = d.struct({
  layer: d.u32,
  position: d.u32,
  allocation: d.u32,
  sinkLength: d.u32,
  window: d.u32,
  kvHeads: d.u32,
  thetaBits: d.u32,
  reserved: d.u32,
});

const AttentionSchema = d.struct({
  layer: d.u32,
  position: d.u32,
  allocation: d.u32,
  sinkLength: d.u32,
  window: d.u32,
  kvHeads: d.u32,
  heads: d.u32,
  reserved: d.u32,
});

export function createTypeGpuParameterFactory(root: TgpuRoot): ResidentParameterFactory {
  const query = (label: string): ResidentParameter<QueryParameters> => {
    const typed = root.createBuffer(QuerySchema).$usage("storage").$name(label);
    return {
      buffer: root.unwrap(typed),
      write: (value) => typed.write(value),
      destroy: () => typed.destroy(),
    };
  };
  const kv = (label: string): ResidentParameter<KvParameters> => {
    const typed = root.createBuffer(KvSchema).$usage("storage").$name(label);
    return {
      buffer: root.unwrap(typed),
      write: (value) => typed.write(value),
      destroy: () => typed.destroy(),
    };
  };
  const attention = (label: string): ResidentParameter<AttentionParameters> => {
    const typed = root.createBuffer(AttentionSchema).$usage("storage").$name(label);
    return {
      buffer: root.unwrap(typed),
      write: (value) => typed.write(value),
      destroy: () => typed.destroy(),
    };
  };
  return { query, kv, attention };
}

export const typeGpuParameterSchemas = {
  query: QuerySchema,
  kv: KvSchema,
  attention: AttentionSchema,
} as const;
