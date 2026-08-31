export { NeedleError, type NeedleErrorCode } from "./errors.js";

export {
  NeedleModel,
  type BackendSelection,
  type LoadModelOptions,
  type GenerateOptions,
  type GeneratedToken,
  type GenerationResult,
} from "./model/model.js";
export {
  NeedleRuntime,
  type RuntimeOptions,
  type RuntimeResetOptions,
} from "./model/runtime.js";
export {
  parseCact,
  CACT_TAG,
  CACT_HEADER_BYTES,
  CACT_RECORD_BYTES,
  CactDType,
  type CactWeights,
  type CactGeometry,
  type CactTensor,
  type CactTensorRecord,
  type CqMatrix,
  type DenseTensor,
  type RawTensor,
  type CactLayer,
  type CactEngramSite,
  type CactProbeHead,
} from "./model/cact.js";
export {
  NeedleTokenizer,
  parseTokenizerMetadata,
  CHAT_MARKERS,
  PAD_TOKEN_ID,
  EOS_TOKEN_ID,
  BOS_TOKEN_ID,
  UNKNOWN_TOKEN_ID,
  TokenPieceType,
  type TokenizerMetadata,
} from "./model/tokenizer.js";

export {
  Needle,
  createNeedle,
  extract,
  type NeedleOptions,
  type CompleteOptions,
  type RunOptions,
  type ExtractOptions,
  type NeedleResponse,
  type NeedleFunctionCall,
  type CompletionMetrics,
} from "./tools/agent.js";
export {
  defineTool,
  tool,
  normalizeTool,
  normalizeTools,
  serializeTools,
  type NeedleTool,
  type ToolInput,
  type RawNeedleTool,
  type OpenAIFunctionTool,
  type ToolExecutionContext,
  type JsonSchema,
  type JsonValue,
  type JsonPrimitive,
} from "./tools/schema.js";
export {
  JsonSchemaGrammar,
  ToolCallGrammar,
  type GrammarToolCall,
} from "./tools/grammar.js";
export {
  rankToolsBM25,
  retrieveTools,
  type RetrieveToolsOptions,
} from "./tools/retrieval.js";

export {
  loadWeights,
  sha256,
  hasEmbeddedWeights,
  getEmbeddedWeights,
  clearMemoryWeightCache,
  NEEDLE_2_REPOSITORY,
  NEEDLE_2_REVISION,
  NEEDLE_2_URL,
  NEEDLE_2_SHA256,
  NEEDLE_2_BYTES,
  type WeightSource,
  type UrlWeightSource,
  type FileWeightSource,
  type EmbeddedWeightSource,
  type LoadWeightsOptions,
  type WeightProgress,
} from "./weights/source.js";

export type {
  InferenceBackend,
  BackendKind,
  MatrixRowRange,
} from "./backends/backend.js";
