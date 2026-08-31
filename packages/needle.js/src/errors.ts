export type NeedleErrorCode =
  | "INVALID_CACT"
  | "UNSUPPORTED_CACT"
  | "WEIGHTS_NOT_FOUND"
  | "WEIGHTS_INTEGRITY"
  | "WEBGPU_UNAVAILABLE"
  | "BACKEND_UNAVAILABLE"
  | "CONTEXT_OVERFLOW"
  | "INVALID_TOOL_SCHEMA"
  | "GRAMMAR_DEAD_END"
  | "GENERATION_ABORTED";

/** An error with a stable code suitable for application-level handling. */
export class NeedleError extends Error {
  readonly code: NeedleErrorCode;
  override readonly cause?: unknown;

  constructor(code: NeedleErrorCode, message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = "NeedleError";
    this.code = code;
    if (options && "cause" in options) this.cause = options.cause;
  }
}

export function invariant(
  condition: unknown,
  code: NeedleErrorCode,
  message: string,
): asserts condition {
  if (!condition) throw new NeedleError(code, message);
}
