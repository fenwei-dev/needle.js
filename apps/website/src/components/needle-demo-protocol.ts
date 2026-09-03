export type DemoBackend = "resident" | "cpu";
export type DemoMode = "raw" | "ai-sdk" | "pi-agent";

export interface RoomState {
  readonly lightOn: boolean;
  readonly curtainOpen: boolean;
  readonly fanOn: boolean;
  readonly fanSpeed: "low" | "medium" | "high";
}

export interface WorkerRunRequest {
  readonly type: "run";
  readonly id: number;
  readonly mode: DemoMode;
  readonly backend: DemoBackend;
  readonly prompt: string;
  readonly room: RoomState;
}

export interface WorkerDisposeRequest {
  readonly type: "dispose";
}

export type WorkerRequest = WorkerRunRequest | WorkerDisposeRequest;

export interface WorkerStatusEvent {
  readonly type: "status";
  readonly id: number;
  readonly status: string;
  readonly backend?: string;
}

export interface WorkerResultEvent {
  readonly type: "result";
  readonly id: number;
  readonly reply: string;
  readonly raw: unknown;
  readonly room: RoomState;
  readonly actions: readonly string[];
  readonly backend: string;
  readonly elapsedMs: number;
}

export interface WorkerErrorEvent {
  readonly type: "error";
  readonly id: number;
  readonly message: string;
}

export type WorkerEvent = WorkerStatusEvent | WorkerResultEvent | WorkerErrorEvent;
