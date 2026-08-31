import { type LoadModelOptions, NeedleModel } from "needle.js";

export interface NeedleModelPoolOptions extends LoadModelOptions {
  readonly model?: NeedleModel | PromiseLike<NeedleModel>;
  readonly disposeModel?: boolean;
}

export class NeedleModelPool {
  #modelPromise: Promise<NeedleModel> | undefined;
  #resolvedModel: NeedleModel | undefined;
  readonly #ownsModel: boolean;

  constructor(private readonly options: NeedleModelPoolOptions) {
    this.#ownsModel = options.disposeModel ?? options.model === undefined;
  }

  get(): Promise<NeedleModel> {
    if (!this.#modelPromise) {
      this.#modelPromise = this.options.model
        ? Promise.resolve(this.options.model)
        : NeedleModel.load(this.options);
      void this.#modelPromise.then(
        (model) => {
          this.#resolvedModel = model;
        },
        () => {
          // The original promise remains the single error reported to callers.
        },
      );
    }
    return this.#modelPromise;
  }

  async dispose(): Promise<void> {
    if (!this.#ownsModel) return;
    const model =
      this.#resolvedModel ?? (this.#modelPromise ? await this.#modelPromise : undefined);
    await model?.dispose();
    this.#resolvedModel = undefined;
    this.#modelPromise = undefined;
  }
}
