import { Operator, OperatorConfig, types, executeOperator } from "@fiftyone/operators";

class ApplyRemoteModelOperator extends Operator {
  get config(): OperatorConfig {
    return new OperatorConfig({
      name: "apply_yolo_model",
      label: "Apply YOLO Model",
      description: "Run inference with a YOLOv8 model using remotely stored weights",
      icon: "input",
      unlisted: false,
    });
  }

  async resolveInput(): Promise<types.Property> {
    const inputs = new types.Object();

    // Header: Model Configuration
    inputs.str("_model_info", {
      label: "🤖 Model Configuration",
      description: "Apply a YOLOv8 model to your current view. Predictions will be stored in the specified field.",
      view: new types.Header({ divider: true }),
    });

    inputs.str("weights_path", {
      default: "gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.pt",
      required: true,
      description: "GCS (gs://), S3 (s3://), or local path to YOLOv8 model weights",
      label: "Model weights path",
    });

    inputs.str("det_field", {
      required: true,
      label: "Prediction field",
      description: "Field name where predictions will be stored",
      default: "predictions",
    });

    // Header: Inference Parameters
    inputs.str("_inference_params", {
      label: "⚙️ Inference Parameters",
      view: new types.Header({ divider: true }),
    });

    inputs.int("target_device_index", {
      default: 0,
      required: false,
      label: "CUDA device",
      description: "GPU device index (defaults to cuda:0)",
    });

    return new types.Property(
      inputs,
      { view: new types.View({ label: "Apply YOLOv8 Model" }) }
    );
  }

  async resolvePlacement(): Promise<types.Placement> {
    return new types.Placement({
      place: types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
      view: new types.Button({
        label: "Apply YOLO Model",
        icon: "input",
        prompt: true,
      }),
    });
  }

  async execute(ctx: any) {
    // Delegate to Python operator
    return executeOperator("apply_remote_model_2", ctx.params);
  }
}

export { ApplyRemoteModelOperator };
