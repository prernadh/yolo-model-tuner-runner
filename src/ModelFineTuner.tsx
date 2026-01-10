import { Operator, OperatorConfig, types, executeOperator } from "@fiftyone/operators";
import { useRecoilValue } from "recoil";
import * as fos from "@fiftyone/state";

class ModelFineTunerOperator extends Operator {
  get config(): OperatorConfig {
    return new OperatorConfig({
      name: "finetune_yolov8",
      label: "Finetune YOLOv8",
      description: "Finetune a YOLOv8 model on the current view",
      icon: "build_circle",
      unlisted: false,
    });
  }

  useHooks() {
    const dataset = useRecoilValue(fos.dataset);
    return { dataset };
  }

  async resolveInput(ctx: any): Promise<types.Property> {
    const inputs = new types.Object();
    const { dataset } = ctx.hooks;

    // Warning header about training data requirements
    inputs.str("_training_info", {
      label: "⚠️ Training Data Requirements",
      description: "IMPORTANT: Only samples tagged with 'train' and 'val' will be exported for training. Please ensure your samples have these tags before proceeding. You can tag samples using dataset.tag_samples() in Python or through the App UI.",
      view: new types.Warning(),
    });

    // Get detection fields from dataset schema
    // The dataset.sampleFields array doesn't have type info in the format we need
    // So we'll just provide a text input for now (user can see field names in the app)
    inputs.str("det_field", {
      required: true,
      label: "Ground truth field",
      description: "Enter the name of your detection field (e.g., 'ground_truth', 'detections'). This field must contain Detections labels in samples tagged 'train' and 'val'.",
      default: "ground_truth",
    });

    // Section: Model Configuration
    inputs.str("_model_config_header", {
      label: "Model Configuration",
      view: new types.Header({ divider: true }),
    });

    inputs.str("weights_path", {
      default: "gs://voxel51-demo-fiftyone-ai/yolo/yolov8n.pt",
      required: true,
      description: "Path to initial YOLOv8 weights. Supports GCS (gs://), S3 (s3://), or local paths. Common models: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)",
      label: "Initial model weights",
    });

    inputs.str("export_uri", {
      default: "gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.pt",
      required: true,
      description: "Where to save the finetuned model. Supports GCS (gs://), S3 (s3://), or local paths",
      label: "Output model path",
    });

    // Section: Training Configuration
    inputs.str("_training_config_header", {
      label: "Training Configuration",
      view: new types.Header({ divider: true }),
    });

    inputs.int("epochs", {
      default: 1,
      required: true,
      label: "Number of epochs",
      description: "More epochs = better training but takes longer. Start with 10-50 for real training",
      min: 1,
      max: 1000,
    });

    inputs.int("target_device_index", {
      default: 0,
      required: false,
      label: "CUDA device index",
      description: "GPU device to use for training (0, 1, 2, etc.). Defaults to cuda:0",
      min: 0,
    });

    // Section: Advanced Options
    inputs.str("_advanced_header", {
      label: "Advanced Options (Optional)",
      view: new types.Header({ divider: true }),
    });

    inputs.bool("to_coreml", {
      label: "Export to CoreML",
      description: "Also export model in CoreML format for deployment on Apple devices (iOS, macOS)",
      default: false,
    });

    inputs.str("core_ml_export_uri", {
      default: "gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.mlpackage",
      required: false,
      label: "CoreML output path",
      description: "Where to save CoreML model (only used if 'Export to CoreML' is enabled)",
    });

    return new types.Property(
      inputs,
      { view: new types.View({ label: "Finetune YOLOv8 Model" }) }
    );
  }

  async resolvePlacement(): Promise<types.Placement> {
    return new types.Placement({
      place: types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
      view: new types.Button({
        label: "Finetune YOLOv8",
        icon: "build_circle",
        prompt: true,
      }),
    });
  }

  async execute(ctx: any) {
    // Delegate to Python operator
    return executeOperator("model_fine_tuner_2", ctx.params);
  }
}

export { ModelFineTunerOperator };
