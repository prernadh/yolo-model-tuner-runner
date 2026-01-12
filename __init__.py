"""
YOLOv8 Model Trainer Plugin for FiftyOne

This plugin provides operators to train and apply YOLOv8 object detection
models using FiftyOne. The UI is defined in JavaScript while execution
happens in Python.
"""

import os
import logging
from datetime import datetime
from typing import Optional, List

import torch
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.core.storage as fos

logger = logging.getLogger(__name__)

# Working directories for model training
TRAIN_ROOT = "/tmp/yolo/"
MODEL_ROOT = os.path.join(TRAIN_ROOT, "models")
DATA_ROOT = os.path.join(TRAIN_ROOT, "data")
PROJECT_ROOT = os.path.join(TRAIN_ROOT, "projects")


def _ensure_directories():
    """Create necessary directories if they don't exist."""
    for directory in [MODEL_ROOT, DATA_ROOT, PROJECT_ROOT]:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def _setup_cuda_device(model, target_device_index: int = 0):
    """
    Configure CUDA device for the model.

    Args:
        model: YOLO model instance
        target_device_index: GPU device index to use

    Returns:
        tuple: (configured model, device count)
    """
    cuda_device_count = torch.cuda.device_count()
    logger.info(f"Found {cuda_device_count} CUDA device(s)")

    if cuda_device_count > 0:
        if cuda_device_count > 1 and target_device_index < cuda_device_count:
            device = f"cuda:{target_device_index}"
        else:
            device = "cuda:0"
        model.to(device)
        logger.info(f"Using device: {device}")
    else:
        logger.warning("No CUDA devices found, using CPU")

    return model, cuda_device_count


def _download_model_weights(weights_path: str) -> str:
    """
    Download model weights to local directory.

    Args:
        weights_path: Remote or local path to model weights

    Returns:
        str: Local path to downloaded weights
    """
    _ensure_directories()
    local_weights_path = os.path.join(MODEL_ROOT, os.path.basename(weights_path))
    fos.copy_file(weights_path, local_weights_path)
    logger.info(f"Model weights downloaded to: {local_weights_path}")
    return local_weights_path


def export_yolo_data(
    samples,
    export_dir: str,
    classes: List[str],
    label_field: str = "ground_truth",
    split: Optional[str] = None,
):
    """
    Export FiftyOne samples to YOLOv5 dataset format.

    Args:
        samples: FiftyOne samples or view to export
        export_dir: Directory where dataset will be exported
        classes: List of class names
        label_field: Name of the label field containing detections
        split: Split name ('train', 'val', etc.) or list of splits
    """
    if isinstance(split, list):
        # Recursively export each split
        for split_name in split:
            export_yolo_data(samples, export_dir, classes, label_field, split_name)
        return

    # Determine split view
    if split is None:
        split_view = samples
        split = "val"
    else:
        split_view = samples.match_tags(split)

    # Export to YOLO format
    split_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        classes=classes,
        split=split,
    )
    logger.info(f"Exported {len(split_view)} samples for split '{split}'")


class ModelFineTuner(foo.Operator):
    """Operator to finetune YOLOv8 models on FiftyOne datasets."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="model_fine_tuner",
            label="Finetune YOLOv8 Model",
            description="Finetune a YOLOv8 model on the current dataset",
            unlisted=True,  # Called from JS panel
            allow_immediate_execution=False,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def execute(self, ctx):
        """
        Execute model training.

        Steps:
        1. Download model weights to local directory
        2. Export dataset to YOLO format
        3. Train YOLOv8 model
        4. Save trained weights to specified location

        Returns:
            dict: Training results including weights path and status
        """
        try:
            from ultralytics import YOLO

            # Parse parameters
            det_field = ctx.params["det_field"]
            weights_path = ctx.params["weights_path"]
            export_uri = ctx.params["export_uri"]
            epochs = ctx.params["epochs"]
            target_device_index = ctx.params.get("target_device_index", 0)

            logger.info(f"Starting model training with parameters: det_field={det_field}, "
                       f"epochs={epochs}, device_index={target_device_index}")
            logger.info(f"Weights path: {weights_path}")
            logger.info(f"Export URI: {export_uri}")

            dataset = ctx.dataset
            logger.info(f"Training on dataset: {dataset.name} ({len(dataset)} samples)")

            det_label_field = f"{det_field}.detections.label"
            classes = dataset.distinct(det_label_field)
            logger.info(f"Found {len(classes)} classes: {classes}")
        except Exception as e:
            logger.error(f"Failed to initialize training: {str(e)}", exc_info=True)
            raise

        # Download model weights
        local_weights_path = _download_model_weights(weights_path)

        # Prepare export directory
        dataset_root = os.path.join(DATA_ROOT, dataset.name)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        export_dir = os.path.join(dataset_root, timestamp)
        logger.info(f"Exporting dataset to: {export_dir}")

        # Export dataset to YOLO format
        export_yolo_data(
            ctx.dataset,
            export_dir,
            classes=classes,
            label_field=det_field,
            split=["train", "val"],
        )

        # Verify dataset.yaml exists
        data_yaml = os.path.join(export_dir, "dataset.yaml")
        if not fos.exists(data_yaml):
            raise FileNotFoundError(f"Failed to export dataset to {data_yaml}")

        ctx.set_progress(progress=0.1, label="Dataset exported. Starting training...")
        logger.info("Starting training")

        # Initialize and configure model
        logger.info(f"Loading YOLO model from {local_weights_path}")
        model = YOLO(local_weights_path)
        model, cuda_device_count = _setup_cuda_device(model, target_device_index)

        # Train model
        logger.info(f"Starting training for {epochs} epochs with image size 640")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            name="finetuned",
            project=PROJECT_ROOT,
            exist_ok=True,
        )

        best_weights = os.path.join(results.save_dir, "weights", "best.pt")
        logger.info(f"Training complete. Best weights saved to {best_weights}")

        ctx.set_progress(
            progress=0.9, label="Training complete. Saving final weights..."
        )

        # Save trained weights to specified location
        logger.info(f"Copying weights to final destination: {export_uri}")
        fos.copy_file(best_weights, export_uri)
        logger.info(f"Successfully saved finetuned weights to {export_uri}")

        return {
            "finetuned_weights_path": export_uri,
            "status": "success",
            "cuda_device_count": cuda_device_count,
        }


class GetTagCounts(foo.Operator):
    """Operator to get tag counts for the dataset."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_tag_counts",
            label="Get Tag Counts",
            description="Get count of samples with each tag in the dataset",
            unlisted=True,  # Called from JS panel
            allow_immediate_execution=True,
        )

    def execute(self, ctx):
        """
        Get tag counts for the dataset.

        Returns:
            dict: Tag counts dictionary
        """
        dataset = ctx.dataset
        logger.info(f"Getting tag counts for dataset: {dataset.name}")
        tag_counts = dataset.count_sample_tags()
        logger.info(f"Tag counts: {tag_counts}")

        result = {
            "tag_counts": tag_counts,
        }
        logger.info(f"Returning result: {result}")
        return result


class ApplyRemoteModel(foo.Operator):
    """Operator to apply trained YOLOv8 models to FiftyOne datasets."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="apply_remote_model",
            label="Apply YOLOv8 Model",
            description="Run inference with a YOLOv8 model on the current dataset",
            unlisted=True,  # Called from JS panel
            allow_immediate_execution=False,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def execute(self, ctx):
        """
        Execute model inference.

        Steps:
        1. Download model weights to local directory
        2. Configure CUDA device
        3. Apply model to dataset

        Returns:
            dict: Inference results including status
        """
        from ultralytics import YOLO

        # Parse parameters
        det_field = ctx.params["det_field"]
        weights_path = ctx.params["weights_path"]
        target_device_index = ctx.params.get("target_device_index", 0)

        logger.info(f"Starting model inference with parameters: det_field={det_field}, "
                   f"device_index={target_device_index}")
        logger.info(f"Weights path: {weights_path}")

        dataset = ctx.dataset
        logger.info(f"Applying model to dataset: {dataset.name} ({len(dataset)} samples)")

        # Download model weights
        local_weights_path = _download_model_weights(weights_path)

        # Initialize and configure model
        logger.info(f"Loading YOLO model from {local_weights_path}")
        model = YOLO(local_weights_path)
        model, cuda_device_count = _setup_cuda_device(model, target_device_index)

        # Apply model to dataset
        logger.info(f"Applying model to dataset, predictions will be saved to field '{det_field}'")
        ctx.dataset.apply_model(model, label_field=det_field)
        logger.info(f"Inference complete. Predictions saved to '{det_field}' field")

        return {
            "status": "success",
            "cuda_device_count": cuda_device_count,
        }


def register(plugin):
    """Register plugin operators with FiftyOne."""
    plugin.register(ModelFineTuner)
    plugin.register(GetTagCounts)
    plugin.register(ApplyRemoteModel)
