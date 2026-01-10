# YOLOv8 Trainer Panel

A comprehensive FiftyOne plugin that provides an interactive panel for training and applying YOLOv8 models on your datasets with visual tag distribution tracking and inference capabilities.

## Features

### 📊 Train Model Tab
![](https://github.com/prernadh/yolo-model-tuner-runner/blob/main/model_trainer_gif.gif)

- 🎯 **Visual Tag Distribution**: Interactive histogram showing train/val/both/untagged sample counts
- 📈 **Clickable Bars**: Filter dataset by clicking on histogram bars (train, val, both, or untagged)
- ✅ **Training Readiness**: Real-time validation showing when your dataset is ready to train
- 🎨 **Glowing Visualization**: Clean, modern UI with color-coded bars
- ⚙️ **Training Configuration**: Set epochs, CUDA device, model weights, and export paths
- 🔄 **Delegated Execution**: Long-running training jobs run in the background

### 🚀 Apply Model Tab

- 📥 **Model Inference**: Apply trained YOLO models to your dataset
- 🎯 **Custom Fields**: Specify where predictions should be stored
- 🔧 **GPU Selection**: Choose which CUDA device to use
- 📦 **Cloud Storage**: Load models from GCS, S3, or local paths

### 📋 Tag Distribution Dashboard

The panel displays an interactive histogram with four bars:
- **🟢 Green (Train)**: Samples with only the `train` tag
- **🔵 Blue (Val)**: Samples with only the `val` tag
- **🔴 Red (Both)**: Samples with both `train` and `val` tags
- **⚪ Gray (Untagged)**: Samples without either tag

**Clickable Filtering**: Click any bar to filter the dataset view:
- Click train → show only train samples
- Click val → show only val samples
- Click both → show samples with both tags
- Click untagged → show samples without train or val tags
- Click total → show all samples

## Installation

### Requirements

- FiftyOne >= 1.2.0
- Python: `ultralytics` (see [requirements.txt](requirements.txt))
- Node.js and Yarn (for building the JS bundle)
- CUDA-enabled GPU (recommended for training and inference)

### Install Plugin

```bash
fiftyone plugins download https://github.com/prernadh/yolo-model-tuner-runner
```

### Manual Installation

1. Clone or download this plugin directory

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build the JavaScript bundle:
```bash
cd /path/to/fiftyone-plugins
FIFTYONE_DIR=/path/to/fiftyone yarn workspace @prernadh/yolo-model-tuner-runner build
```

## Usage

### Opening the Panel

1. **Load a dataset** in FiftyOne App
2. **Open the Panels menu** (Click the `+` icon next to `Samples`)
3. **Select "YOLOv8 Trainer"** from the available panels
4. The panel will appear in the sidebar

### Train Model Tab

#### Step 1: Tag Your Data

Tag samples with `train` and `val` tags using Python:

```python
import fiftyone as fo

dataset = fo.load_dataset("my_dataset")

# Tag 80% for training
train_view = dataset.limit(int(len(dataset) * 0.8))
train_view.tag_samples("train")

# Tag 20% for validation
val_view = dataset.skip(int(len(dataset) * 0.8))
val_view.tag_samples("val")

dataset.save()
```

Or tag samples in the App using the tagging workflow.

#### Step 2: Review Tag Distribution

The histogram shows:
- How many samples are tagged for training
- How many samples are tagged for validation
- How many have both tags (overlap)
- How many are untagged

Click any bar to filter and review samples in that category.

#### Step 3: Configure Training

1. **Ground Truth Field**: The field containing your labeled detections (default: `ground_truth`)
2. **Initial Model Weights**: Path to starting weights (GCS/S3/local)
   - Example: `gs://voxel51-demo-fiftyone-ai/yolo/yolov8n.pt`
   - Or use Ultralytics model names: `yolov8n.pt`, `yolov8s.pt`, etc.
3. **Output Model Path**: Where to save trained weights
   - Example: `gs://your-bucket/models/yolov8n_finetuned.pt`
4. **Epochs**: Number of training epochs (start with 1-10 for testing)
5. **CUDA Device**: GPU index to use (0 for first GPU)

#### Step 4: Start Training

1. Click **"Start Training"** button
2. Training runs in the background (if delegated execution is enabled)
3. Monitor progress in the FiftyOne execution panel or logs

### Apply Model Tab

#### Apply a Trained Model

1. Switch to the **Apply Model** tab
2. Configure:
   - **Model Weights Path**: Path to your trained model
   - **Prediction Field**: Where to store predictions (e.g., `predictions`)
   - **CUDA Device**: GPU index to use
3. Click **"Apply Model"**
4. Predictions will be added to your dataset in the specified field

## Architecture

### Data Flow

```
Panel Opens → React: Load tag distribution from dataset
                                ↓
User clicks bar → React: Filter view to show selected samples
                                ↓
User clicks "Start Training" → Python: model_fine_tuner_2
                                ↓
                        Export dataset to YOLO format
                                ↓
                        Train YOLOv8 model with ultralytics
                                ↓
                        Save weights to cloud storage
```

### Hybrid Design

- **React Panel**: Rich interactive UI with tabs, histograms, and forms
- **Python Operators**: Handle training and inference execution
- **Recoil State**: Dataset-level tag counts that persist across view changes

## Python Operators

### Internal Operators (unlisted)

1. `model_fine_tuner_2` - Train YOLOv8 models
   - Downloads weights from cloud storage
   - Exports dataset to YOLO format (using `train` and `val` tags)
   - Trains model with ultralytics
   - Saves finetuned weights back to cloud

2. `apply_remote_model_2` - Apply trained models
   - Downloads model weights
   - Runs inference on dataset
   - Stores predictions in specified field

## File Structure

```
yolo-model-tuner-runner/
├── __init__.py                 # Python operators (~280 lines, refactored)
├── fiftyone.yml                # Plugin metadata
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── package.json                # React dependencies
├── vite.config.ts              # Build configuration
├── tsconfig.json               # TypeScript config
├── src/
│   ├── index.ts                # Panel registration
│   └── ModelFineTunerPanel.tsx # Main panel component (~680 lines)
└── dist/
    └── index.umd.js            # Compiled bundle (~24.8 kB)
```

## Technical Details

### JavaScript Dependencies

- React 18.2.0
- Recoil (for FiftyOne state management)
- @fiftyone/operators, @fiftyone/state, @fiftyone/components
- @mui/material (for theming)

### Python Dependencies

- FiftyOne SDK
- ultralytics (for YOLOv8)
- torch (for GPU support)

### Key Technologies

- **TypeScript** for type safety
- **React Hooks** (useState, useMemo, useEffect, useRecoilValue)
- **FiftyOne Operator System** for Python ↔ JavaScript communication
- **Custom Recoil Selectors** for dataset-level tag counts with `root: true`
- **View Stages** for filtering samples by tags

## Development

### Build the Plugin

```bash
cd /path/to/fiftyone-plugins
FIFTYONE_DIR=/path/to/fiftyone yarn workspace @prernadh/yolo-model-tuner-runner build
```

### Development Mode

```bash
yarn workspace @prernadh/yolo-model-tuner-runner dev
```

**Build Stats:**
- Build time: ~240ms
- Bundle size: ~24.8 kB (gzip: ~8.4 kB)

## Advanced Features

### Dataset-Level Tag Counts

The panel uses a custom Recoil selector with `root: true` to ensure tag counts always reflect the full dataset, not the current filtered view. This means:

- Tag distribution remains constant when filtering
- "Ready to Train" status doesn't change when viewing subsets
- Histogram always shows the full picture

### Smart Tag Overlap Detection

The panel detects samples with both `train` and `val` tags:
- Calculates overlap: `both_count = train_count + val_count - total_count`
- Shows exclusive counts: train-only, val-only, and both
- Helps identify data labeling issues

### Multi-GPU Support

- Automatically detects available CUDA devices
- Allows selection of specific GPU via device index
- Falls back to CPU if no GPU is available (with warning logs)

## Use Cases

1. **Quick Prototyping**: Train YOLOv8 models directly from the FiftyOne App
2. **Iterative Development**: Tag, train, evaluate, repeat without leaving the UI
3. **Model Deployment**: Apply trained models to new data for inference
4. **Data Quality**: Use histogram to verify train/val split before training
5. **GPU Management**: Select specific GPUs for training on multi-GPU systems

## Troubleshooting

### Panel doesn't appear

- Verify FiftyOne plugins are enabled in your settings
- Run `fiftyone plugins list` to check installation
- Rebuild the plugin:
```bash
cd /path/to/fiftyone-plugins
FIFTYONE_DIR=/path/to/fiftyone yarn workspace @prernadh/yolo-model-tuner-runner build
```
- Restart the FiftyOne App

### Tag counts show zero

- Ensure samples are tagged with `train` and `val` tags
- Check: `dataset.count_sample_tags()` in Python
- Tags are case-sensitive: use lowercase `train` and `val`

### Training fails

- Check logs for error messages
- Verify weights path is accessible (GCS/S3 credentials configured)
- Ensure dataset has labels in the specified detection field
- Check GPU memory availability
- Verify ultralytics is installed: `pip install ultralytics`

### "Action Required" warning persists

- Ensure you have at least 1 sample tagged `train`
- Ensure you have at least 1 sample tagged `val`
- Refresh the panel by switching tabs
- Check if tags were saved: `dataset.count_sample_tags()`

### Apply Model fails

- Verify model weights path exists and is accessible
- Check that the model is compatible (YOLOv8 format)
- Ensure field name doesn't conflict with existing fields
- Check GPU availability and CUDA setup

## Example Workflows

### Workflow 1: Train a Custom Detector

```python
import fiftyone as fo

# Load dataset
dataset = fo.load_dataset("my_detections")

# Split into train/val
train_samples = dataset.take(800)
train_samples.tag_samples("train")

val_samples = dataset.skip(800)
val_samples.tag_samples("val")

dataset.save()
```

1. Open YOLOv8 Trainer panel
2. Verify histogram shows correct distribution
3. Configure training (set epochs=50 for real training)
4. Click "Start Training"
5. Monitor progress in execution panel

### Workflow 2: Apply a Trained Model

1. Switch to "Apply Model" tab
2. Enter path to your trained weights
3. Set prediction field name (e.g., `my_model_v1`)
4. Click "Apply Model"
5. Review predictions in the App

### Workflow 3: Data Quality Check

1. Open the panel and review tag distribution
2. Click "Both" bar to see samples with overlapping tags
3. Remove duplicate tags as needed
4. Click "Untagged" to see unlabeled samples
5. Tag remaining samples appropriately

## Version

**1.0.0** - Full-featured release with dual-tab interface

## License

Apache 2.0

## Author

@prernadh

## Changelog

### 1.0.0 (Current)

- ✨ Dual-tab interface (Train Model / Apply Model)
- 📊 Interactive histogram with 4 bars (train/val/both/untagged)
- 🖱️ Clickable bars for dataset filtering
- 🎨 Modern UI with glowing bars and clean styling
- 📝 Comprehensive logging throughout Python operators
- 🔧 Refactored Python code with helper functions
- 🎯 Dataset-level tag counts using custom Recoil selector
- ⚡ Type hints and proper error handling
- 🚀 Apply Model functionality for inference
- 📦 Support for GCS, S3, and local model weights
