import React, { useState, useMemo, useEffect } from "react";
import { useRecoilValue, useSetRecoilState } from "recoil";
import * as fos from "@fiftyone/state";
import { useOperatorExecutor } from "@fiftyone/operators";
import { OperatorExecutionOption } from "@fiftyone/operators/src/state";
import { useTheme } from "@mui/material";
import { OperatorButtonWithWarning } from "./OperatorButtonWithWarning";

interface TagStats {
  train_count: number;
  val_count: number;
  both_count: number;
  total_count: number;
  untagged_count: number;
}

interface DetectionField {
  name: string;
  type: string;
}

export function ModelFineTunerPanel() {
  const theme = useTheme();
  const dataset = useRecoilValue(fos.dataset);
  const totalSampleCount = useRecoilValue(fos.datasetSampleCount) || 0;
  const setView = useSetRecoilState(fos.view);

  // Use operator executor to get tag counts
  const tagCountsExecutor = useOperatorExecutor<{ tag_counts: { [key: string]: number } }>(
    "@prernadh/yolo-model-tuner-runner/get_tag_counts"
  );

  // Extract tag counts from executor result
  const tagCounts = tagCountsExecutor.result?.tag_counts || {};

  // Fetch tag counts when dataset changes and poll every 2 seconds
  useEffect(() => {
    if (!dataset) return;

    // Initial fetch
    tagCountsExecutor.execute({});

    // Poll for updates every 2 seconds
    const interval = setInterval(() => {
      tagCountsExecutor.execute({});
    }, 2000);

    return () => clearInterval(interval);
  }, [dataset?.name]);

  const [activeTab, setActiveTab] = useState<"train" | "apply">("train");
  const [selectedField, setSelectedField] = useState<string>("ground_truth");
  const [weightsPath, setWeightsPath] = useState<string>("gs://voxel51-demo-fiftyone-ai/yolo/yolov8n.pt");
  const [exportUri, setExportUri] = useState<string>("gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.pt");
  const [epochs, setEpochs] = useState<number>(1);
  const [deviceIndex, setDeviceIndex] = useState<number>(0);
  const [isTraining, setIsTraining] = useState<boolean>(false);

  // State for Apply Model tab
  const [applyWeightsPath, setApplyWeightsPath] = useState<string>("gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.pt");
  const [applyField, setApplyField] = useState<string>("predictions");
  const [applyDeviceIndex, setApplyDeviceIndex] = useState<number>(0);
  const [isApplying, setIsApplying] = useState<boolean>(false);

  // State for success notifications
  const [showSuccessOverlay, setShowSuccessOverlay] = useState<boolean>(false);
  const [successMessage, setSuccessMessage] = useState<string>("");
  const [countdown, setCountdown] = useState<number>(5);


  // Calculate tag statistics from recoil state
  const tagStats = useMemo(() => {
    const trainCount = tagCounts["train"] || 0;
    const valCount = tagCounts["val"] || 0;
    const totalCount = totalSampleCount || 0;

    // Calculate overlap (samples with both tags)
    // If train + val > total, then there must be overlap
    const bothCount = Math.max(0, trainCount + valCount - totalCount);

    // Calculate exclusive counts (samples with only one tag)
    const trainOnlyCount = trainCount - bothCount;
    const valOnlyCount = valCount - bothCount;

    // Untagged = samples that don't have either 'train' or 'val' tags
    const taggedCount = trainOnlyCount + valOnlyCount + bothCount;
    const untaggedCount = Math.max(0, totalCount - taggedCount);

    return {
      train_count: trainOnlyCount,
      val_count: valOnlyCount,
      both_count: bothCount,
      total_count: totalCount,
      untagged_count: untaggedCount,
    };
  }, [tagCounts, totalSampleCount]);

  const styles = useMemo(() => createStyles(theme), [theme]);

  const trainRatio = tagStats.total_count > 0 ? (tagStats.train_count / tagStats.total_count) * 100 : 0;
  const valRatio = tagStats.total_count > 0 ? (tagStats.val_count / tagStats.total_count) * 100 : 0;
  const bothRatio = tagStats.total_count > 0 ? (tagStats.both_count / tagStats.total_count) * 100 : 0;
  const untaggedRatio = tagStats.total_count > 0 ? (tagStats.untagged_count / tagStats.total_count) * 100 : 0;

  const isReadyToTrain = (tagStats.train_count > 0 || tagStats.both_count > 0) && (tagStats.val_count > 0 || tagStats.both_count > 0);

  const handleFilterByTag = (tag: string | null) => {
    if (tag === null) {
      // Clear all filters - show all samples
      setView([]);
    } else if (tag === "untagged") {
      // Show samples without train or val tags
      setView([
        {
          _cls: "fiftyone.core.stages.MatchTags",
          kwargs: [
            ["tags", ["train", "val"]],
            ["bool", false]
          ]
        }
      ]);
    } else if (tag === "both") {
      // Show samples with both train AND val tags
      setView([
        {
          _cls: "fiftyone.core.stages.MatchTags",
          kwargs: [["tags", ["train"]]]
        },
        {
          _cls: "fiftyone.core.stages.MatchTags",
          kwargs: [["tags", ["val"]]]
        }
      ]);
    } else {
      // Show samples with specific tag
      setView([
        {
          _cls: "fiftyone.core.stages.MatchTags",
          kwargs: [["tags", [tag]]]
        }
      ]);
    }
  };

  // Handler when user selects execution option (immediate or delegated)
  const handleTrainStart = (option: OperatorExecutionOption) => {
    setIsTraining(true);
    console.log(`Training started with ${option.isDelegated ? 'delegated' : 'immediate'} execution`);
  };

  // Handler for successful training execution
  const handleTrainSuccess = (response: any) => {
    console.log("Training completed successfully", response);
    setIsTraining(false);

    if (response?.delegated) {
      // Delegated execution - operation queued
      const operatorId = response?.result?.id?.$oid;
      console.log("Training delegated with operation ID:", operatorId);
      setSuccessMessage("Training successfully scheduled!");
      setShowSuccessOverlay(true);
      setCountdown(5);

      // Start countdown
      let count = 5;
      const countdownInterval = setInterval(() => {
        count--;
        setCountdown(count);
        if (count === 0) {
          clearInterval(countdownInterval);
          setShowSuccessOverlay(false);
        }
      }, 1000);
    } else {
      // Immediate execution - operation completed
      console.log("Training completed immediately");
    }
  };

  // Handler for training errors
  const handleTrainError = (error: any) => {
    console.error("Training failed:", error);
    setIsTraining(false);
  };

  // Handler when user selects execution option for apply model
  const handleApplyStart = (option: OperatorExecutionOption) => {
    setIsApplying(true);
    console.log(`Model application started with ${option.isDelegated ? 'delegated' : 'immediate'} execution`);
  };

  // Handler for successful model application
  const handleApplySuccess = (response: any) => {
    console.log("Model application completed successfully", response);
    setIsApplying(false);

    if (response?.delegated) {
      // Delegated execution - operation queued
      const operatorId = response?.result?.id?.$oid;
      console.log("Model application delegated with operation ID:", operatorId);
      setSuccessMessage("Model application successfully scheduled!");
      setShowSuccessOverlay(true);
      setCountdown(5);

      // Start countdown
      let count = 5;
      const countdownInterval = setInterval(() => {
        count--;
        setCountdown(count);
        if (count === 0) {
          clearInterval(countdownInterval);
          setShowSuccessOverlay(false);
        }
      }, 1000);
    } else {
      // Immediate execution - operation completed
      console.log("Model application completed immediately");
    }
  };

  // Handler for model application errors
  const handleApplyError = (error: any) => {
    console.error("Model application failed:", error);
    setIsApplying(false);
  };

  return (
    <div style={styles.container}>
      {/* Success Overlay */}
      {showSuccessOverlay && (
        <div style={styles.successOverlay}>
          <div style={styles.successOverlayContent}>
            <div style={styles.successIcon}>✓</div>
            <h3 style={styles.successTitle}>{successMessage}</h3>
            <p style={styles.successSubtitle}>Check the Runs tab for progress</p>
            <p style={styles.countdownText}>
              Taking you back in {countdown}...
            </p>
          </div>
        </div>
      )}

      <div style={styles.header}>
        <h2 style={styles.title}>YOLOv8 Model Trainer</h2>
        <p style={styles.subtitle}>
          Train and apply YOLOv8 models on your dataset
        </p>
      </div>

      {/* Tab Navigation */}
      <div style={styles.tabContainer}>
        <div
          style={{
            ...styles.tab,
            ...(activeTab === "train" ? styles.activeTab : {}),
          }}
          onClick={() => setActiveTab("train")}
        >
          Train Model
        </div>
        <div
          style={{
            ...styles.tab,
            ...(activeTab === "apply" ? styles.activeTab : {}),
          }}
          onClick={() => setActiveTab("apply")}
        >
          Apply Model
        </div>
      </div>

      {/* Train Tab Content */}
      {activeTab === "train" && (
        <>
          {/* Tag Distribution Section */}
      <div style={styles.section}>
        <h3 style={styles.sectionTitle}>Training Data Distribution</h3>

        <div style={styles.statsCard}>
          <div style={styles.statsHeader}>
            <span style={styles.statsTitle}>
              {isReadyToTrain ? "✅ Ready to Train" : "⚠️ Action Required"}
            </span>
          </div>

          {/* Vertical Bar Chart Histogram */}
          <div style={styles.histogramContainer}>
            <div style={styles.histogramChart}>
              {/* Train Bar */}
              <div style={styles.barWrapper} onClick={() => handleFilterByTag("train")}>
                <div style={styles.barContainer}>
                  <div
                    style={{
                      ...styles.bar,
                      height: tagStats.total_count > 0 ? `${(tagStats.train_count / tagStats.total_count) * 100}%` : '0%',
                      backgroundColor: "#4caf50",
                      cursor: "pointer",
                      boxShadow: "0 0 20px rgba(76, 175, 80, 0.6), 0 0 40px rgba(76, 175, 80, 0.3)",
                    }}
                    title={`Click to filter: ${tagStats.train_count} samples`}
                  />
                </div>
                <div style={{...styles.barLabel, color: "#4caf50"}}>train</div>
                <div style={{...styles.barCount, color: "#4caf50"}}>
                  {tagStats.train_count}
                </div>
              </div>

              {/* Val Bar */}
              <div style={styles.barWrapper} onClick={() => handleFilterByTag("val")}>
                <div style={styles.barContainer}>
                  <div
                    style={{
                      ...styles.bar,
                      height: tagStats.total_count > 0 ? `${(tagStats.val_count / tagStats.total_count) * 100}%` : '0%',
                      backgroundColor: "#2196f3",
                      cursor: "pointer",
                      boxShadow: "0 0 20px rgba(33, 150, 243, 0.6), 0 0 40px rgba(33, 150, 243, 0.3)",
                    }}
                    title={`Click to filter: ${tagStats.val_count} samples`}
                  />
                </div>
                <div style={{...styles.barLabel, color: "#2196f3"}}>val</div>
                <div style={{...styles.barCount, color: "#2196f3"}}>
                  {tagStats.val_count}
                </div>
              </div>

              {/* Both Tags Bar */}
              <div style={styles.barWrapper} onClick={() => handleFilterByTag("both")}>
                <div style={styles.barContainer}>
                  <div
                    style={{
                      ...styles.bar,
                      height: tagStats.total_count > 0 ? `${(tagStats.both_count / tagStats.total_count) * 100}%` : '0%',
                      backgroundColor: "#f44336",
                      cursor: "pointer",
                      boxShadow: "0 0 20px rgba(244, 67, 54, 0.6), 0 0 40px rgba(244, 67, 54, 0.3)",
                    }}
                    title={`Click to filter: ${tagStats.both_count} samples`}
                  />
                </div>
                <div style={{...styles.barLabel, color: "#f44336"}}>both</div>
                <div style={{...styles.barCount, color: "#f44336"}}>
                  {tagStats.both_count}
                </div>
              </div>

              {/* Untagged Bar */}
              <div style={styles.barWrapper} onClick={() => handleFilterByTag("untagged")}>
                <div style={styles.barContainer}>
                  <div
                    style={{
                      ...styles.bar,
                      height: tagStats.total_count > 0 ? `${(tagStats.untagged_count / tagStats.total_count) * 100}%` : '0%',
                      backgroundColor: "#9e9e9e",
                      cursor: "pointer",
                      boxShadow: "0 0 20px rgba(158, 158, 158, 0.5), 0 0 40px rgba(158, 158, 158, 0.2)",
                    }}
                    title={`Click to filter: ${tagStats.untagged_count} samples`}
                  />
                </div>
                <div style={{...styles.barLabel, color: "#9e9e9e"}}>untagged</div>
                <div style={{...styles.barCount, color: "#9e9e9e"}}>
                  {tagStats.untagged_count}
                </div>
              </div>
            </div>

            {/* Total count summary */}
            <div
              style={{...styles.totalSummary, cursor: "pointer"}}
              onClick={() => handleFilterByTag(null)}
              title="Click to show all samples"
            >
              <span style={styles.totalLabel}>Total Samples:</span>
              <span style={styles.totalValue}>{tagStats.total_count}</span>
            </div>
          </div>

          {!isReadyToTrain && (
            <div style={styles.warning}>
              <strong>Action Required:</strong> This panel uses samples with the tag{" "}
              <span style={styles.tagBadge}>train</span> for training the model and samples with the tag{" "}
              <span style={styles.tagBadge}>val</span> as the validation set. Please use the tagging workflow in the app to add{" "}
              <span style={styles.tagBadge}>train</span> and <span style={styles.tagBadge}>val</span> tags.
            </div>
          )}
        </div>
      </div>

      {/* Configuration Section */}
      <div style={styles.section}>
        <h3 style={styles.sectionTitle}>Configuration</h3>

        <div style={styles.formGroup}>
          <label style={styles.label}>Ground Truth Field</label>
          <input
            type="text"
            value={selectedField}
            onChange={(e) => setSelectedField(e.target.value)}
            style={styles.input}
            placeholder="ground_truth"
          />
          <div style={styles.hint}>Detection field containing ground truth labels</div>
        </div>

        <div style={styles.formGroup}>
          <label style={styles.label}>Initial Model Weights</label>
          <input
            type="text"
            value={weightsPath}
            onChange={(e) => setWeightsPath(e.target.value)}
            style={styles.input}
            placeholder="gs://bucket/yolov8n.pt"
          />
          <div style={styles.hint}>GCS (gs://), S3 (s3://), or local path to YOLOv8 weights</div>
        </div>

        <div style={styles.formGroup}>
          <label style={styles.label}>Output Model Path</label>
          <input
            type="text"
            value={exportUri}
            onChange={(e) => setExportUri(e.target.value)}
            style={styles.input}
            placeholder="gs://bucket/yolov8n_finetuned.pt"
          />
          <div style={styles.hint}>Where to save finetuned weights</div>
        </div>

        <div style={styles.formRow}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Epochs</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              style={styles.input}
              min={1}
              max={1000}
            />
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>CUDA Device</label>
            <input
              type="number"
              value={deviceIndex}
              onChange={(e) => setDeviceIndex(parseInt(e.target.value))}
              style={styles.input}
              min={0}
            />
          </div>
        </div>

      </div>

      {/* Action Section */}
      <div style={styles.footer}>
        <OperatorButtonWithWarning
          operatorUri="@prernadh/yolo-model-tuner-runner/model_fine_tuner"
          executionParams={{
            det_field: selectedField,
            weights_path: weightsPath,
            export_uri: exportUri,
            epochs: epochs,
            target_device_index: deviceIndex,
          }}
          onOptionSelected={handleTrainStart}
          onSuccess={handleTrainSuccess}
          onError={handleTrainError}
          disabled={isTraining || !isReadyToTrain}
          variant="contained"
          color="primary"
          style={{ width: "100%" }}
        >
          {isTraining ? "Training..." : "Start Training"}
        </OperatorButtonWithWarning>
      </div>
        </>
      )}

      {/* Apply Model Tab Content */}
      {activeTab === "apply" && (
        <>
          {/* Configuration Section */}
          <div style={styles.section}>
            <h3 style={styles.sectionTitle}>Model Configuration</h3>

            <div style={styles.formGroup}>
              <label style={styles.label}>Model Weights Path</label>
              <input
                type="text"
                value={applyWeightsPath}
                onChange={(e) => setApplyWeightsPath(e.target.value)}
                style={styles.input}
                placeholder="gs://bucket/yolov8n_finetuned.pt"
              />
              <div style={styles.hint}>GCS (gs://), S3 (s3://), or local path to trained YOLOv8 weights</div>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Prediction Field</label>
              <input
                type="text"
                value={applyField}
                onChange={(e) => setApplyField(e.target.value)}
                style={styles.input}
                placeholder="predictions"
              />
              <div style={styles.hint}>Field name where predictions will be stored</div>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>CUDA Device</label>
              <input
                type="number"
                value={applyDeviceIndex}
                onChange={(e) => setApplyDeviceIndex(parseInt(e.target.value))}
                style={styles.input}
                min={0}
              />
              <div style={styles.hint}>GPU device index (0 for first GPU)</div>
            </div>
          </div>

          {/* Action Section */}
          <div style={styles.footer}>
            <OperatorButtonWithWarning
              operatorUri="@prernadh/yolo-model-tuner-runner/apply_remote_model"
              executionParams={{
                det_field: applyField,
                weights_path: applyWeightsPath,
                target_device_index: applyDeviceIndex,
              }}
              onOptionSelected={handleApplyStart}
              onSuccess={handleApplySuccess}
              onError={handleApplyError}
              disabled={isApplying}
              variant="contained"
              color="primary"
              style={{ width: "100%" }}
            >
              {isApplying ? "Applying Model..." : "Apply Model"}
            </OperatorButtonWithWarning>
          </div>
        </>
      )}
    </div>
  );
}

function createStyles(theme: any) {
  const isDark = theme.palette.mode === "dark";

  return {
    container: {
      display: "flex",
      flexDirection: "column" as const,
      height: "100%",
      padding: "16px",
      backgroundColor: theme.palette.background.default,
      color: theme.palette.text.primary,
      fontFamily: "system-ui, -apple-system, sans-serif",
      overflowY: "auto" as const,
    },
    header: {
      marginBottom: "24px",
    },
    title: {
      fontSize: "20px",
      fontWeight: 600,
      margin: "0 0 8px 0",
      color: theme.palette.text.primary,
    },
    subtitle: {
      fontSize: "14px",
      color: theme.palette.text.secondary,
      margin: 0,
    },
    tabContainer: {
      display: "flex",
      borderBottom: `2px solid ${theme.palette.divider}`,
      marginBottom: "24px",
      gap: "8px",
    },
    tab: {
      padding: "12px 24px",
      cursor: "pointer",
      fontSize: "14px",
      fontWeight: 500,
      color: theme.palette.text.secondary,
      borderBottom: "2px solid transparent",
      marginBottom: "-2px",
      transition: "all 0.2s ease",
      userSelect: "none" as const,
    },
    activeTab: {
      color: theme.palette.primary.main,
      borderBottom: `2px solid ${theme.palette.primary.main}`,
      fontWeight: 600,
    },
    section: {
      marginBottom: "24px",
    },
    sectionTitle: {
      fontSize: "16px",
      fontWeight: 600,
      marginBottom: "12px",
      color: theme.palette.text.primary,
    },
    statsCard: {
      padding: "16px",
      borderRadius: "8px",
      marginBottom: "16px",
      backgroundColor: "transparent",
    },
    statsHeader: {
      marginBottom: "24px",
    },
    statsTitle: {
      fontSize: "16px",
      fontWeight: 600,
      color: "#333",
    },
    histogramContainer: {
      marginBottom: "16px",
    },
    histogramChart: {
      display: "flex",
      alignItems: "flex-end",
      justifyContent: "space-around",
      height: "200px",
      padding: "16px",
      backgroundColor: "transparent",
      borderRadius: "12px",
      gap: "24px",
    },
    barWrapper: {
      display: "flex",
      flexDirection: "column" as const,
      alignItems: "center",
      flex: 1,
      gap: "12px",
      cursor: "pointer",
      transition: "transform 0.2s ease, opacity 0.2s ease",
    },
    barContainer: {
      width: "100%",
      height: "120px",
      display: "flex",
      flexDirection: "column" as const,
      justifyContent: "flex-end",
      alignItems: "center",
    },
    bar: {
      width: "60%",
      minHeight: "4px",
      borderRadius: "6px 6px 0 0",
      transition: "height 0.3s ease, box-shadow 0.3s ease",
      position: "relative" as const,
    },
    barLabel: {
      fontSize: "13px",
      fontWeight: 600,
      marginTop: "4px",
    },
    barCount: {
      fontSize: "24px",
      fontWeight: "bold",
    },
    totalSummary: {
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      gap: "8px",
      padding: "12px",
      backgroundColor: isDark ? theme.palette.background.paper : "#f5f5f5",
      borderRadius: "6px",
      marginTop: "12px",
      border: `1px solid ${theme.palette.divider}`,
      transition: "opacity 0.2s ease",
    },
    totalLabel: {
      fontSize: "14px",
      color: theme.palette.text.secondary,
      fontWeight: 500,
    },
    totalValue: {
      fontSize: "18px",
      fontWeight: "bold",
      color: theme.palette.text.primary,
    },
    warning: {
      padding: "12px",
      backgroundColor: "#fff",
      borderRadius: "4px",
      fontSize: "13px",
      color: "#d84315",
      border: "1px solid #ff9800",
    },
    tagBadge: {
      display: "inline-block",
      padding: "2px 6px",
      backgroundColor: "#f0f0f0",
      border: "1px solid #ccc",
      borderRadius: "3px",
      fontFamily: "monospace",
      fontSize: "12px",
      fontWeight: 600,
      color: "#333",
      margin: "0 2px",
    },
    formGroup: {
      marginBottom: "16px",
    },
    formRow: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: "16px",
    },
    label: {
      display: "block",
      fontSize: "14px",
      fontWeight: 500,
      marginBottom: "6px",
      color: theme.palette.text.primary,
    },
    input: {
      width: "100%",
      padding: "8px 12px",
      fontSize: "14px",
      borderRadius: "4px",
      border: `1px solid ${theme.palette.divider}`,
      backgroundColor: theme.palette.background.paper,
      color: theme.palette.text.primary,
      boxSizing: "border-box" as const,
    },
    hint: {
      fontSize: "12px",
      color: theme.palette.text.secondary,
      marginTop: "4px",
    },
    checkboxLabel: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      cursor: "pointer",
      fontSize: "14px",
      color: theme.palette.text.primary,
    },
    checkbox: {
      cursor: "pointer",
    },
    footer: {
      marginTop: "auto",
      paddingTop: "16px",
      borderTop: `1px solid ${theme.palette.divider}`,
    },
    successOverlay: {
      position: "absolute" as const,
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: "rgba(0, 0, 0, 0.8)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 9999,
    },
    successOverlayContent: {
      textAlign: "center" as const,
      color: "#fff",
      padding: "40px",
    },
    successIcon: {
      fontSize: "72px",
      color: "#4caf50",
      marginBottom: "20px",
      fontWeight: "bold" as const,
    },
    successTitle: {
      fontSize: "24px",
      fontWeight: 600,
      marginBottom: "12px",
      color: "#fff",
    },
    successSubtitle: {
      fontSize: "16px",
      marginBottom: "24px",
      color: "rgba(255, 255, 255, 0.8)",
    },
    countdownText: {
      fontSize: "18px",
      fontWeight: 500,
      color: "#4caf50",
    },
  };
}
