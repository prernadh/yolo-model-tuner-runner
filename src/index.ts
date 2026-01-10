import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import { ModelFineTunerPanel } from "./ModelFineTunerPanel";

registerComponent({
  name: "ModelFineTunerPanel",
  label: "YOLOv8 Trainer",
  component: ModelFineTunerPanel,
  type: PluginComponentType.Panel,
  activator: myActivator,
});

function myActivator({ dataset }) {
  return true;
}
