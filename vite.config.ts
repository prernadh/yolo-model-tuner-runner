import { defineConfig } from "@voxel51/fiftyone-js-plugin-build";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default defineConfig(__dirname);
