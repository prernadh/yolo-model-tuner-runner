import { defineConfig as defineViteConfig } from "vite";
import { defineConfig } from "@fiftyone/plugin-build";

const baseConfig = defineConfig(__dirname);

export default defineViteConfig({
  ...baseConfig,
  build: {
    ...baseConfig.build,
    rollupOptions: {
      ...baseConfig.build?.rollupOptions,
      external: [
        ...(Array.isArray(baseConfig.build?.rollupOptions?.external)
          ? baseConfig.build.rollupOptions.external
          : []),
        '@chainner/node-path',
        '@fiftyone/playback',
        '@fiftyone/looker',
        '@fiftyone/plugins',
        '@mui/material',
        'fast-png',
        'lru-cache',
        'monotone-convex-hull-2d',
        'three',
      ],
      output: {
        ...baseConfig.build?.rollupOptions?.output,
        interop: 'auto',
        globals: {
          ...(typeof baseConfig.build?.rollupOptions?.output === 'object' &&
              'globals' in baseConfig.build.rollupOptions.output
              ? baseConfig.build.rollupOptions.output.globals
              : {}),
          '@fiftyone/plugins': '__fop__',
          '@fiftyone/looker': '__fol__',
          '@fiftyone/playback': '__fopb__',
          '@mui/material': '__mui__',
        },
      },
      onwarn(warning: any, warn: any) {
        // Suppress worker-related warnings and CSS import issues
        if (warning.code === 'UNRESOLVED_IMPORT' &&
            (warning.message?.includes('worker') ||
             warning.source === '@fiftyone/utilities' ||
             warning.source === '@chainner/node-path')) {
          return;
        }
        // Suppress "use client" directive warnings from bundled modules
        if (warning.code === 'MODULE_LEVEL_DIRECTIVE') {
          return;
        }
        warn(warning);
      },
    },
  },
  worker: {
    rollupOptions: {
      external: [
        '@fiftyone/utilities',
        '@fiftyone/state',
        'color-string',
        'buffer',
        'path',
        '@chainner/node-path',
      ],
    },
  },
});
