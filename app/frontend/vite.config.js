import { defineConfig } from "vite";

export default defineConfig({
  root: "src",          // aquí vive tu index.html
  build: {
    outDir: "../dist",  // salida de la build
    emptyOutDir: true
  }
});
