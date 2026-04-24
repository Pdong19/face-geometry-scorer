import { loadFaceMesh } from "./mediapipe-loader";

/**
 * Starts loading the MediaPipe Face Mesh model in the background.
 * Safe to call multiple times — loadFaceMesh() deduplicates via its singleton promise.
 * Uses requestIdleCallback to avoid blocking the main thread.
 */
export function preloadModels(): void {
  const start = () => {
    loadFaceMesh().catch(() => {
      // Silent fail — model will be loaded on demand when user reaches /scan
    });
  };

  if (typeof requestIdleCallback === "function") {
    requestIdleCallback(start);
  } else {
    setTimeout(start, 2000);
  }
}
