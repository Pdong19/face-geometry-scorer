import type {
  FaceMesh,
  NormalizedLandmarkListList,
} from "@mediapipe/face_mesh";

/** Pinned to exact version for supply-chain security. */
const CDN_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/";

let loadPromise: Promise<FaceMesh> | null = null;
let pendingResolve:
  | ((results: NormalizedLandmarkListList) => void)
  | null = null;
let detecting = false;

/** Load the MediaPipe Face Mesh script from CDN (Webpack mangles the npm package). */
function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.crossOrigin = "anonymous";
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

async function createFaceMesh(): Promise<FaceMesh> {
  await loadScript(`${CDN_BASE}face_mesh.js`);

  // The CDN script attaches FaceMesh to the global window object
  type FaceMeshCtor = new (config: { locateFile: (file: string) => string }) => FaceMesh;
  const Ctor = (window as unknown as Record<string, unknown>).FaceMesh as FaceMeshCtor | undefined;
  if (!Ctor) {
    throw new Error("FaceMesh constructor not found on window after loading CDN script");
  }

  const mesh = new Ctor({
    locateFile: (file: string) => `${CDN_BASE}${file}`,
  });

  mesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  // Register onResults once during initialization
  mesh.onResults((results) => {
    if (pendingResolve) {
      pendingResolve(results.multiFaceLandmarks ?? []);
      pendingResolve = null;
    }
  });

  await mesh.initialize();
  return mesh;
}

/**
 * Loads and caches a singleton FaceMesh instance.
 * WASM binaries are fetched from jsDelivr CDN on first call.
 * Subsequent calls return the cached instance.
 */
export function loadFaceMesh(): Promise<FaceMesh> {
  if (!loadPromise) {
    loadPromise = createFaceMesh().catch((err) => {
      loadPromise = null; // Allow retry on failure
      throw new Error(
        `Failed to load MediaPipe Face Mesh: ${err instanceof Error ? err.message : String(err)}`,
      );
    });
  }
  return loadPromise;
}

/**
 * Runs face detection on a canvas element and returns raw landmark arrays.
 * Uses a module-level pendingResolve to avoid re-registering onResults each call.
 */
export async function detectLandmarks(
  faceMesh: FaceMesh,
  canvas: HTMLCanvasElement,
): Promise<NormalizedLandmarkListList> {
  if (detecting) {
    throw new Error("detectLandmarks already in progress");
  }
  detecting = true;

  const TIMEOUT_MS = 30_000;

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      detecting = false;
      pendingResolve = null;
      reject(new Error("Face detection timed out"));
    }, TIMEOUT_MS);

    pendingResolve = (results) => {
      clearTimeout(timer);
      detecting = false;
      resolve(results);
    };
    faceMesh.send({ image: canvas }).catch((err) => {
      clearTimeout(timer);
      detecting = false;
      reject(err);
    });
  });
}
