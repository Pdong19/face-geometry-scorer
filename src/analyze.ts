import type {
  FaceAnalysisResult,
  FaceLandmarks,
} from "./types";
import { loadFaceMesh, detectLandmarks } from "./mediapipe-loader";
import { computeAllGeometry } from "./geometry";
import { extractSkinRegions } from "./skin-regions";
import { computeWhiteBalanceScales, applyWhiteBalance } from "./color-science";
import {
  computeColorUniformity,
  computeTextureRoughness,
  computeBlemishDensity,
  computeDarkCircles,
  computeLuminosity,
  averageLightness,
} from "./skin-analysis";
import { computeCompositeScore } from "./composite";

/** Thrown when no face is detected in the provided image. */
export class FaceNotFoundError extends Error {
  constructor() {
    super("No face detected in the image");
    this.name = "FaceNotFoundError";
  }
}

/**
 * Analyzes a face image for geometric attractiveness.
 *
 * Pipeline:
 * 1. Loads the MediaPipe Face Mesh model (cached singleton)
 * 2. Converts ImageData to a canvas for MediaPipe input
 * 3. Detects face landmarks (468 3D points)
 * 4. Runs geometry scoring: symmetry, proportions, jawline, facial thirds, eyes
 * 5. Extracts skin regions (forehead, cheeks, chin, under-eyes) via landmark polygons
 * 6. Normalizes white balance using forehead as reference
 * 7. Runs skin analysis: color uniformity, texture, blemishes, dark circles, luminosity
 * 8. Computes weighted composite score (1-10 scale) with sigmoid calibration
 *
 * @throws {FaceNotFoundError} If no face is detected in the image.
 */
export async function analyzeFace(
  imageData: ImageData,
  maxResolution?: number,
): Promise<FaceAnalysisResult> {
  const faceMesh = await loadFaceMesh();

  // Convert ImageData to canvas for MediaPipe
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Could not create canvas context");
  }
  ctx.putImageData(imageData, 0, 0);

  let tempCanvas: HTMLCanvasElement | null = null;

  try {
    // Optionally scale down for low-end devices — scale the main canvas so
    // landmark detection and skin analysis use the same coordinate space.
    if (maxResolution) {
      const maxDim = Math.max(canvas.width, canvas.height);
      if (maxDim > maxResolution) {
        const scale = maxResolution / maxDim;
        const w = Math.round(canvas.width * scale);
        const h = Math.round(canvas.height * scale);
        tempCanvas = document.createElement("canvas");
        tempCanvas.width = w;
        tempCanvas.height = h;
        const tempCtx = tempCanvas.getContext("2d");
        if (tempCtx) {
          tempCtx.drawImage(canvas, 0, 0, w, h);
          canvas.width = w;
          canvas.height = h;
          ctx.drawImage(tempCanvas, 0, 0);
        }
      }
    }

    // Multi-frame averaging: run detection 3× with 1px shifts to reduce
    // landmark jitter. Each pass shifts the canvas slightly, then offsets
    // the detected landmarks back before averaging.
    // Revert note: to restore single-pass, remove this loop and use
    // a single detectLandmarks(faceMesh, canvas) call.
    const shifts = [
      { dx: 0, dy: 0 },
      { dx: 1, dy: 0 },
      { dx: 0, dy: 1 },
    ];
    const allDetections: FaceLandmarks[] = [];

    for (const shift of shifts) {
      if (shift.dx !== 0 || shift.dy !== 0) {
        const shiftCanvas = document.createElement("canvas");
        shiftCanvas.width = canvas.width;
        shiftCanvas.height = canvas.height;
        const shiftCtx = shiftCanvas.getContext("2d");
        if (shiftCtx) {
          shiftCtx.drawImage(canvas, shift.dx, shift.dy);
          const faces = await detectLandmarks(faceMesh, shiftCanvas);
          if (faces.length > 0) {
            const offsetLandmarks: FaceLandmarks = faces[0].map((lm) => ({
              x: lm.x - shift.dx / shiftCanvas.width,
              y: lm.y - shift.dy / shiftCanvas.height,
              z: lm.z,
            }));
            allDetections.push(offsetLandmarks);
          }
        }
        // Always release shifted canvas memory
        shiftCanvas.width = 0;
        shiftCanvas.height = 0;
      } else {
        const faces = await detectLandmarks(faceMesh, canvas);
        if (faces.length === 0) {
          throw new FaceNotFoundError();
        }
        allDetections.push(
          faces[0].map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
        );
      }
    }

    if (allDetections.length === 0) {
      throw new FaceNotFoundError();
    }

    // Average landmarks across all successful detections
    const landmarks: FaceLandmarks = allDetections[0].map((_, i) => {
      let sx = 0, sy = 0, sz = 0;
      for (const det of allDetections) {
        sx += det[i].x;
        sy += det[i].y;
        sz += det[i].z;
      }
      const n = allDetections.length;
      return { x: sx / n, y: sy / n, z: sz / n };
    });

    // Run geometry scoring
    const geometry = computeAllGeometry(landmarks);

    // --- Skin analysis pipeline ---
    // 1. Extract pixel data for six face regions using landmark polygons
    const scaledImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const regions = extractSkinRegions(landmarks, scaledImageData);

    // 2. Normalize white balance using forehead as reference.
    //    Scale factors are computed once from the forehead region's average
    //    color, then applied to all other regions.
    const wbScales = computeWhiteBalanceScales(regions.forehead);
    if (wbScales) {
      applyWhiteBalance(regions.leftCheek, wbScales);
      applyWhiteBalance(regions.rightCheek, wbScales);
      applyWhiteBalance(regions.chin, wbScales);
      applyWhiteBalance(regions.underEyeLeft, wbScales);
      applyWhiteBalance(regions.underEyeRight, wbScales);
    }

    // 3. Pre-compute cheek L* averages once — reused by darkCircles + luminosity
    const cheekLAvgL = averageLightness(regions.leftCheek);
    const cheekRAvgL = averageLightness(regions.rightCheek);

    // 4. Compute skin metrics (average left/right where applicable)
    const uniformityL = computeColorUniformity(regions.leftCheek);
    const uniformityR = computeColorUniformity(regions.rightCheek);
    const colorUniformity = Math.round((uniformityL + uniformityR) / 2);

    const textureL = computeTextureRoughness(regions.leftCheek);
    const textureR = computeTextureRoughness(regions.rightCheek);
    const textureRoughness = Math.round((textureL + textureR) / 2);

    const blemishL = computeBlemishDensity(regions.leftCheek);
    const blemishR = computeBlemishDensity(regions.rightCheek);
    const blemishChin = computeBlemishDensity(regions.chin);
    const blemishDensity = Math.round((blemishL + blemishR + blemishChin) / 3);

    const darkCirclesL = computeDarkCircles(regions.underEyeLeft, regions.leftCheek, cheekLAvgL);
    const darkCirclesR = computeDarkCircles(regions.underEyeRight, regions.rightCheek, cheekRAvgL);
    const darkCircles = Math.round((darkCirclesL + darkCirclesR) / 2);

    const luminosityL = computeLuminosity(regions.leftCheek, cheekLAvgL);
    const luminosityR = computeLuminosity(regions.rightCheek, cheekRAvgL);
    const luminosity = Math.round((luminosityL + luminosityR) / 2);

    const skin = { colorUniformity, textureRoughness, blemishDensity, darkCircles, luminosity };

    // --- Composite score ---
    const composite = computeCompositeScore(geometry, skin);

    return {
      landmarks,
      geometry,
      skin,
      composite,
      confidence: 1.0,
    };
  } finally {
    // Release canvas backing stores to free GPU/CPU memory
    canvas.width = 0;
    canvas.height = 0;
    if (tempCanvas) {
      tempCanvas.width = 0;
      tempCanvas.height = 0;
    }
  }
}
