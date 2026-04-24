import type { FaceLandmarks } from "./types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Pixel data for a single skin region, plus its bounding box for spatial ops. */
export interface RegionPixels {
  /** Flat RGBA array: [r0, g0, b0, a0, r1, g1, b1, a1, ...] */
  pixels: Uint8ClampedArray;
  /** Number of pixels (pixels.length / 4). */
  count: number;
  /** Bounding box width — needed for spatial filters like Laplacian. */
  width: number;
  /** Bounding box height. */
  height: number;
  /**
   * 2D grid mask: 1 at (row * width + col) if that pixel is inside the
   * polygon, 0 otherwise. Used by spatial kernels to skip out-of-region pixels.
   */
  mask: Uint8Array;
}

/** All extracted skin regions from a single face. */
export interface SkinRegionData {
  forehead: RegionPixels;
  leftCheek: RegionPixels;
  rightCheek: RegionPixels;
  chin: RegionPixels;
  underEyeLeft: RegionPixels;
  underEyeRight: RegionPixels;
}

// ---------------------------------------------------------------------------
// Region landmark definitions
// ---------------------------------------------------------------------------

/**
 * Each region is a polygon defined by MediaPipe Face Mesh landmark indices.
 * Indices follow the canonical UV map:
 * https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
 */

/** Forehead: area between brow ridge and estimated hairline. */
export const FOREHEAD_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454,
  // across top (hairline estimate — these are the topmost tracked landmarks)
  // back across brow ridge
  234, 127, 162, 21, 54, 103, 67, 109,
];

/**
 * Left cheek (subject's left): bounded by outer eye corner, jaw, and nose.
 * Clean non-self-intersecting polygon from cheekbone → jaw → nose side.
 */
export const LEFT_CHEEK_POLY = [
  116, 123, 132, 58, 172, 150, 176, 148, // jaw to mid-chin
  49, 48, 115, // nose side
];

export const RIGHT_CHEEK_POLY = [
  345, 352, 361, 288, 397, 379, 400, 377, // jaw to mid-chin (mirrored)
  279, 278, 344, // nose side (mirrored)
];

/** Chin: jawline arc below the mouth — subset of FACEMESH face oval. */
export const CHIN_INDICES = [
  379, 378, 400, 377,    // right jaw, going down toward chin
  152,                    // chin point (bottom center)
  148, 176, 149, 150,    // left jaw, going up from chin
];

/** Under-eye left: small region directly below the left eye. */
export const UNDER_EYE_LEFT_INDICES = [
  111, 117, 118, 119, 120, 121, 128, 245, 193, 122,
];

/** Under-eye right: mirror of left. */
export const UNDER_EYE_RIGHT_INDICES = [
  340, 346, 347, 348, 349, 350, 357, 465, 417, 351,
];

// ---------------------------------------------------------------------------
// Polygon rasterization
// ---------------------------------------------------------------------------

/**
 * Point-in-polygon test using ray casting algorithm.
 * Counts horizontal ray crossings from point (px, py) to +∞.
 * Odd crossings = inside.
 */
export function pointInPolygon(
  px: number,
  py: number,
  polygon: { x: number; y: number }[],
): boolean {
  let inside = false;
  const n = polygon.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

/**
 * Extracts pixel data within a polygon defined by landmark indices.
 *
 * Steps:
 * 1. Convert normalized landmark coords (0-1) to pixel coords.
 * 2. Compute axis-aligned bounding box of the polygon.
 * 3. For each pixel in the bounding box, test if it's inside the polygon.
 * 4. Collect RGBA values for interior pixels into a flat array.
 * 5. Build a boolean mask grid for spatial operations.
 *
 * @param landmarks - 468-point face mesh (normalized 0-1 coords).
 * @param imageData - Raw pixel data from the canvas.
 * @param indices - Landmark indices forming the polygon boundary.
 * @returns RegionPixels with pixel data, count, dimensions, and mask.
 */
function extractRegion(
  landmarks: FaceLandmarks,
  imageData: ImageData,
  indices: number[],
): RegionPixels {
  const w = imageData.width;
  const h = imageData.height;
  const data = imageData.data;

  // Convert normalized landmarks to pixel coordinates
  const polygon = indices.map((i) => ({
    x: landmarks[i].x * w,
    y: landmarks[i].y * h,
  }));

  // Compute bounding box (clamped to image bounds)
  let minX = w;
  let minY = h;
  let maxX = 0;
  let maxY = 0;
  for (const p of polygon) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  minX = Math.max(0, Math.floor(minX));
  minY = Math.max(0, Math.floor(minY));
  maxX = Math.min(w - 1, Math.ceil(maxX));
  maxY = Math.min(h - 1, Math.ceil(maxY));

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;

  // Pre-allocate buffer at maximum possible size (entire bounding box).
  // Avoids dynamic array growth + copy that creates ~9.6MB garbage per analysis.
  const maxPixels = boxW * boxH;
  const buffer = new Uint8ClampedArray(maxPixels * 4);
  const mask = new Uint8Array(maxPixels);
  let writeIdx = 0;

  for (let row = minY; row <= maxY; row++) {
    for (let col = minX; col <= maxX; col++) {
      if (pointInPolygon(col, row, polygon)) {
        const srcIdx = (row * w + col) * 4;
        buffer[writeIdx] = data[srcIdx];
        buffer[writeIdx + 1] = data[srcIdx + 1];
        buffer[writeIdx + 2] = data[srcIdx + 2];
        buffer[writeIdx + 3] = data[srcIdx + 3];
        writeIdx += 4;
        mask[(row - minY) * boxW + (col - minX)] = 1;
      }
    }
  }

  const pixelCount = writeIdx / 4;

  return {
    pixels: buffer.subarray(0, writeIdx),
    count: pixelCount,
    width: boxW,
    height: boxH,
    mask,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Extracts pixel data for six skin regions using MediaPipe face landmarks.
 *
 * Each region is defined as a polygon of landmark indices. Pixels inside
 * each polygon are collected into flat RGBA arrays with spatial metadata
 * for downstream analysis (color uniformity, texture, blemishes, etc.).
 *
 * @param landmarks - 468-point face mesh (normalized 0-1 coordinates).
 * @param imageData - Raw pixel data from the source image canvas.
 * @returns SkinRegionData with pixel arrays for all six face zones.
 */
export function extractSkinRegions(
  landmarks: FaceLandmarks,
  imageData: ImageData,
): SkinRegionData {
  return {
    forehead: extractRegion(landmarks, imageData, FOREHEAD_INDICES),
    leftCheek: extractRegion(landmarks, imageData, LEFT_CHEEK_POLY),
    rightCheek: extractRegion(landmarks, imageData, RIGHT_CHEEK_POLY),
    chin: extractRegion(landmarks, imageData, CHIN_INDICES),
    underEyeLeft: extractRegion(landmarks, imageData, UNDER_EYE_LEFT_INDICES),
    underEyeRight: extractRegion(landmarks, imageData, UNDER_EYE_RIGHT_INDICES),
  };
}
