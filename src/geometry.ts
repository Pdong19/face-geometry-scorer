import type { FaceLandmarkPoint, FaceLandmarks, GeometryScores } from "./types";
import { clamp } from "./utils";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function dist2D(a: FaceLandmarkPoint, b: FaceLandmarkPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Angle (in degrees) at vertex B formed by points A-B-C. */
function angleDeg(
  a: FaceLandmarkPoint,
  b: FaceLandmarkPoint,
  c: FaceLandmarkPoint,
): number {
  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  const dot = ba.x * bc.x + ba.y * bc.y;
  const magBA = Math.hypot(ba.x, ba.y);
  const magBC = Math.hypot(bc.x, bc.y);
  if (magBA === 0 || magBC === 0) return 0;
  const cosAngle = clamp(dot / (magBA * magBC), -1, 1);
  return Math.acos(cosAngle) * (180 / Math.PI);
}

// ---------------------------------------------------------------------------
// Landmark index constants
// ---------------------------------------------------------------------------

/**
 * 16 corresponding left/right landmark pairs for symmetry measurement.
 * Each pair: [leftIndex, rightIndex] where "left" is the person's left.
 */
const SYMMETRY_PAIRS: [number, number][] = [
  [33, 263],   // Outer eye corners
  [133, 362],  // Inner eye corners
  [159, 386],  // Eye top
  [145, 374],  // Eye bottom
  [46, 276],   // Outer eyebrow
  [107, 336],  // Inner eyebrow
  [105, 334],  // Eyebrow peak
  [61, 291],   // Mouth corners
  [48, 278],   // Nose side
  [98, 327],   // Nostrils
  [116, 345],  // Cheekbone high
  [123, 352],  // Cheekbone low
  [234, 454],  // Jaw corner
  [132, 361],  // Mid jaw
  [58, 288],   // Lower jaw
  [21, 251],   // Temple
];

/** Jawline contour indices from left ear → chin → right ear. */
const JAWLINE_CONTOUR = [
  234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
  152,
  377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
];

// ---------------------------------------------------------------------------
// Head pose estimation
// ---------------------------------------------------------------------------

/**
 * Estimates yaw (left-right rotation) from landmark asymmetry.
 * When the head is turned, one side of the face is compressed.
 * Returns an asymmetry factor 0-1 (0 = perfectly frontal, 1 = extreme turn).
 */
function estimateYawAsymmetry(landmarks: FaceLandmarks): number {
  const nose = landmarks[1];
  const leftJaw = landmarks[234];
  const rightJaw = landmarks[454];
  const leftDist = Math.abs(nose.x - leftJaw.x);
  const rightDist = Math.abs(nose.x - rightJaw.x);
  const maxDist = Math.max(leftDist, rightDist);
  if (maxDist < 0.001) return 0;
  const ratio = Math.min(leftDist, rightDist) / maxDist;
  // ratio of 1.0 = frontal, 0.5 = significant turn
  // Convert to penalty: frontal → 0, turned → up to 0.5
  return clamp(1 - ratio, 0, 0.5);
}

/**
 * Estimates pitch (up-down tilt) from the nose bridge position relative
 * to face height. Landmark 1 (nose bridge) sits at ~38% of face height
 * from the top in a frontal face. Deviation indicates head tilt.
 * Returns an asymmetry factor 0-0.4 (0 = frontal, 0.4 = extreme tilt).
 */
function estimatePitchAsymmetry(landmarks: FaceLandmarks): number {
  const faceHeight = dist2D(landmarks[10], landmarks[152]);
  if (faceHeight < 0.001) return 0;
  const noseFromTop = dist2D(landmarks[10], landmarks[1]);
  const actualRatio = noseFromTop / faceHeight;
  const expectedRatio = 0.38;
  return clamp(Math.abs(actualRatio - expectedRatio), 0, 0.4);
}

// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------

/**
 * Measures bilateral facial symmetry by comparing corresponding left/right
 * landmark distances from the nose bridge center line.
 *
 * Uses both X-axis AND Y-axis deviations (not just horizontal) to detect
 * vertical asymmetries like uneven eyes or brow heights.
 *
 * Applies a head pose penalty: if the face is turned, symmetry measurement
 * is inherently less reliable and shouldn't score high.
 *
 * Calibrated so typical faces score 60-80, not 90+.
 *
 * @returns Score 0-100 where 100 is perfectly symmetric.
 */
export function computeSymmetry(landmarks: FaceLandmarks): number {
  const centerX = (landmarks[6].x + landmarks[1].x) / 2;
  const centerY = (landmarks[6].y + landmarks[1].y) / 2;

  let totalDeviation = 0;

  for (const [leftIdx, rightIdx] of SYMMETRY_PAIRS) {
    // Horizontal asymmetry (distance from center line)
    const leftDistX = Math.abs(landmarks[leftIdx].x - centerX);
    const rightDistX = Math.abs(landmarks[rightIdx].x - centerX);
    const maxDistX = Math.max(leftDistX, rightDistX);

    // Vertical asymmetry (Y-offset difference between paired landmarks)
    const leftDistY = landmarks[leftIdx].y - centerY;
    const rightDistY = landmarks[rightIdx].y - centerY;
    const yDiff = Math.abs(leftDistY - rightDistY);
    const faceHeight = Math.abs(landmarks[10].y - landmarks[152].y);
    const yDeviation = faceHeight > 0.001 ? yDiff / faceHeight : 0;

    // X deviation
    let xDeviation = 0;
    if (maxDistX > 0.001) {
      xDeviation = Math.abs(leftDistX - rightDistX) / maxDistX;
    }

    // Combined: X weighted 60%, Y weighted 40%
    totalDeviation += xDeviation * 0.6 + yDeviation * 0.4;
  }

  const avgDeviation = totalDeviation / SYMMETRY_PAIRS.length;

  // Amplify deviations: multiply by 3 instead of using raw fraction.
  // Raw avgDeviation is typically 0.03-0.12 for real faces.
  // ×3 maps that to 9-36% penalty → scores of 64-91 instead of 88-97.
  let score = clamp(Math.round(100 * (1 - avgDeviation * 3)), 0, 100);

  // Head pose penalty: reduce score if face is significantly turned
  const yawPenalty = estimateYawAsymmetry(landmarks);
  score = clamp(Math.round(score * (1 - yawPenalty * 0.3)), 0, 100);

  // Pitch penalty: reduce score if face is tilted up/down
  const pitchPenalty = estimatePitchAsymmetry(landmarks);
  score = clamp(Math.round(score * (1 - pitchPenalty * 0.3)), 0, 100);

  return score;
}

// ---------------------------------------------------------------------------
// Proportions
// ---------------------------------------------------------------------------

interface RatioSpec {
  name: string;
  ideal: number;
  weight: number;
  deadZone?: number;
}

const RATIO_SPECS: RatioSpec[] = [
  { name: "faceHeightWidth", ideal: 1.618, weight: 1.2 },
  { name: "eyeSpacingWidth", ideal: 1.0, weight: 1.0 },
  { name: "noseWidthFace", ideal: 0.26, weight: 0.8, deadZone: 0.03 },
  { name: "lowerFaceRatio", ideal: 0.33, weight: 1.0 },
  { name: "mouthNoseWidth", ideal: 1.618, weight: 0.8 },
];

/**
 * Scores facial proportions against golden ratio ideals.
 *
 * Uses a gentler deviation multiplier of 220 (down from 300) so
 * natural variation is not over-penalized. Ratios with a `deadZone`
 * (e.g. nose width ±0.03) absorb small deviations before any penalty
 * kicks in, reducing ethnic bias toward narrow-nose ideals.
 *
 * @returns Score 0-100 where 100 is perfectly proportioned.
 */
export function computeProportions(landmarks: FaceLandmarks): number {
  const faceWidth = dist2D(landmarks[234], landmarks[454]);
  const faceHeight = dist2D(landmarks[10], landmarks[152]);
  const eyeSpacing = dist2D(landmarks[133], landmarks[362]);
  const leftEyeWidth = dist2D(landmarks[33], landmarks[133]);
  const rightEyeWidth = dist2D(landmarks[263], landmarks[362]);
  const avgEyeWidth = (leftEyeWidth + rightEyeWidth) / 2;
  const noseWidth = dist2D(landmarks[98], landmarks[327]);
  const mouthWidth = dist2D(landmarks[61], landmarks[291]);
  const lowerFace = dist2D(landmarks[2], landmarks[152]);

  const measured: number[] = [
    faceWidth > 0 ? faceHeight / faceWidth : 0,
    avgEyeWidth > 0 ? eyeSpacing / avgEyeWidth : 0,
    faceWidth > 0 ? noseWidth / faceWidth : 0,
    faceHeight > 0 ? lowerFace / faceHeight : 0,
    noseWidth > 0 ? mouthWidth / noseWidth : 0,
  ];

  let weightedSum = 0;
  let totalWeight = 0;

  for (let i = 0; i < RATIO_SPECS.length; i++) {
    const spec = RATIO_SPECS[i];
    const rawDev = Math.abs(measured[i] - spec.ideal);
    const effectiveDev = spec.deadZone ? Math.max(0, rawDev - spec.deadZone) : rawDev;
    const deviation = effectiveDev / spec.ideal;
    // Gentler multiplier: 220 instead of 300
    const score = clamp(100 - deviation * 220, 0, 100);
    weightedSum += score * spec.weight;
    totalWeight += spec.weight;
  }

  return clamp(Math.round(weightedSum / totalWeight), 0, 100);
}

// ---------------------------------------------------------------------------
// Jawline
// ---------------------------------------------------------------------------

/**
 * Evaluates jawline definition by combining mandible angle sharpness (70%)
 * and contour smoothness (30%).
 *
 * Mandible angle is measured at the jaw corners (landmarks 234 and 454)
 * using arm points **2 steps away** on the face oval contour for stable
 * direction vectors. Adjacent contour points are too close together and
 * produce near-180° angles for all face shapes.
 *
 * Arm landmarks (from the face oval contour):
 *   Left:  chin-side 132 (234→93→132), temple-side 162 (234→127→162)
 *   Right: chin-side 361 (454→323→361), temple-side 389 (454→356→389)
 *
 * A sharper angle (more defined jaw corner) scores higher (power curve):
 *   120° → 100, 130° → ~96, 140° → ~88, 150° → ~75, 160° → ~58
 *
 * Smoothness is the inverse variance of second-derivative curvature along
 * the jawline contour — lower variance means a cleaner line.
 *
 * @returns Score 0-100 where 100 is a sharply defined, smooth jawline.
 */
export function computeJawline(landmarks: FaceLandmarks): number {
  // --- Mandible angle ---
  // Use landmarks 2 steps away on the face oval contour for stable vectors.
  // Left jaw angle: vertex 234, arms from 132 (toward chin) and 162 (toward temple)
  const leftAngle = angleDeg(landmarks[132], landmarks[234], landmarks[162]);
  // Right jaw angle: vertex 454, arms from 361 (toward chin) and 389 (toward temple)
  const rightAngle = angleDeg(landmarks[361], landmarks[454], landmarks[389]);
  const avgAngle = (leftAngle + rightAngle) / 2;
  // Power curve: gentler in the 135-155° range where most faces land.
  // 120° → 100, 130° → 96, 140° → 88, 150° → 75, 160° → 58, 170° → 40
  const normalizedAngle = (avgAngle - 120) / 50;
  const angleScore = clamp(
    Math.round(100 - Math.pow(Math.max(0, normalizedAngle), 1.6) * 60),
    0,
    100,
  );

  // --- Contour smoothness ---
  const contour = JAWLINE_CONTOUR.map((i) => landmarks[i]);
  const curvatures: number[] = [];
  for (let i = 1; i < contour.length - 1; i++) {
    const dx = contour[i - 1].x - 2 * contour[i].x + contour[i + 1].x;
    const dy = contour[i - 1].y - 2 * contour[i].y + contour[i + 1].y;
    curvatures.push(Math.hypot(dx, dy));
  }

  let smoothnessScore = 100;
  if (curvatures.length > 0) {
    const mean = curvatures.reduce((a, b) => a + b, 0) / curvatures.length;
    const variance =
      curvatures.reduce((sum, c) => sum + (c - mean) ** 2, 0) /
      curvatures.length;
    // Steeper penalty: 8000 instead of 6000
    smoothnessScore = clamp(Math.round(100 - variance * 8000), 0, 100);
  }

  return clamp(Math.round(angleScore * 0.7 + smoothnessScore * 0.3), 0, 100);
}

// ---------------------------------------------------------------------------
// Facial thirds
// ---------------------------------------------------------------------------

/**
 * Evaluates how evenly the face divides into three horizontal thirds:
 * - Upper: forehead (landmark 10) to brow line (avg of 107 & 336)
 * - Middle: brow line to nose base (landmark 2)
 * - Lower: nose base to chin (landmark 152)
 *
 * MediaPipe's landmark 10 sits on the upper forehead, not the hairline,
 * so the upper third is systematically undermeasured. We compensate by
 * extending the upper third by 20% of the middle third height (a
 * conservative hairline estimate) before scoring.
 *
 * Multiplier of 250 means even small deviations produce real spread:
 * 10% deviation → 75, 20% → 50, 30% → 25.
 *
 * @returns Score 0-100 where 100 is perfectly equal thirds.
 */
export function computeFacialThirds(landmarks: FaceLandmarks): number {
  const foreheadY = landmarks[10].y;
  const browY = (landmarks[107].y + landmarks[336].y) / 2;
  const noseBaseY = landmarks[2].y;
  const chinY = landmarks[152].y;

  const rawUpper = Math.abs(browY - foreheadY);
  const middle = Math.abs(noseBaseY - browY);
  const lower = Math.abs(chinY - noseBaseY);

  // Compensate for landmark 10 being below actual hairline.
  // Add ~20% of middle third as estimated forehead extension.
  const upper = rawUpper + middle * 0.2;
  const total = upper + middle + lower;

  if (total < 0.001) return 0;

  const idealThird = total / 3;
  const maxDeviation = Math.max(
    Math.abs(upper - idealThird),
    Math.abs(middle - idealThird),
    Math.abs(lower - idealThird),
  );
  const deviationPct = maxDeviation / idealThird;

  // Multiplier 250: steeper than before (was 150).
  // 10% deviation → 75, 20% → 50, 30% → 25
  const rawScore = clamp(Math.round(100 - deviationPct * 250), 0, 100);

  // Pitch penalty: tilted faces distort the thirds measurement
  const pitchPenalty = estimatePitchAsymmetry(landmarks);
  return clamp(Math.round(rawScore * (1 - pitchPenalty * 0.3)), 0, 100);
}

// ---------------------------------------------------------------------------
// Eye metrics
// ---------------------------------------------------------------------------

/**
 * Evaluates eye aesthetics by combining canthal tilt (40%),
 * eye aspect ratio (30%), and inter-eye spacing ratio (30%).
 *
 * Three components create more differentiation than just two.
 *
 * Canthal tilt: angle from inner to outer corner relative to horizontal.
 * Positive (upward outer corner) scores higher.
 *   Score = 55 + (tilt_degrees * 4), clamped 0-100.
 *   (Raised baseline to 55, reduced sensitivity to reduce looksmaxing bias)
 *
 * Aspect ratio: eye height / width. Ideal range 0.28-0.35.
 *   Score peaks at 100 in ideal range and decreases with deviation.
 *
 * Spacing ratio: distance between eyes / eye width. Ideal ≈ 1.0.
 *   Too close or too wide both penalized.
 *
 * @returns Score 0-100.
 */
export function computeEyeMetrics(landmarks: FaceLandmarks): number {
  // --- Canthal tilt ---
  const leftTilt = Math.atan2(
    landmarks[33].y - landmarks[133].y,
    landmarks[33].x - landmarks[133].x,
  ) * (180 / Math.PI);
  const rightTilt = Math.atan2(
    landmarks[263].y - landmarks[362].y,
    landmarks[263].x - landmarks[362].x,
  ) * (180 / Math.PI);
  // Negative Y means upward in image coords (Y increases downward),
  // so a negative tilt angle = upward outer corner = positive canthal tilt
  const avgTilt = -(leftTilt + rightTilt) / 2;
  // Raised baseline (55 not 40), gentler sensitivity (4 not 6).
  // Neutral tilt (0°) = 55 (above average). Positive tilt still rewarded.
  const tiltScore = clamp(Math.round(55 + avgTilt * 4), 0, 100);

  // --- Aspect ratio ---
  const leftHeight = dist2D(landmarks[159], landmarks[145]);
  const leftWidth = dist2D(landmarks[33], landmarks[133]);
  const rightHeight = dist2D(landmarks[386], landmarks[374]);
  const rightWidth = dist2D(landmarks[263], landmarks[362]);

  const leftAR = leftWidth > 0 ? leftHeight / leftWidth : 0;
  const rightAR = rightWidth > 0 ? rightHeight / rightWidth : 0;
  const avgAR = (leftAR + rightAR) / 2;

  const idealAR = 0.315;
  const arDeviation = Math.abs(avgAR - idealAR) / idealAR;
  const arScore = clamp(Math.round(100 - arDeviation * 250), 0, 100);

  // --- Inter-eye spacing ratio ---
  const eyeSpacing = dist2D(landmarks[133], landmarks[362]);
  const avgEyeWidth = (leftWidth + rightWidth) / 2;
  const spacingRatio = avgEyeWidth > 0 ? eyeSpacing / avgEyeWidth : 0;
  const spacingDev = Math.abs(spacingRatio - 1.0);
  const spacingScore = clamp(Math.round(100 - spacingDev * 300), 0, 100);

  return clamp(Math.round(tiltScore * 0.4 + arScore * 0.3 + spacingScore * 0.3), 0, 100);
}

// ---------------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------------

/** Runs all geometry computations and returns the full GeometryScores. */
export function computeAllGeometry(landmarks: FaceLandmarks): GeometryScores {
  return {
    symmetry: computeSymmetry(landmarks),
    proportions: computeProportions(landmarks),
    jawline: computeJawline(landmarks),
    facialThirds: computeFacialThirds(landmarks),
    eyeMetrics: computeEyeMetrics(landmarks),
  };
}
