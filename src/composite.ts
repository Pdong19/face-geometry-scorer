import type { GeometryScores, SkinScores, CompositeScore } from "./types";
import { clamp } from "./utils";

/**
 * Computes the final composite attractiveness score from geometry and skin subscores.
 *
 * ## Weighted Composite
 *
 * Each subscore (0-100) is weighted according to its perceptual importance:
 *
 * | Factor              | Weight | Source                                        |
 * |---------------------|--------|-----------------------------------------------|
 * | Facial Symmetry     | 16%    | geometry.symmetry                             |
 * | Proportions (φ)     | 16%    | geometry.proportions                          |
 * | Skin Clarity        | 19%    | avg(uniformity, texture, blemish)              |
 * | Jawline Definition  | 14%    | geometry.jawline                              |
 * | Eye Metrics         | 13%    | geometry.eyeMetrics                           |
 * | Facial Thirds       |  8%    | geometry.facialThirds                         |
 * | Dark Circles        |  7%    | skin.darkCircles                              |
 * | Luminosity          |  7%    | skin.luminosity                               |
 *
 * ## Consistency Penalty
 *
 * A small penalty for high variance across subscores prevents someone with
 * one great feature and several poor ones from scoring the same as someone
 * balanced. Up to 5% penalty for extreme variance.
 *
 * ## Calibration Curve
 *
 * The raw 0-100 composite is mapped to a 1.0-10.0 scale using a sigmoid
 * (logistic) function that produces a realistic distribution:
 *
 *   score = 1 + 9 / (1 + e^(-k * (raw - midpoint)))
 *
 * Parameters:
 * - midpoint = 55 (average raw scores land around 50-60)
 * - k = 0.07 (gentler curve, wider spread across 1-10 range)
 *
 * Target distribution:
 * - ~5% score 9-10
 * - ~15% score 7-8
 * - ~50% score 5-6
 * - ~25% score 3-4
 * - ~5% score 1-2
 */
export function computeCompositeScore(
  geometry: GeometryScores,
  skin: SkinScores,
): CompositeScore {
  // Skin clarity = average of the three texture/color metrics
  const skinClarity =
    (skin.colorUniformity + skin.textureRoughness + skin.blemishDensity) / 3;

  // Weighted raw score (0-100 scale) — updated 2026-03-24
  // Skin metrics boosted from 27% to 33% total; symmetry reduced from 22% to 16%
  const raw =
    geometry.symmetry * 0.16 +
    geometry.proportions * 0.16 +
    skinClarity * 0.19 +
    geometry.jawline * 0.14 +
    geometry.eyeMetrics * 0.13 +
    geometry.facialThirds * 0.08 +
    skin.darkCircles * 0.07 +
    skin.luminosity * 0.07;

  // --- Consistency penalty ---
  // Penalise high variance across subscores. Someone with 90 symmetry but
  // 40 jawline should score lower than someone with 65 across the board.
  const allScores = [
    geometry.symmetry, geometry.proportions, geometry.jawline,
    geometry.eyeMetrics, geometry.facialThirds, skinClarity,
    skin.darkCircles, skin.luminosity,
  ];
  const mean = allScores.reduce((a, b) => a + b, 0) / allScores.length;
  const variance = allScores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / allScores.length;
  // Quadratic consistency penalty: high variance penalized up to 5%.
  // stdDev 10 → 0.4%, stdDev 30 → 3.6%, stdDev 50 → 5% (cap)
  const stdDev = Math.sqrt(variance);
  const consistencyMultiplier = clamp(1 - (stdDev ** 2) / 25000, 0.95, 1.0);

  const adjustedRaw = raw * consistencyMultiplier;

  // Sigmoid calibration: maps raw (0-100) → final (1-10)
  //
  // Logistic function: f(x) = 1 / (1 + e^(-k*(x - midpoint)))
  //
  // midpoint = 55: average raw scores land around 50-60. Midpoint of 55
  //   centers the bulk of users around 5-6, with natural spread to 3-8.
  //
  // k = 0.07: gentler curve gives wider spread across the 1-10 range,
  //   making the full scale usable rather than clustering at 3-5.
  const midpoint = 55;
  const k = 0.07;
  const sigmoid = 1 / (1 + Math.exp(-k * (adjustedRaw - midpoint)));

  // Map sigmoid output (0-1) to 1.0-10.0 scale, round to 1 decimal
  const overall = Math.round((1 + sigmoid * 9) * 10) / 10;



  return {
    overall: clamp(overall, 1.0, 10.0),
    subscores: { ...geometry, ...skin },
  };
}
