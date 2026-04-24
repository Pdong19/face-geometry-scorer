import { describe, it, expect } from "vitest";
import { computeCompositeScore } from "../composite";
import type { GeometryScores, SkinScores } from "../types";

function makeScores(
  geometryValue: number,
  skinValue: number,
): { geometry: GeometryScores; skin: SkinScores } {
  return {
    geometry: {
      symmetry: geometryValue,
      proportions: geometryValue,
      jawline: geometryValue,
      facialThirds: geometryValue,
      eyeMetrics: geometryValue,
    },
    skin: {
      colorUniformity: skinValue,
      textureRoughness: skinValue,
      blemishDensity: skinValue,
      darkCircles: skinValue,
      luminosity: skinValue,
    },
  };
}

describe("computeCompositeScore", () => {
  it("all subscores at 50 → overall ~4-5 (slightly below average)", () => {
    const { geometry, skin } = makeScores(50, 50);
    const result = computeCompositeScore(geometry, skin);
    expect(result.overall).toBeGreaterThanOrEqual(3.5);
    expect(result.overall).toBeLessThanOrEqual(5.5);
  });

  it("all subscores at 70 → overall ~7-8 (above average)", () => {
    const { geometry, skin } = makeScores(70, 70);
    const result = computeCompositeScore(geometry, skin);
    expect(result.overall).toBeGreaterThanOrEqual(6.5);
    expect(result.overall).toBeLessThanOrEqual(8.5);
  });

  it("all subscores at 90 → overall ~8-9+", () => {
    const { geometry, skin } = makeScores(90, 90);
    const result = computeCompositeScore(geometry, skin);
    expect(result.overall).toBeGreaterThanOrEqual(7.5);
    expect(result.overall).toBeLessThanOrEqual(10.0);
  });

  it("all subscores at 20 → overall ~1-2 (well below average)", () => {
    const { geometry, skin } = makeScores(20, 20);
    const result = computeCompositeScore(geometry, skin);
    expect(result.overall).toBeGreaterThanOrEqual(1.0);
    expect(result.overall).toBeLessThanOrEqual(2.5);
  });

  it("overall is always in 1.0-10.0 range", () => {
    for (const val of [0, 25, 50, 75, 100]) {
      const { geometry, skin } = makeScores(val, val);
      const result = computeCompositeScore(geometry, skin);
      expect(result.overall).toBeGreaterThanOrEqual(1.0);
      expect(result.overall).toBeLessThanOrEqual(10.0);
    }
  });

  it("overall is rounded to 1 decimal place", () => {
    const { geometry, skin } = makeScores(73, 61);
    const result = computeCompositeScore(geometry, skin);
    expect(result.overall * 10).toBe(Math.round(result.overall * 10));
  });

  it("weights are applied correctly (one low subscore < all high)", () => {
    // High symmetry only, everything else low
    const highSym = makeScores(30, 30);
    highSym.geometry.symmetry = 100;

    // Low symmetry only, everything else high
    const lowSym = makeScores(100, 100);
    lowSym.geometry.symmetry = 30;

    const result1 = computeCompositeScore(highSym.geometry, highSym.skin);
    const result2 = computeCompositeScore(lowSym.geometry, lowSym.skin);

    // The second should score higher because it has more total weighted score
    // (only symmetry at 16% is low, vs. all others high)
    expect(result2.overall).toBeGreaterThan(result1.overall);
  });

  it("subscores are spread into the composite result", () => {
    const { geometry, skin } = makeScores(70, 60);
    const result = computeCompositeScore(geometry, skin);
    expect(result.subscores.symmetry).toBe(70);
    expect(result.subscores.proportions).toBe(70);
    expect(result.subscores.colorUniformity).toBe(60);
    expect(result.subscores.textureRoughness).toBe(60);
  });

  it("high-variance subscores are penalized more than before", () => {
    // Unbalanced: 95 symmetry, 35 everything else
    const unbalanced = makeScores(35, 35);
    unbalanced.geometry.symmetry = 95;

    // Balanced: all at 45 (similar raw weighted score)
    const balanced = makeScores(45, 45);

    const unbalancedResult = computeCompositeScore(unbalanced.geometry, unbalanced.skin);
    const balancedResult = computeCompositeScore(balanced.geometry, balanced.skin);

    // Balanced should score higher despite similar raw due to consistency penalty
    expect(balancedResult.overall).toBeGreaterThanOrEqual(unbalancedResult.overall);
  });

  it("monotonically increases with input scores", () => {
    const scores: number[] = [];
    for (let v = 0; v <= 100; v += 10) {
      const { geometry, skin } = makeScores(v, v);
      scores.push(computeCompositeScore(geometry, skin).overall);
    }
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i]).toBeGreaterThanOrEqual(scores[i - 1]);
    }
  });
});
