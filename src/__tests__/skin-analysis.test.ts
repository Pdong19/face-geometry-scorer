import { describe, it, expect } from "vitest";
import { computeLuminosity, computeTextureRoughness } from "../skin-analysis";
import type { RegionPixels } from "../skin-regions";

/** Create a fake RegionPixels where all pixels have the same RGB. */
export function uniformRegion(r: number, g: number, b: number, count = 100): RegionPixels {
  const pixels = new Uint8ClampedArray(count * 4);
  for (let i = 0; i < count; i++) {
    pixels[i * 4] = r;
    pixels[i * 4 + 1] = g;
    pixels[i * 4 + 2] = b;
    pixels[i * 4 + 3] = 255;
  }
  return { pixels, count, width: 10, height: 10, mask: new Uint8Array(100).fill(1) };
}

describe("computeLuminosity", () => {
  it("scores highest for mid-range lightness (L* ~65)", () => {
    // RGB (160, 160, 160) ≈ L* 66
    const score = computeLuminosity(uniformRegion(160, 160, 160));
    expect(score).toBeGreaterThanOrEqual(95);
  });

  it("scores symmetrically for light and dark skin tones", () => {
    // Dark skin tone: RGB (100, 70, 50) ≈ L* ~33
    const darkScore = computeLuminosity(uniformRegion(100, 70, 50));
    // Light skin tone: RGB (230, 210, 200) ≈ L* ~86
    const lightScore = computeLuminosity(uniformRegion(230, 210, 200));
    // Both should be moderate, neither extremely penalized
    expect(darkScore).toBeGreaterThanOrEqual(20);
    expect(lightScore).toBeGreaterThanOrEqual(20);
    // And they should be within ~25 points of each other (was 50+ gap)
    expect(Math.abs(darkScore - lightScore)).toBeLessThan(25);
  });

  it("returns 50 for empty region", () => {
    const empty: RegionPixels = {
      pixels: new Uint8ClampedArray(0),
      count: 0, width: 0, height: 0,
      mask: new Uint8Array(0),
    };
    expect(computeLuminosity(empty)).toBe(50);
  });

  it("always returns 0-100", () => {
    for (const v of [0, 50, 100, 150, 200, 255]) {
      const score = computeLuminosity(uniformRegion(v, v, v));
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(100);
    }
  });
});

describe("computeTextureRoughness", () => {
  it("returns 50 for region with < 9 pixels", () => {
    const tiny: RegionPixels = {
      pixels: new Uint8ClampedArray(32),
      count: 8, width: 4, height: 2,
      mask: new Uint8Array(8).fill(1),
    };
    expect(computeTextureRoughness(tiny)).toBe(50);
  });

  it("returns 0-100 for any valid region", () => {
    const region = uniformRegion(128, 128, 128, 100);
    region.width = 10;
    region.height = 10;
    region.mask = new Uint8Array(100).fill(1);
    const score = computeTextureRoughness(region);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(100);
  });

  it("uniform region scores higher than noisy region", () => {
    const uniform = uniformRegion(128, 128, 128, 100);
    uniform.width = 10;
    uniform.height = 10;
    uniform.mask = new Uint8Array(100).fill(1);

    const noisy: RegionPixels = {
      pixels: new Uint8ClampedArray(400),
      count: 100, width: 10, height: 10,
      mask: new Uint8Array(100).fill(1),
    };
    for (let i = 0; i < 100; i++) {
      const v = i % 2 === 0 ? 200 : 50;
      noisy.pixels[i * 4] = v;
      noisy.pixels[i * 4 + 1] = v;
      noisy.pixels[i * 4 + 2] = v;
      noisy.pixels[i * 4 + 3] = 255;
    }

    expect(computeTextureRoughness(uniform)).toBeGreaterThan(
      computeTextureRoughness(noisy)
    );
  });

  it("high-res region is not penalized vs low-res with same pattern", () => {
    function makePattern(w: number, h: number): RegionPixels {
      const count = w * h;
      const pixels = new Uint8ClampedArray(count * 4);
      for (let i = 0; i < count; i++) {
        const v = i % 2 === 0 ? 150 : 100;
        pixels[i * 4] = v;
        pixels[i * 4 + 1] = v;
        pixels[i * 4 + 2] = v;
        pixels[i * 4 + 3] = 255;
      }
      return { pixels, count, width: w, height: h, mask: new Uint8Array(count).fill(1) };
    }
    const lowRes = makePattern(8, 8);
    const highRes = makePattern(20, 20);
    const lowScore = computeTextureRoughness(lowRes);
    const highScore = computeTextureRoughness(highRes);
    expect(highScore).toBeGreaterThanOrEqual(lowScore - 10);
  });
});
