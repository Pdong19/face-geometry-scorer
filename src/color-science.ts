import type { RegionPixels } from "./skin-regions";

// ---------------------------------------------------------------------------
// CIELAB conversion
// ---------------------------------------------------------------------------

/** CIELAB color value. */
export interface LabColor {
  L: number;
  a: number;
  b: number;
}

/**
 * Converts sRGB (0-255) to CIELAB color space.
 *
 * Pipeline: sRGB → linear RGB → XYZ (D65 illuminant) → CIELAB.
 *
 * sRGB gamma removal: each channel is linearized via the IEC 61966-2-1
 * transfer function (threshold at 0.04045).
 *
 * XYZ to Lab uses the CIE 1976 formula with D65 white point
 * (Xn=95.047, Yn=100.0, Zn=108.883).
 *
 * L* range: 0 (black) to 100 (white).
 * a* range: roughly -128 to +127 (green to red).
 * b* range: roughly -128 to +127 (blue to yellow).
 */
export function rgbToLab(r: number, g: number, b: number): LabColor {
  // sRGB to linear RGB (remove gamma)
  let rLin = r / 255;
  let gLin = g / 255;
  let bLin = b / 255;

  rLin = rLin > 0.04045 ? Math.pow((rLin + 0.055) / 1.055, 2.4) : rLin / 12.92;
  gLin = gLin > 0.04045 ? Math.pow((gLin + 0.055) / 1.055, 2.4) : gLin / 12.92;
  bLin = bLin > 0.04045 ? Math.pow((bLin + 0.055) / 1.055, 2.4) : bLin / 12.92;

  // Linear RGB to XYZ (sRGB matrix, D65 illuminant)
  const x = (rLin * 0.4124564 + gLin * 0.3575761 + bLin * 0.1804375) / 0.95047;
  const y = rLin * 0.2126729 + gLin * 0.7151522 + bLin * 0.0721750;
  const z = (rLin * 0.0193339 + gLin * 0.1191920 + bLin * 0.9503041) / 1.08883;

  // XYZ to Lab (CIE 1976 formula)
  const epsilon = 0.008856; // (6/29)^3
  const kappa = 903.3; // (29/3)^3 × 3 ≈ 903.3

  const fx = x > epsilon ? Math.cbrt(x) : (kappa * x + 16) / 116;
  const fy = y > epsilon ? Math.cbrt(y) : (kappa * y + 16) / 116;
  const fz = z > epsilon ? Math.cbrt(z) : (kappa * z + 16) / 116;

  return {
    L: 116 * fy - 16,
    a: 500 * (fx - fy),
    b: 200 * (fy - fz),
  };
}

// ---------------------------------------------------------------------------
// HSV conversion
// ---------------------------------------------------------------------------

/** HSV color value. */
export interface HsvColor {
  /** Hue in degrees, 0-360. */
  h: number;
  /** Saturation, 0-1. */
  s: number;
  /** Value (brightness), 0-1. */
  v: number;
}

/**
 * Converts sRGB (0-255) to HSV color space.
 *
 * H: 0-360° (red=0, green=120, blue=240).
 * S: 0-1 (0 = achromatic, 1 = fully saturated).
 * V: 0-1 (0 = black, 1 = brightest).
 */
export function rgbToHsv(r: number, g: number, b: number): HsvColor {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;

  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;

  let h = 0;
  if (delta > 0) {
    if (max === rn) {
      h = 60 * (((gn - bn) / delta) % 6);
    } else if (max === gn) {
      h = 60 * ((bn - rn) / delta + 2);
    } else {
      h = 60 * ((rn - gn) / delta + 4);
    }
  }
  if (h < 0) h += 360;

  const s = max === 0 ? 0 : delta / max;

  return { h, s, v: max };
}

// ---------------------------------------------------------------------------
// White balance normalization
// ---------------------------------------------------------------------------

/**
 * Computes the average RGB color of a region's pixels.
 * Used as a reference for white balance correction.
 */
function averageRgb(region: RegionPixels): { r: number; g: number; b: number } {
  if (region.count === 0) return { r: 128, g: 128, b: 128 };

  let rSum = 0;
  let gSum = 0;
  let bSum = 0;

  for (let i = 0; i < region.pixels.length; i += 4) {
    rSum += region.pixels[i];
    gSum += region.pixels[i + 1];
    bSum += region.pixels[i + 2];
  }

  return {
    r: rSum / region.count,
    g: gSum / region.count,
    b: bSum / region.count,
  };
}

/** Per-channel scale factors for white balance correction. */
export interface WhiteBalanceScales {
  r: number;
  g: number;
  b: number;
}

/**
 * Computes white balance scale factors from a reference region.
 *
 * Uses the Von Kries chromatic adaptation model (diagonal transform):
 * each channel is scaled so the reference region's average color
 * maps to a neutral target (128, 128, 128).
 *
 * Compute once and pass to `applyWhiteBalance` for each region.
 *
 * @param reference - Reference region (typically forehead) for white balance.
 * @returns Scale factors, or null if reference is too dark to normalize.
 */
export function computeWhiteBalanceScales(reference: RegionPixels): WhiteBalanceScales | null {
  const avg = averageRgb(reference);

  // Avoid division by zero; skip normalization if reference is too dark
  if (avg.r < 5 || avg.g < 5 || avg.b < 5) return null;

  return {
    r: 128 / avg.r,
    g: 128 / avg.g,
    b: 128 / avg.b,
  };
}

/**
 * Applies precomputed white balance correction to a region's pixels.
 *
 * Mutates the region's pixel data in place for performance.
 * This prevents harsh blue/yellow lighting from skewing skin scores.
 *
 * @param region - Region to normalize (mutated in place).
 * @param scales - Precomputed scale factors from `computeWhiteBalanceScales`.
 */
export function applyWhiteBalance(region: RegionPixels, scales: WhiteBalanceScales): void {
  const pixels = region.pixels;
  for (let i = 0; i < pixels.length; i += 4) {
    pixels[i] = Math.min(255, Math.round(pixels[i] * scales.r));
    pixels[i + 1] = Math.min(255, Math.round(pixels[i + 1] * scales.g));
    pixels[i + 2] = Math.min(255, Math.round(pixels[i + 2] * scales.b));
  }
}
