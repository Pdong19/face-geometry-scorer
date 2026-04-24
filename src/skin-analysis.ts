import type { RegionPixels } from "./skin-regions";
import { rgbToLab, rgbToHsv } from "./color-science";
import { clamp } from "./utils";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Computes the average CIELAB L* (lightness) across a region's pixels. */
export function averageLightness(region: RegionPixels): number {
  if (region.count === 0) return 0;
  let totalL = 0;
  for (let i = 0; i < region.pixels.length; i += 4) {
    totalL += rgbToLab(region.pixels[i], region.pixels[i + 1], region.pixels[i + 2]).L;
  }
  return totalL / region.count;
}

/**
 * Finds the approximate median using histogram bins.
 * O(n) instead of O(n log n) sort-based median.
 */
function histogramMedian(bins: Uint32Array, total: number): number {
  const half = total / 2;
  let cumulative = 0;
  for (let i = 0; i < bins.length; i++) {
    cumulative += bins[i];
    if (cumulative >= half) return i;
  }
  return bins.length - 1;
}

// ---------------------------------------------------------------------------
// Color Uniformity
// ---------------------------------------------------------------------------

/**
 * Measures how uniform the skin color is within a region.
 *
 * Algorithm:
 * 1. Convert every pixel to CIELAB color space.
 * 2. Compute the standard deviation of each channel:
 *    - L* (lightness): captures uneven brightness/shadows.
 *    - a* (green-red axis): captures redness variation.
 *    - b* (blue-yellow axis): captures yellowness variation.
 * 3. Combine with weights: L contributes 50%, a and b each 25%.
 *    Lightness variation is most perceptible, so it's weighted higher.
 * 4. Map to 0-100: lower std dev = more uniform = higher score.
 *
 * Calibration: scaling factor of 4.0 yields ~60-70 for average skin.
 * A perfectly uniform region scores 100; highly mottled skin scores ~30-40.
 *
 * @returns Score 0-100 where 100 is perfectly uniform color.
 */
export function computeColorUniformity(region: RegionPixels): number {
  if (region.count === 0) return 50;

  // Single-pass variance using E[X²] - E[X]² (same approach as computeTextureRoughness)
  let sumL = 0, sumL2 = 0, sumA = 0, sumA2 = 0, sumB = 0, sumB2 = 0;
  let n = 0;

  for (let i = 0; i < region.pixels.length; i += 4) {
    const lab = rgbToLab(region.pixels[i], region.pixels[i + 1], region.pixels[i + 2]);
    sumL += lab.L; sumL2 += lab.L * lab.L;
    sumA += lab.a; sumA2 += lab.a * lab.a;
    sumB += lab.b; sumB2 += lab.b * lab.b;
    n++;
  }

  if (n === 0) return 50;

  const stdL = Math.sqrt(Math.max(0, sumL2 / n - (sumL / n) ** 2));
  const stdA = Math.sqrt(Math.max(0, sumA2 / n - (sumA / n) ** 2));
  const stdB = Math.sqrt(Math.max(0, sumB2 / n - (sumB / n) ** 2));

  // Weighted combination: lightness variation is most visible
  const weightedStd = stdL * 0.5 + stdA * 0.25 + stdB * 0.25;

  // Scale factor: std dev of ~20 → score ~0, std dev of ~0 → score ~100
  // Average skin has std dev around 7-10, yielding scores of 50-65
  return clamp(Math.round(100 - weightedStd * 5.0), 0, 100);
}

// ---------------------------------------------------------------------------
// Texture Roughness
// ---------------------------------------------------------------------------

/**
 * Evaluates skin texture smoothness using the Laplacian edge detector.
 *
 * Algorithm:
 * 1. Convert region pixels to a 2D grayscale grid (within the bounding box).
 * 2. Apply a 3×3 Laplacian convolution kernel:
 *       [ 0, -1,  0]
 *       [-1,  4, -1]
 *       [ 0, -1,  0]
 *    This kernel detects edges and rapid intensity changes.
 * 3. Compute the variance of the Laplacian response across all valid pixels.
 *    - High variance = many edges = rough/porous/textured skin.
 *    - Low variance = smooth, even skin.
 * 4. Map variance to a 0-100 score.
 *
 * The Laplacian is applied only to pixels inside the polygon mask to avoid
 * edge artifacts at region boundaries.
 *
 * Calibration: variance scaling of 0.15 yields ~65-75 for normal skin,
 * ~40-50 for acne-scarred skin, ~80+ for very smooth skin.
 *
 * @returns Score 0-100 where 100 is perfectly smooth texture.
 */
export function computeTextureRoughness(region: RegionPixels): number {
  if (region.count < 9 || region.width < 3 || region.height < 3) return 50;

  const { pixels, width, height, mask } = region;

  // Build grayscale grid from the bounding box.
  // Only masked (in-polygon) pixels have meaningful values.
  const gray = new Float32Array(width * height);
  let pixelIdx = 0;
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const maskIdx = row * width + col;
      if (mask[maskIdx]) {
        const r = pixels[pixelIdx * 4];
        const g = pixels[pixelIdx * 4 + 1];
        const b = pixels[pixelIdx * 4 + 2];
        // ITU-R BT.601 luminance weights
        gray[maskIdx] = 0.299 * r + 0.587 * g + 0.114 * b;
        pixelIdx++;
      }
    }
  }

  // Defensive: verify pixel array was fully consumed (catches scan-order desync)
  if (pixelIdx !== region.count) return 50;

  // Apply 3×3 Laplacian kernel to interior pixels that are fully surrounded
  // by masked neighbors (avoids edge artifacts at polygon boundaries).
  // Variance is computed in a single pass using sum and sum-of-squares,
  // avoiding an intermediate array allocation.
  let lapSum = 0;
  let lapSumSq = 0;
  let lapCount = 0;

  for (let row = 1; row < height - 1; row++) {
    for (let col = 1; col < width - 1; col++) {
      const idx = row * width + col;
      // All 4 cardinal neighbors must be inside the mask
      if (
        !mask[idx] ||
        !mask[idx - 1] ||
        !mask[idx + 1] ||
        !mask[idx - width] ||
        !mask[idx + width]
      ) {
        continue;
      }

      const lap =
        4 * gray[idx] -
        gray[idx - 1] -
        gray[idx + 1] -
        gray[idx - width] -
        gray[idx + width];

      lapSum += lap;
      lapSumSq += lap * lap;
      lapCount++;
    }
  }

  if (lapCount === 0) return 50;

  // Variance = E[X²] - E[X]² (single-pass formula, guarded against float cancellation)
  const variance = Math.max(0, lapSumSq / lapCount - (lapSum / lapCount) ** 2);

  // Normalize variance by region resolution to prevent high-res cameras
  // from being penalized for capturing real skin texture detail.
  // Reference: ~100×100 region at typical webcam resolution.
  const referenceDensity = 10000;
  const bboxArea = region.width * region.height;
  const densityRatio = bboxArea / referenceDensity;
  const normFactor = clamp(Math.sqrt(densityRatio), 0.5, 2.0);
  const normalizedVariance = variance / normFactor;

  // Higher variance = rougher texture = lower score
  return clamp(Math.round(100 - normalizedVariance * 0.18), 0, 100);
}

// ---------------------------------------------------------------------------
// Blemish Density
// ---------------------------------------------------------------------------

/**
 * Detects blemishes (pimples, spots, discoloration) by identifying pixels
 * that deviate significantly from the region's median skin color.
 *
 * Algorithm:
 * 1. Convert all pixels to HSV color space.
 * 2. Compute the median H, S, V for the region (robust to outliers).
 * 3. Flag a pixel as a "blemish" if either:
 *    a. It is significantly redder than median (hue shifted toward red/orange
 *       by >30° AND saturation is >0.15 higher) — detects inflamed spots.
 *    b. It is significantly darker than median (value >0.25 lower) — detects
 *       dark spots, hyperpigmentation.
 * 4. Blemish ratio = flagged pixels / total pixels.
 * 5. Map to score: ratio of 0 = 100, ratio of 0.20 = 0.
 *
 * The median is used instead of mean because it's robust to the blemishes
 * themselves skewing the reference color.
 *
 * @returns Score 0-100 where 100 is blemish-free.
 */
export function computeBlemishDensity(region: RegionPixels): number {
  if (region.count === 0) return 50;

  // Use histogram bins for O(n) median instead of O(n log n) sort.
  // H: 0-359 integer bins, S/V: 0-100 bins (multiply by 100).
  const hBins = new Uint32Array(360);
  const sBins = new Uint32Array(101);
  const vBins = new Uint32Array(101);

  // Store per-pixel HSV for the second pass (blemish detection)
  const pixelCount = region.count;
  const hArr = new Float32Array(pixelCount);
  const sArr = new Float32Array(pixelCount);
  const vArr = new Float32Array(pixelCount);

  for (let i = 0, p = 0; i < region.pixels.length; i += 4, p++) {
    const hsv = rgbToHsv(region.pixels[i], region.pixels[i + 1], region.pixels[i + 2]);
    hArr[p] = hsv.h;
    sArr[p] = hsv.s;
    vArr[p] = hsv.v;
    hBins[Math.min(359, Math.round(hsv.h))]++;
    sBins[Math.min(100, Math.round(hsv.s * 100))]++;
    vBins[Math.min(100, Math.round(hsv.v * 100))]++;
  }

  const medH = histogramMedian(hBins, pixelCount);
  const medS = histogramMedian(sBins, pixelCount) / 100;
  const medV = histogramMedian(vBins, pixelCount) / 100;

  let blemishCount = 0;

  for (let i = 0; i < pixelCount; i++) {
    // Hue distance accounting for circular wraparound (0° = 360°)
    let hueDiff = Math.abs(hArr[i] - medH);
    if (hueDiff > 180) hueDiff = 360 - hueDiff;

    // Check if pixel is in the red/orange range (hue 0-40 or 340-360)
    const isReddish = hArr[i] < 40 || hArr[i] > 340;

    // Blemish condition 1: redder and more saturated (loosened thresholds
    // so subtle blemishes count — old thresholds were too strict)
    const isInflamed = isReddish && hueDiff > 15 && sArr[i] - medS > 0.08;

    // Blemish condition 2: darker than surrounding skin (lowered from 0.25)
    const isDarkSpot = medV - vArr[i] > 0.15;

    if (isInflamed || isDarkSpot) {
      blemishCount++;
    }
  }

  const blemishRatio = blemishCount / region.count;

  // Scale: 0% blemishes = 100, 20%+ blemishes = 0
  return clamp(Math.round(100 - blemishRatio * 500), 0, 100);
}

// ---------------------------------------------------------------------------
// Dark Circles
// ---------------------------------------------------------------------------

/**
 * Measures under-eye dark circles by comparing luminosity between
 * the under-eye region and a reference cheek region.
 *
 * Algorithm:
 * 1. Convert both regions to CIELAB.
 * 2. Compute average L* (lightness) for each region.
 * 3. Delta = cheek_L - underEye_L.
 *    - Positive delta = under-eyes are darker than cheeks = dark circles.
 *    - Zero or negative delta = under-eyes match or are lighter = no circles.
 * 4. Scale: delta of 0 → score 100, delta of 20+ → score ~0.
 *
 * Using the cheek as reference normalizes for overall skin tone — dark
 * circles are measured as relative darkness, not absolute.
 *
 * @param underEye - Pixel data for the under-eye region.
 * @param cheek - Pixel data for the cheek region (reference).
 * @returns Score 0-100 where 100 is no dark circles.
 */
export function computeDarkCircles(
  underEye: RegionPixels,
  cheek: RegionPixels,
  precomputedCheekAvgL?: number,
): number {
  if (underEye.count === 0 || cheek.count === 0) return 50;

  const underEyeL = averageLightness(underEye);
  const cheekL = precomputedCheekAvgL ?? averageLightness(cheek);

  // Positive delta means under-eyes are darker than cheeks
  const delta = Math.max(0, cheekL - underEyeL);

  // Scale: delta of 0 → 100, delta of 20 → 0
  return clamp(Math.round(100 - delta * 5), 0, 100);
}

// ---------------------------------------------------------------------------
// Luminosity
// ---------------------------------------------------------------------------

/**
 * Measures skin luminosity ("glow factor") from cheek lightness.
 *
 * Algorithm:
 * 1. Convert cheek pixels to CIELAB.
 * 2. Compute average L* (perceptual lightness, 0-100).
 * 3. Map L* to a score using a Gaussian bell curve centered on L* 65:
 *    - L* 35 → score ~56 (very dark / underexposed)
 *    - L* 50 → score ~88 (dark skin, healthy lighting)
 *    - L* 65 → score 100 (optimal)
 *    - L* 80 → score ~88 (light skin, healthy lighting)
 *    - L* 95 → score ~56 (very bright / overexposed)
 *
 * The bell curve ensures natural skin tone variation scores equally —
 * both dark and light skin receive similar scores when well-lit.
 * Only extreme values (underexposed or overexposed) are penalized.
 *
 * L* is used because it's perceptually linear — a 10-unit change looks
 * equally significant across the range, unlike raw RGB values.
 *
 * @param cheek - Pixel data for the cheek region.
 * @returns Score 0-100 where 100 is maximum luminosity/glow.
 */
export function computeLuminosity(cheek: RegionPixels, precomputedAvgL?: number): number {
  if (cheek.count === 0) return 50;

  const avgL = precomputedAvgL ?? averageLightness(cheek);

  // Gaussian bell curve centered on optimal lightness (L* 65).
  // Both very dark (underexposed) and very bright (overexposed) score lower,
  // but natural skin tone variation within normal lighting scores equally.
  // L* 35 → ~56, L* 50 → ~88, L* 65 → 100, L* 80 → ~88, L* 95 → ~56
  const center = 65;
  const score = 100 * Math.exp(-0.00055 * (avgL - center) ** 2);

  return clamp(Math.round(score), 0, 100);
}
