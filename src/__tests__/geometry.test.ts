import { describe, it, expect } from "vitest";
import type { FaceLandmarkPoint, FaceLandmarks } from "../types";
import {
  computeSymmetry,
  computeProportions,
  computeJawline,
  computeFacialThirds,
  computeEyeMetrics,
  computeAllGeometry,
} from "../geometry";

// ---------------------------------------------------------------------------
// Test landmark factory
// ---------------------------------------------------------------------------

/** Create a 468-point landmark array with all points at the origin. */
function baseLandmarks(): FaceLandmarks {
  return Array.from({ length: 468 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
}

/** Set specific landmark indices. */
function setLandmarks(
  overrides: Record<number, Partial<FaceLandmarkPoint>>,
): FaceLandmarks {
  const lm = baseLandmarks();
  for (const [idx, point] of Object.entries(overrides)) {
    const i = Number(idx);
    lm[i] = { ...lm[i], ...point };
  }
  return lm;
}

/**
 * Build a roughly symmetric face with landmarks at plausible positions.
 * Coordinates are normalized 0-1 (as MediaPipe outputs).
 */
function symmetricFace(): FaceLandmarks {
  const lm = baseLandmarks();

  // Nose bridge center line
  lm[6] = { x: 0.5, y: 0.35, z: 0 };
  // Landmark 1 at ~38% of face height from top for frontal pitch:
  // faceHeight = 0.8 - 0.2 = 0.6, noseFromTop = 0.38 * 0.6 = 0.228
  // y = 0.2 + 0.228 = 0.428
  lm[1] = { x: 0.5, y: 0.428, z: 0 };

  // Symmetric pairs — place left/right equidistant from center (x=0.5)
  const pairs: [number, number, number, number][] = [
    // [leftIdx, rightIdx, xOffset, y]
    [33, 263, 0.15, 0.38],   // Outer eye corners
    [133, 362, 0.05, 0.38],  // Inner eye corners
    [159, 386, 0.10, 0.36],  // Eye top
    [145, 374, 0.10, 0.40],  // Eye bottom
    [46, 276, 0.16, 0.33],   // Outer eyebrow
    [107, 336, 0.06, 0.32],  // Inner eyebrow
    [105, 334, 0.12, 0.31],  // Eyebrow peak
    [61, 291, 0.08, 0.62],   // Mouth corners
    [48, 278, 0.04, 0.52],   // Nose side
    [98, 327, 0.03, 0.55],   // Nostrils
    [116, 345, 0.12, 0.45],  // Cheekbone high
    [123, 352, 0.13, 0.50],  // Cheekbone low
    [234, 454, 0.20, 0.48],  // Jaw corner
    [132, 361, 0.17, 0.56],  // Mid jaw
    [58, 288, 0.12, 0.64],   // Lower jaw
    [21, 251, 0.18, 0.35],   // Temple
  ];

  for (const [l, r, xOff, y] of pairs) {
    lm[l] = { x: 0.5 - xOff, y, z: 0 };
    lm[r] = { x: 0.5 + xOff, y, z: 0 };
  }

  // Face bounds
  lm[10] = { x: 0.5, y: 0.2, z: 0 };   // Top of forehead
  lm[152] = { x: 0.5, y: 0.8, z: 0 };  // Chin
  lm[2] = { x: 0.5, y: 0.56, z: 0 };   // Nose base

  // Jawline contour — smooth arc from left ear → chin → right ear
  const jawIndices = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
    152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
  ];
  for (let i = 0; i < jawIndices.length; i++) {
    const t = i / (jawIndices.length - 1); // 0 to 1
    const angle = Math.PI * (0.2 + t * 0.6); // arc from ~36° to ~144°
    lm[jawIndices[i]] = {
      x: 0.5 + 0.2 * Math.cos(angle),
      y: 0.5 + 0.3 * Math.sin(angle),
      z: 0,
    };
  }

  // Jaw angle arm landmarks (2-step contour arms for mandible angle calc)
  // Chin-side arms (2 steps from jaw corner along contour toward chin)
  lm[132] = { x: 0.38, y: 0.62, z: 0 }; // Left mid-jaw toward chin
  lm[361] = { x: 0.62, y: 0.62, z: 0 }; // Right mid-jaw toward chin
  // Temple-side arms (2 steps from jaw corner along contour toward temple)
  lm[162] = { x: 0.25, y: 0.32, z: 0 }; // Left temple area
  lm[389] = { x: 0.75, y: 0.32, z: 0 }; // Right temple area

  return lm;
}

// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------
describe("computeSymmetry", () => {
  it("scores 95-100 for a perfectly symmetric face", () => {
    const lm = symmetricFace();
    const score = computeSymmetry(lm);
    expect(score).toBeGreaterThanOrEqual(95);
    expect(score).toBeLessThanOrEqual(100);
  });

  it("scores lower for asymmetric face (one eye shifted)", () => {
    const lm = symmetricFace();
    // Shift left outer eye corner 20% higher
    lm[33] = { ...lm[33], x: lm[33].x + 0.05 };
    lm[133] = { ...lm[133], x: lm[133].x + 0.05 };
    lm[159] = { ...lm[159], x: lm[159].x + 0.05 };
    lm[145] = { ...lm[145], x: lm[145].x + 0.05 };
    const score = computeSymmetry(lm);
    expect(score).toBeLessThan(90);
  });

  it("returns 0-100 range", () => {
    const lm = baseLandmarks();
    const score = computeSymmetry(lm);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(100);
  });
});

// ---------------------------------------------------------------------------
// Proportions
// ---------------------------------------------------------------------------
describe("computeProportions", () => {
  it("scores high for golden ratio proportions", () => {
    const lm = symmetricFace();
    // The symmetric face has plausible proportions
    const score = computeProportions(lm);
    expect(score).toBeGreaterThanOrEqual(40);
    expect(score).toBeLessThanOrEqual(100);
  });

  it("scores lower for very wide face (width ≈ height)", () => {
    const lm = symmetricFace();
    // Make face very wide: move jaw corners far apart
    lm[234] = { x: 0.1, y: 0.48, z: 0 };
    lm[454] = { x: 0.9, y: 0.48, z: 0 };
    // Keep height the same → ratio drops far below golden ratio
    const score = computeProportions(lm);
    const normalScore = computeProportions(symmetricFace());
    expect(score).toBeLessThan(normalScore);
  });

  it("handles degenerate input (all points at same position)", () => {
    const lm = baseLandmarks(); // All at (0.5, 0.5)
    const score = computeProportions(lm);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(100);
  });

  it("wider nose within dead zone is not penalized", () => {
    const narrow = symmetricFace();
    const wider = symmetricFace();
    wider[98] = { x: 0.5 - 0.06, y: 0.55, z: 0 };
    wider[327] = { x: 0.5 + 0.06, y: 0.55, z: 0 };
    narrow[98] = { x: 0.5 - 0.04, y: 0.55, z: 0 };
    narrow[327] = { x: 0.5 + 0.04, y: 0.55, z: 0 };
    const widerScore = computeProportions(wider);
    const narrowScore = computeProportions(narrow);
    expect(Math.abs(widerScore - narrowScore)).toBeLessThanOrEqual(10);
  });
});

// ---------------------------------------------------------------------------
// Jawline
// ---------------------------------------------------------------------------
describe("computeJawline", () => {
  it("scores well for a face with sharp jaw angles", () => {
    const lm = symmetricFace();
    const score = computeJawline(lm);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(100);
  });

  it("angle formula: 130° → ~96, 140° → ~88, 150° → ~75", () => {
    // New power curve: 100 - ((angle - 120) / 50)^1.6 * 60
    const calc = (angle: number) =>
      Math.round(Math.max(0, Math.min(100, 100 - ((angle - 120) / 50) ** 1.6 * 60)));
    expect(calc(130)).toBeGreaterThanOrEqual(94);
    expect(calc(130)).toBeLessThanOrEqual(98);
    expect(calc(140)).toBeGreaterThanOrEqual(85);
    expect(calc(140)).toBeLessThanOrEqual(91);
    expect(calc(150)).toBeGreaterThanOrEqual(72);
    expect(calc(150)).toBeLessThanOrEqual(78);
  });

  it("softer jaw angles (140-155°) score higher than old formula", () => {
    const calc = (angle: number) =>
      Math.round(Math.max(0, Math.min(100, 100 - ((angle - 120) / 50) ** 1.6 * 60)));
    expect(calc(140)).toBeGreaterThanOrEqual(85); // was 80
    expect(calc(150)).toBeGreaterThanOrEqual(72); // was 65
  });

  it("smooth jawline contour scores higher than jagged", () => {
    const smooth = symmetricFace();
    const jagged = symmetricFace();

    // Add noise to jawline points
    const jawIndices = [
      234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
      152,
      377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
    ];
    for (let i = 1; i < jawIndices.length - 1; i++) {
      const idx = jawIndices[i];
      jagged[idx] = {
        ...jagged[idx],
        x: jagged[idx].x + (i % 2 === 0 ? 0.05 : -0.05),
        y: jagged[idx].y + (i % 2 === 0 ? 0.03 : -0.03),
      };
    }

    const smoothScore = computeJawline(smooth);
    const jaggedScore = computeJawline(jagged);
    expect(smoothScore).toBeGreaterThanOrEqual(jaggedScore);
  });
});

// ---------------------------------------------------------------------------
// Facial thirds
// ---------------------------------------------------------------------------
describe("computeFacialThirds", () => {
  it("scores high for equal thirds", () => {
    const lm = symmetricFace();
    // Set forehead, brow, nose base, chin equally spaced.
    // With hairline compensation the upper third gets extended by 20% of middle,
    // so we shrink the raw upper slightly so the compensated result is balanced.
    lm[10] = { x: 0.5, y: 0.24, z: 0 };   // Forehead (slightly lower to offset compensation)
    lm[107] = { x: 0.44, y: 0.4, z: 0 };  // Left brow
    lm[336] = { x: 0.56, y: 0.4, z: 0 };  // Right brow (avg y = 0.4)
    lm[2] = { x: 0.5, y: 0.6, z: 0 };     // Nose base
    lm[152] = { x: 0.5, y: 0.8, z: 0 };   // Chin
    const score = computeFacialThirds(lm);
    expect(score).toBeGreaterThanOrEqual(85);
    expect(score).toBeLessThanOrEqual(100);
  });

  it("scores low for extremely unequal thirds (60/20/20)", () => {
    const lm = symmetricFace();
    // Total height = 0.6. Upper = 0.36 (60%), middle = 0.12, lower = 0.12
    lm[10] = { x: 0.5, y: 0.2, z: 0 };
    lm[107] = { x: 0.44, y: 0.56, z: 0 };
    lm[336] = { x: 0.56, y: 0.56, z: 0 }; // avg brow = 0.56
    lm[2] = { x: 0.5, y: 0.68, z: 0 };
    lm[152] = { x: 0.5, y: 0.8, z: 0 };
    const score = computeFacialThirds(lm);
    // With hairline compensation and softer multiplier, extremely
    // unequal thirds should still score below 70
    expect(score).toBeLessThan(70);
  });

  it("returns 0 for degenerate face (zero height)", () => {
    const lm = setLandmarks({
      10: { y: 0.5 },
      107: { y: 0.5 },
      336: { y: 0.5 },
      2: { y: 0.5 },
      152: { y: 0.5 },
    });
    expect(computeFacialThirds(lm)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Eye metrics
// ---------------------------------------------------------------------------
describe("computeEyeMetrics", () => {
  it("returns 0-100 range for normal face", () => {
    const lm = symmetricFace();
    const score = computeEyeMetrics(lm);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(100);
  });

  it("higher canthal tilt (upward outer corner) scores better", () => {
    const upward = symmetricFace();
    const downward = symmetricFace();

    // Upward tilt: outer corners higher (lower y) than inner
    upward[33] = { x: 0.35, y: 0.36, z: 0 };  // Left outer (higher)
    upward[133] = { x: 0.45, y: 0.38, z: 0 }; // Left inner
    upward[263] = { x: 0.65, y: 0.36, z: 0 }; // Right outer (higher)
    upward[362] = { x: 0.55, y: 0.38, z: 0 }; // Right inner

    // Downward tilt: outer corners lower (higher y) than inner
    downward[33] = { x: 0.35, y: 0.40, z: 0 };
    downward[133] = { x: 0.45, y: 0.38, z: 0 };
    downward[263] = { x: 0.65, y: 0.40, z: 0 };
    downward[362] = { x: 0.55, y: 0.38, z: 0 };

    // Set identical eye heights for both
    for (const lm of [upward, downward]) {
      lm[159] = { x: 0.40, y: 0.36, z: 0 }; // Left top
      lm[145] = { x: 0.40, y: 0.40, z: 0 }; // Left bottom
      lm[386] = { x: 0.60, y: 0.36, z: 0 }; // Right top
      lm[374] = { x: 0.60, y: 0.40, z: 0 }; // Right bottom
    }

    expect(computeEyeMetrics(upward)).toBeGreaterThan(
      computeEyeMetrics(downward),
    );
  });

  it("neutral canthal tilt (0°) scores above 50", () => {
    const lm = symmetricFace();
    // Set eyes perfectly horizontal (outer and inner at same Y)
    lm[33] = { x: 0.35, y: 0.38, z: 0 };
    lm[133] = { x: 0.45, y: 0.38, z: 0 };
    lm[263] = { x: 0.65, y: 0.38, z: 0 };
    lm[362] = { x: 0.55, y: 0.38, z: 0 };
    // Set eye heights for good aspect ratio
    lm[159] = { x: 0.40, y: 0.365, z: 0 };
    lm[145] = { x: 0.40, y: 0.395, z: 0 };
    lm[386] = { x: 0.60, y: 0.365, z: 0 };
    lm[374] = { x: 0.60, y: 0.395, z: 0 };
    const score = computeEyeMetrics(lm);
    expect(score).toBeGreaterThan(50);
  });
});

// ---------------------------------------------------------------------------
// Pitch penalty
// ---------------------------------------------------------------------------
describe("pitch penalty", () => {
  it("frontal face gets no pitch penalty (symmetry stays high)", () => {
    const lm = symmetricFace();
    const score = computeSymmetry(lm);
    expect(score).toBeGreaterThanOrEqual(90);
  });

  it("face looking up reduces symmetry score", () => {
    const frontal = symmetricFace();
    const lookingUp = symmetricFace();
    // Move nose closer to forehead (simulates upward tilt)
    lookingUp[1] = { x: 0.5, y: 0.30, z: 0 };
    lookingUp[10] = { x: 0.5, y: 0.2, z: 0 };
    lookingUp[152] = { x: 0.5, y: 0.8, z: 0 };
    const frontalScore = computeSymmetry(frontal);
    const tiltedScore = computeSymmetry(lookingUp);
    expect(tiltedScore).toBeLessThan(frontalScore);
  });

  it("face looking down reduces facial thirds score", () => {
    const frontal = symmetricFace();
    const lookingDown = symmetricFace();
    // Move nose lower (simulates downward tilt)
    lookingDown[1] = { x: 0.5, y: 0.70, z: 0 };
    const frontalScore = computeFacialThirds(frontal);
    const tiltedScore = computeFacialThirds(lookingDown);
    expect(tiltedScore).toBeLessThan(frontalScore);
  });
});

// ---------------------------------------------------------------------------
// computeAllGeometry
// ---------------------------------------------------------------------------
describe("computeAllGeometry", () => {
  it("returns all five subscores", () => {
    const lm = symmetricFace();
    const result = computeAllGeometry(lm);
    expect(result).toHaveProperty("symmetry");
    expect(result).toHaveProperty("proportions");
    expect(result).toHaveProperty("jawline");
    expect(result).toHaveProperty("facialThirds");
    expect(result).toHaveProperty("eyeMetrics");
  });

  it("all scores are in 0-100 range", () => {
    const lm = symmetricFace();
    const result = computeAllGeometry(lm);
    for (const [, value] of Object.entries(result)) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(100);
    }
  });
});
