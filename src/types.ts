/** A single 3D point from the face mesh, normalized to 0-1 relative to image dimensions. */
export interface FaceLandmarkPoint {
  x: number;
  y: number;
  z: number;
}

/** 468-point face mesh from MediaPipe. */
export type FaceLandmarks = FaceLandmarkPoint[];

/** Facial geometry subscores, each 0-100. */
export interface GeometryScores {
  symmetry: number;
  proportions: number;
  jawline: number;
  facialThirds: number;
  eyeMetrics: number;
}

/** Skin quality subscores, each 0-100. */
export interface SkinScores {
  colorUniformity: number;
  textureRoughness: number;
  blemishDensity: number;
  darkCircles: number;
  luminosity: number;
}

/** Final combined score with all subscores. */
export interface CompositeScore {
  /** Overall attractiveness rating, 1-10 scale. */
  overall: number;
  subscores: GeometryScores & SkinScores;
  age?: number;
}

/** Compute aggregate skin clarity from individual skin sub-scores. */
export function computeSkinClarity(skin: SkinScores): number {
  return Math.round((skin.colorUniformity + skin.textureRoughness + skin.blemishDensity) / 3);
}

/** Complete result of face analysis. */
export interface FaceAnalysisResult {
  landmarks: FaceLandmarks;
  geometry: GeometryScores;
  skin: SkinScores;
  composite: CompositeScore;
  /** Detection confidence, 0-1. */
  confidence: number;
}
