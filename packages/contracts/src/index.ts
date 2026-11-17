export type Tool = 'remove-background' | 'upscale' | 'clean' | 'black-white' | 'vectorize';
export type Profile = 'economy' | 'balanced';
export type JobStatus = 'queued' | 'preparing' | 'loading_model' | 'processing' | 'encoding' | 'done' | 'cancelling' | 'cancelled' | 'error' | 'interrupted';

export interface ArtifactInfo {
  id: string;
  filename: string;
  mime: string;
  width: number;
  height: number;
  hasAlpha: boolean;
  path?: string;
  parentId?: string;
  rootId?: string;
  versionNumber?: number;
  operation?: Tool;
  engine?: string;
  options?: JobOptions;
}

export interface JobOptions {
  mode?: 'ai' | 'color';
  model?: 'u2netp' | 'u2net-human-seg' | 'isnet-general-use' | 'birefnet-general' | 'birefnet-hr-matting';
  maskContrast?: number;
  profile?: Profile;
  scale?: 2 | 4 | 8;
  upscaleModel?: 'general' | 'illustration' | 'vector-refine';
  detailMode?: 'faithful' | 'detailed';
  denoise?: number;
  sharpness?: number;
  backgroundEngine?: 'birefnet-hr' | 'ben2';
  refineEdges?: boolean;
  cleanStyle?: 'illustration' | 'photo';
  cleanStrength?: number;
  cleanAlpha?: boolean;
  autoContrast?: boolean;
  thresholdMode?: 'adaptive' | 'otsu' | 'manual';
  threshold?: number;
  vectorMode?: 'color' | 'bw';
  vectorEngine?: 'vtracer' | 'potrace';
  maxColors?: number;
  color?: string;
  tolerance?: number;
  softness?: number;
  connectedToEdge?: boolean;
  refinement?: 'none' | 'color' | 'person' | 'person-vitmatte' | 'portrait-vitmatte';
  edge?: 'neutral' | 'soft' | 'firm';
}

export interface JobRecord {
  id: string;
  artifactId: string;
  operation: Tool;
  options: JobOptions;
  status: JobStatus;
  phase: string;
  progress: { completedTiles?: number; totalTiles?: number } | null;
  resultArtifactId: string | null;
  error: { code: string; message: string } | null;
  createdAt: string;
  updatedAt: string;
  metrics?: { durationMs: number; peakRssBytes: number; modelId?: string };
}
