export type Tool = 'remove-background' | 'upscale' | 'clean' | 'black-white' | 'vectorize' | 'mask-correct' | 'reconstruct';
export type Profile = 'economy' | 'balanced';
export type JobStatus = 'queued' | 'preparing' | 'loading_model' | 'processing' | 'encoding' | 'done' | 'cancelling' | 'cancelled' | 'error' | 'interrupted';

export interface ArtifactInfo {
  maskSourceId?: string;
  fidelity?: { edgeAgreement: number; review: string; message: string };
  vectorReport?: { paths: number; commands: number; bytes: number };
  candidate?: boolean;
  warning?: string;
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
  useAI?: boolean;
  restorationEngine?: 'flux' | 'diffbir';
  logoText?: string;
  enhanceMode?: 'faithful' | 'reconstruct' | 'photo';
  enhance?: boolean;
  prepareBW?: boolean;
  mode?: 'ai' | 'color';
  model?: 'u2netp' | 'u2net-human-seg' | 'isnet-general-use' | 'birefnet-general' | 'birefnet-hr-matting';
  maskContrast?: number;
  profile?: Profile;
  scale?: 2 | 4 | 8;
  upscaleModel?: 'general' | 'illustration' | 'vector-refine';
  detailMode?: 'faithful' | 'detailed';
  denoise?: number;
  sharpness?: number;
  backgroundEngine?: 'auto' | 'birefnet-hr' | 'ben2' | 'hybrid' | 'finegrain' | 'fey' | 'lucida';
  analysisContrast?: boolean;
  advancedMatting?: boolean;
  quality?: 'normal' | 'detailed' | 'max';
  boxes?: number[][];
  refineEdges?: boolean;
  cleanStyle?: 'illustration' | 'photo';
  cleanStrength?: number;
  cleanDenoise?: number;
  cleanContrast?: number;
  cleanSharpness?: number;
  cleanEdges?: number;
  cleanAlpha?: boolean;
  autoContrast?: boolean;
  thresholdMode?: 'adaptive' | 'otsu' | 'manual' | 'auto' | 'sauvola' | 'solid';
  threshold?: number;
  vectorMode?: 'color' | 'bw';
  vectorEngine?: 'vtracer' | 'potrace' | 'inkvec';
  curveTolerance?: number;
  maxColors?: number;
  color?: string;
  tolerance?: number;
  softness?: number;
  connectedToEdge?: boolean;
  refinement?: 'none' | 'color' | 'person' | 'person-vitmatte' | 'portrait-vitmatte';
  edge?: 'neutral' | 'soft' | 'firm';
}

export interface JobRecord {
  startedAtEpoch?: number;
  heartbeatEpoch?: number;
  candidateArtifactIds?: string[];
  warnings?: string[];
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
