import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import type { ArtifactInfo, JobRecord } from '../../../packages/contracts/src/index.ts';
import './styles.css';
import './gpu-upscale.css';
import { MaskEditor } from './MaskEditor.tsx';
import { BoxSelector } from './BoxSelector.tsx';

type Scale = 2 | 4 | 8;
type UpscaleModel = 'general' | 'illustration' | 'vector-refine';
type DetailMode = 'faithful' | 'detailed';
type Operation = 'upscale' | 'remove-background' | 'clean' | 'black-white' | 'vectorize' | 'reconstruct';
type Panel = Operation | 'history';
type View = 'original' | 'result' | 'compare';
type ExportFormat = 'png' | 'jpeg' | 'webp' | 'svg';
const terminal = new Set(['done', 'cancelled', 'error', 'interrupted']);
const icons = { upscale: '↗', upload: '↑', zoomOut: '−', zoomIn: '+' };

function App() {
  // O proxy do Colab publica a aplicação em /proxy/<porta>/. Usar caminhos
  // absolutos (/api/...) ignora esse prefixo e causa HTTP 404 na página externa.
  const apiBase = useRef(new URL(location.pathname.endsWith('/') ? location.pathname : `${location.pathname}/`, location.origin));
  const [token] = useState(() => {
    const value = new URLSearchParams(location.hash.slice(1)).get('token') ?? sessionStorage.getItem('superimagem-token') ?? '';
    if (value) sessionStorage.setItem('superimagem-token', value);
    history.replaceState(null, '', location.pathname);
    return value;
  });
  const api = async (url: string, init: RequestInit = {}) => {
    const endpoint = new URL(url.replace(/^\//, ''), apiBase.current);
    const response = await fetch(endpoint, { ...init, headers: { ...(init.headers ?? {}), ...(token ? { Authorization: `Bearer ${token}` } : {}) } });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      if(response.status===409&&body.activeJob?.id)setJob(body.activeJob);
      const detail = typeof body.error === 'string' ? body.error : typeof body.detail === 'string' ? body.detail : response.statusText;
      throw new Error(`HTTP ${response.status}: ${detail || 'Falha na comunicação com o servidor'}`);
    }
    return response;
  };

  const [artifact, setArtifact] = useState<ArtifactInfo | null>(null);
  const [original, setOriginal] = useState<ArtifactInfo | null>(null);
  const [versions, setVersions] = useState<ArtifactInfo[]>([]);
  const [operationSource, setOperationSource] = useState<ArtifactInfo | null>(null);
  const [beforeUrl, setBeforeUrl] = useState('');
  const [afterUrl, setAfterUrl] = useState('');
  const [beforeInspectUrl, setBeforeInspectUrl] = useState('');
  const [afterInspectUrl, setAfterInspectUrl] = useState('');
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState('');
  const [scale, setScale] = useState<Scale>(4);
  const [model, setModel] = useState<UpscaleModel>('general');
  const [detailMode, setDetailMode] = useState<DetailMode>('faithful');
  const [denoise, setDenoise] = useState(0);
  const [sharpness, setSharpness] = useState(25);
  const [useAI, setUseAI] = useState(false);
  const [logoText,setLogoText]=useState('');
  const [restorationEngine,setRestorationEngine]=useState<'flux'|'diffbir'>('flux');
  const [prepareBW,setPrepareBW]=useState(false);
  const enhanceRequest=useRef(false);
  const [operation, setOperation] = useState<Operation>('upscale');
  const [activePanel, setActivePanel] = useState<Panel>('upscale');
  const [backgroundEngine, setBackgroundEngine] = useState<'auto' | 'birefnet-hr' | 'ben2' | 'hybrid' | 'finegrain' | 'fey' | 'lucida'>('auto');
  const [analysisContrast,setAnalysisContrast]=useState(false);
  const [advancedMatting,setAdvancedMatting]=useState(false);
  const [cutoutQuality,setCutoutQuality]=useState<'normal'|'detailed'|'max'>('max');
  const [boxes,setBoxes]=useState<number[][]>([]);
  const [boxUrl,setBoxUrl]=useState('');
  const [candidates, setCandidates] = useState<ArtifactInfo[]>([]);
  const [maskEditor, setMaskEditor] = useState<{sourceUrl:string;maskUrl:string}|null>(null);
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);
  const selectionAnchor = useRef(0);
  const [compareVersion,setCompareVersion]=useState('');
  const [refineEdges, setRefineEdges] = useState(true);
  const [cleanStyle, setCleanStyle] = useState<'illustration' | 'photo'>('illustration');
  const [cleanStrength, setCleanStrength] = useState(35);
  const [thresholdMode, setThresholdMode] = useState<'adaptive' | 'otsu' | 'manual' | 'auto' | 'sauvola'>('auto');
  const [threshold, setThreshold] = useState(128);
  const [vectorMode, setVectorMode] = useState<'color' | 'bw'>('color');
  const [maxColors, setMaxColors] = useState(16);
  const [vectorEngine,setVectorEngine]=useState<'auto'|'vtracer'|'potrace'|'inkvec'>('auto');
  const [curveTolerance,setCurveTolerance]=useState(.2);
  const [exportFormat, setExportFormat] = useState<ExportFormat>('png');
  const [view, setView] = useState<View>('original');
  const [viewBg, setViewBg] = useState<'checker' | 'white' | 'black'>('checker');
  const [zoom, setZoom] = useState(100);
  const [fit, setFit] = useState(true);
  const [compare, setCompare] = useState(50);
  const [gpuLabel, setGpuLabel] = useState('VERIFICANDO');
  const [buildVersion, setBuildVersion] = useState('');
  const [drag, setDrag] = useState(false);
  const [panning, setPanning] = useState(false);
  const [slidingCompare, setSlidingCompare] = useState(false);
  const [panModifier, setPanModifier] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [navigationTrail, setNavigationTrail] = useState<string[]>([]);
  const [navigationIndex, setNavigationIndex] = useState(-1);
  const submitLock = useRef(false);
  const navigationIndexRef = useRef(-1);
  const fileInput = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const panStart = useRef({ x: 0, y: 0, left: 0, top: 0 });
  const [canvasSize, setCanvasSize] = useState({ width: 900, height: 700 });

  const running = submitting || Boolean(job && !terminal.has(job.status));
  const [now,setNow]=useState(Date.now());
  useEffect(()=>{if(!running)return;const timer=setInterval(()=>setNow(Date.now()),1000);return()=>clearInterval(timer);},[running]);
  useEffect(()=>{let stopped=false;const recover=async()=>{try{const data=await(await api('/api/jobs/active')).json();if(!stopped&&data.job&&(!job||terminal.has(job.status))){if(!artifact){const source:ArtifactInfo=await(await api(`/api/artifacts/${data.job.artifactId}`)).json();const url=await loadPreviewImage(source.id);if(stopped){revoke(url);return;}setArtifact(source);setOriginal(source);setVersions([source]);setOperationSource(source);setBeforeUrl(url);}setJob(data.job);}}catch{/* A conexão continua sendo reportada pelo polling principal. */}};void recover();const timer=setInterval(recover,4000);return()=>{stopped=true;clearInterval(timer);};},[job?.id,job?.status]);
  const resultWidth = artifact ? artifact.width * (operation === 'upscale' ? scale : 1) : 1;
  const resultHeight = artifact ? artifact.height * (operation === 'upscale' ? scale : 1) : 1;
  const beforeArtifact = operationSource ?? artifact;
  const referenceWidth = view === 'original' || !afterUrl ? beforeArtifact?.width ?? 1 : artifact?.width ?? 1;
  const referenceHeight = view === 'original' || !afterUrl ? beforeArtifact?.height ?? 1 : artifact?.height ?? 1;
  const previewRatio = Math.min(1, 1024 / Math.max(referenceWidth, referenceHeight));
  const previewWidth = referenceWidth * previewRatio;
  const previewHeight = referenceHeight * previewRatio;
  const fitScale = Math.min(Math.max(1, canvasSize.width - 32) / previewWidth, Math.max(1, canvasSize.height - 32) / previewHeight);
  const stageWidth = Math.max(1, previewWidth * (fit ? fitScale : zoom / 100));
  const outputDimensions = artifact ? `${resultWidth} × ${resultHeight}px` : '—';
  const progress = job?.status === 'done' ? 100 : job?.progress?.totalTiles ? Math.min(100, Math.round(((job.progress.completedTiles ?? 0) / job.progress.totalTiles) * 100)) : null;
  const squares = 24;
  const completedSquares = progress === null ? 0 : Math.round(progress * squares / 100);

  useEffect(() => {
    const element = canvasRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setCanvasSize({ width: entry.contentRect.width, height: entry.contentRect.height }));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const element = canvasRef.current;
    if (!element) return;
    const wheelZoom = (event: WheelEvent) => {
      if (!beforeUrl) return;
      event.preventDefault();
      const stage = element.querySelector<HTMLElement>('.image-stage');
      if (!stage) return;
      const rect = stage.getBoundingClientRect();
      const anchorX = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)));
      const anchorY = Math.max(0, Math.min(1, (event.clientY - rect.top) / Math.max(1, rect.height)));
      const currentZoom = fit ? fitScale * 100 : zoom;
      const nextZoom = Math.max(25, Math.min(400, currentZoom + (event.deltaY < 0 ? 25 : -25)));
      setFit(false);
      setZoom(nextZoom);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const nextRect = stage.getBoundingClientRect();
        element.scrollLeft += nextRect.left + anchorX * nextRect.width - event.clientX;
        element.scrollTop += nextRect.top + anchorY * nextRect.height - event.clientY;
      }));
    };
    element.addEventListener('wheel', wheelZoom, { passive: false });
    return () => element.removeEventListener('wheel', wheelZoom);
  }, [Boolean(beforeUrl), fit, fitScale, zoom]);
  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => { if (event.key === 'Control') setPanModifier(true); };
    const keyUp = (event: KeyboardEvent) => { if (event.key === 'Control') setPanModifier(false); };
    const clear = () => setPanModifier(false);
    window.addEventListener('keydown', keyDown); window.addEventListener('keyup', keyUp); window.addEventListener('blur', clear);
    return () => { window.removeEventListener('keydown', keyDown); window.removeEventListener('keyup', keyUp); window.removeEventListener('blur', clear); };
  }, []);
  useEffect(() => {
    api('/api/health').then(r => r.json()).then(data => { setGpuLabel(data.gpu?.name ? `GPU · ${data.gpu.name}` : data.engine === 'cuda' ? 'GPU CUDA' : 'CPU LOCAL'); setBuildVersion(data.version ?? ''); }).catch(() => setGpuLabel('DESCONECTADO'));
  }, []);

  const revoke = (url: string) => { if (url.startsWith('blob:')) URL.revokeObjectURL(url); };
  const loadPreviewImage = async (id: string, inspection = false) => URL.createObjectURL(await (await api(`/api/artifacts/${id}/preview${inspection ? '?inspection=1' : ''}`)).blob());
  const recordNavigation = (id: string) => {
    setNavigationTrail(current => {
      const base = current.slice(0, navigationIndexRef.current + 1);
      if (base[base.length - 1] === id) return current;
      const next = [...base, id];
      navigationIndexRef.current = next.length - 1;
      setNavigationIndex(next.length - 1);
      return next;
    });
  };
  const upload = async (file?: File) => {
    if (!file) return;
    if(running){setError('Há uma tarefa em andamento. Aguarde ou cancele antes de trocar a imagem.');return;}
    setError(''); const body = new FormData(); body.append('file', file);
    try {
      const item: ArtifactInfo = await (await api('/api/artifacts', { method: 'POST', body })).json();
      const url = await loadPreviewImage(item.id);
      revoke(beforeUrl); revoke(afterUrl); revoke(beforeInspectUrl); revoke(afterInspectUrl);
      navigationIndexRef.current = 0;
      setBoxes([]); setCandidates([]); setSelectedVersions([]);
      setArtifact(item); setOriginal(item); setVersions([item]); setNavigationTrail([item.id]); setNavigationIndex(0); setOperationSource(null); setBeforeUrl(url); setAfterUrl(''); setBeforeInspectUrl(''); setAfterInspectUrl(''); setJob(null); setView('original'); setFit(true); setZoom(100);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  useEffect(() => {
    const pasteImage = (event: ClipboardEvent) => {
      const target = event.target;
      if (target instanceof HTMLElement && (target.isContentEditable || target.closest('input, textarea'))) return;
      const item = Array.from(event.clipboardData?.items ?? []).find(candidate => candidate.type.startsWith('image/'));
      const file = item?.getAsFile();
      if (!file) return;
      event.preventDefault();
      if (submitting || (job && !terminal.has(job.status))) {
        setError('Aguarde o processamento terminar antes de colar outra imagem.');
        return;
      }
      void upload(file);
    };
    window.addEventListener('paste', pasteImage);
    return () => window.removeEventListener('paste', pasteImage);
  });
  const execute = async () => {
    if(operation==='reconstruct'&&!useAI){setError('Marque Usar IA para recriar. Ampliar e P&B continuam disponíveis sem reconstrução.');return;}
    if (!artifact || artifact.candidate || submitLock.current || (job && !terminal.has(job.status))) return;
    submitLock.current = true; setSubmitting(true);
    setError(''); setJob(null); setOperationSource(artifact); revoke(afterUrl); revoke(afterInspectUrl); setAfterUrl(''); setAfterInspectUrl(''); setView('original');
    try {
      const sourcePreview = await loadPreviewImage(artifact.id);
      revoke(beforeUrl); setBeforeUrl(sourcePreview); revoke(beforeInspectUrl); setBeforeInspectUrl('');
      const options = operation === 'reconstruct' ? { useAI, restorationEngine, enhanceMode:'reconstruct', logoText }
        : operation === 'upscale' ? { scale, upscaleModel: model, detailMode, denoise, sharpness, useAI:false, enhanceMode:'faithful', enhance:enhanceRequest.current, prepareBW }
        : operation === 'remove-background' ? { backgroundEngine, refineEdges, analysisContrast, advancedMatting, boxes, quality:cutoutQuality }
        : operation === 'clean' ? { cleanStyle, cleanStrength, cleanAlpha: true, autoContrast: true }
        : operation === 'black-white' ? { thresholdMode, threshold, adaptiveBlock: 31, adaptiveC: 7, speckle: 2, useAI:false }
        : { vectorMode, vectorEngine: vectorEngine==='auto'?(vectorMode === 'bw' ? 'potrace' : 'vtracer'):vectorEngine, curveTolerance, speckle:0, maxColors, adaptive: true, curveMode: 'spline' };
      const value = await (await api('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ artifactId: artifact.id, operation, options }) })).json();
      setJob(value);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { enhanceRequest.current=false; submitLock.current = false; setSubmitting(false); }
  };
  useEffect(() => {
    if (!job || terminal.has(job.status)) return;
    const timer = setInterval(async () => {
      try {
        const next: JobRecord = await (await api(`/api/jobs/${job.id}`)).json(); setJob(next);
        if (terminal.has(next.status)) setError('');
        if (next.status === 'done' && next.resultArtifactId) {
          if (next.candidateArtifactIds) {
            const items = await Promise.all(next.candidateArtifactIds.map(async id => (await api(`/api/artifacts/${id}`)).json() as Promise<ArtifactInfo>));
            setCandidates(items);
          }
          const result: ArtifactInfo = await (await api(`/api/artifacts/${next.resultArtifactId}`)).json();
          const url = await loadPreviewImage(next.resultArtifactId); revoke(afterUrl); setAfterUrl(url); setArtifact(result);
          if (!result.candidate) { recordNavigation(result.id); setVersions(current => current.some(item => item.id === result.id) ? current : [...current, result]); }
          setView('result'); setFit(true);
          const sourceId = next.artifactId;
          if (sourceId) Promise.all([loadPreviewImage(sourceId, true), loadPreviewImage(next.resultArtifactId, true)]).then(([sourceInspect, resultInspect]) => {
            revoke(beforeInspectUrl); revoke(afterInspectUrl); setBeforeInspectUrl(sourceInspect); setAfterInspectUrl(resultInspect);
          }).catch(() => { /* o preview leve continua disponível */ });
        }
      } catch { /* tenta novamente */ }
    }, 500);
    return () => clearInterval(timer);
  }, [job?.id, job?.status]);
  const download = async () => {
    if (!artifact) return;
    const format = artifact.mime === 'image/svg+xml' ? 'svg' : exportFormat;
    const originalName = artifact.filename.replace(/\.[^.]+$/, '').replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_').slice(0, 180) || 'imagem';
    const extension = format === 'jpeg' ? 'jpg' : format;
    const filename = `${originalName}.${extension}`;
    const endpoint = new URL(`/api/artifacts/${artifact.id}/download?format=${format}&background=%23ffffff`.replace(/^\//, ''), apiBase.current);
    const a = document.createElement('a');
    if (token) {
      const blob = await (await api(endpoint.pathname + endpoint.search)).blob();
      a.href = URL.createObjectURL(blob); a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1500);
    } else {
      a.href = endpoint.toString(); a.download = filename; document.body.appendChild(a); a.click(); a.remove();
    }
  };
  const selectVersion = async (item: ArtifactInfo, record = true) => {
    const url = await loadPreviewImage(item.id);
    revoke(beforeUrl); revoke(afterUrl); revoke(beforeInspectUrl); revoke(afterInspectUrl);
    if (record) recordNavigation(item.id);
    setArtifact(item); setOperationSource(null); setBeforeUrl(url); setAfterUrl(''); setBeforeInspectUrl(''); setAfterInspectUrl(''); setJob(null); setView('original'); setFit(true); setZoom(100);
  };
  const navigateVersion = async (direction: -1 | 1) => {
    const nextIndex = navigationIndex + direction;
    if (nextIndex < 0 || nextIndex >= navigationTrail.length) return;
    const target = versions.find(item => item.id === navigationTrail[nextIndex]);
    if (!target) return;
    navigationIndexRef.current = nextIndex; setNavigationIndex(nextIndex); await selectVersion(target, false);
  };
  const undo = () => {
    if (!artifact?.parentId) return;
    const parent = versions.find(item => item.id === artifact.parentId);
    if (parent) void selectVersion(parent);
  };
  const reset = () => { if (original) void selectVersion(original); };
  const deleteVersion = async () => {
    if (!artifact?.parentId) return;
    if (!window.confirm(`Excluir a V${artifact.versionNumber ?? versions.findIndex(item => item.id === artifact.id) + 1}? O arquivo desta versão será removido da sessão.`)) return;
    try {
      const parent = versions.find(item => item.id === artifact.parentId);
      if (!parent) throw new Error('A versão anterior não foi encontrada.');
      await api(`/api/artifacts/${artifact.id}`, { method: 'DELETE' });
      setVersions(current => current.filter(item => item.id !== artifact.id));
      const nextTrail = navigationTrail.filter(id => id !== artifact.id);
      const parentIndex = Math.max(0, nextTrail.lastIndexOf(parent.id));
      navigationIndexRef.current = parentIndex; setNavigationTrail(nextTrail); setNavigationIndex(parentIndex);
      await selectVersion(parent, false);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };
  const changeZoom = (next: number) => { setFit(false); setZoom(Math.max(25, Math.min(400, next))); };
  const fitWidth = () => changeZoom(Math.round(((canvasSize.width - 32) / Math.max(1, previewWidth)) * 100));
  const startPan = (event: React.PointerEvent<HTMLDivElement>) => {
    if (fit || !beforeUrl || (view === 'compare' && !panModifier) || ((event.target as HTMLElement).closest('.divider') && !panModifier)) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    panStart.current = { x: event.clientX, y: event.clientY, left: canvas.scrollLeft, top: canvas.scrollTop };
    setPanning(true); event.currentTarget.setPointerCapture(event.pointerId);
  };
  const movePan = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!panning || !canvasRef.current) return;
    canvasRef.current.scrollLeft = panStart.current.left - (event.clientX - panStart.current.x);
    canvasRef.current.scrollTop = panStart.current.top - (event.clientY - panStart.current.y);
  };
  const stopPan = (event: React.PointerEvent<HTMLDivElement>) => { if (panning) { setPanning(false); event.currentTarget.releasePointerCapture(event.pointerId); } };
  const updateCompare = (clientX: number) => {
    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect) return;
    setCompare(Math.max(0, Math.min(100, ((clientX - rect.left) / Math.max(1, rect.width)) * 100)));
  };
  const startCompare = (event: React.PointerEvent<HTMLDivElement>) => {
    if (panModifier) return;
    event.stopPropagation(); setSlidingCompare(true); event.currentTarget.setPointerCapture(event.pointerId); updateCompare(event.clientX);
  };
  const moveCompare = (event: React.PointerEvent<HTMLDivElement>) => { if (slidingCompare) updateCompare(event.clientX); };
  const stopCompare = (event: React.PointerEvent<HTMLDivElement>) => { if (slidingCompare) { setSlidingCompare(false); event.currentTarget.releasePointerCapture(event.pointerId); } };
  const phaseLabel: Record<string, string> = { queued: 'Na fila', preparing: 'Preparando imagem', loading_model: 'Carregando modelo na GPU', loading_background_model: 'Baixando/carregando recortador', background_inference: 'Recortando fundo na GPU', cleaning: 'Limpando imagem', black_white: 'Preparando preto e branco', vectorizing: 'Gerando curvas vetoriais', inference: 'Ampliando em blocos', reducing_tile: 'Ajustando memória da GPU', alpha: 'Preservando transparência', second_pass: 'Segunda etapa de detalhes', encoding: 'Validando e salvando resultado', cancelling: 'Cancelando', cancelled: 'Cancelado', done: 'Concluído', error: 'Falha' };
  const viewerLabel = fit ? 'AJUSTAR' : `${zoom}% PREVIEW`;
  const beforeDisplayUrl = beforeInspectUrl || beforeUrl;
  const afterDisplayUrl = afterInspectUrl || afterUrl;
  const inspectionReady = Boolean(beforeInspectUrl && afterInspectUrl);

  const operationNames: Record<Operation, string> = { upscale: 'Ampliar', 'remove-background': 'Remover fundo', clean: 'Limpar', 'black-white': 'Preto e branco', vectorize: 'Vetorizar', reconstruct:'Recriar imagem com IA' };
  const versionNumber = (item: ArtifactInfo, fallback = 0) => item.versionNumber ?? fallback;
  const versionDepth = (item: ArtifactInfo, visited = new Set<string>()): number => {
    if (!item.parentId || visited.has(item.id)) return 0;
    visited.add(item.id);
    const parent = versions.find(candidate => candidate.id === item.parentId);
    if (!parent) return 0;
    const siblings = versions.filter(candidate => candidate.parentId === item.parentId).sort((a, b) => versionNumber(a) - versionNumber(b));
    const siblingLane = Math.max(0, siblings.findIndex(candidate => candidate.id === item.id));
    return Math.min(4, versionDepth(parent, visited) + siblingLane);
  };
  const versionLabel = (item: ArtifactInfo) => {
    if (!item.parentId) return 'Imagem original';
    const options = item.options ?? {};
    if (item.operation === 'upscale') return `Ampliação ${options.scale ?? ''}× · ${options.upscaleModel === 'illustration' ? 'Ilustração' : options.upscaleModel === 'vector-refine' ? 'Refino vetorial' : 'Foto'}`;
    if (item.operation === 'remove-background') return `Fundo removido · ${item.engine ?? 'Motor não informado'}`;
    if (item.operation === 'clean') return `Limpeza · ${options.cleanStyle === 'photo' ? 'Foto' : 'Ilustração'}`;
    if (item.operation === 'black-white') return `Preto e branco · ${options.thresholdMode === 'otsu' ? 'Otsu' : options.thresholdMode === 'manual' ? 'Manual' : 'Adaptativo'}`;
    if (item.operation === 'vectorize') return `Vetor ${options.vectorMode === 'bw' ? 'preto e branco' : 'colorido'} · ${item.engine ?? 'Motor não informado'}`;
    return item.engine ?? 'Versão processada';
  };
  return <div className="app-shell upscale-only">
    <aside className="rail"><div className="brand-mark">S</div><strong className="brand-name">SUPERIMAGEM</strong><span className="brand-subtitle">Studio profissional com GPU T4</span><nav>{(['upscale', 'remove-background', 'clean', 'black-white', 'vectorize', 'reconstruct'] as Operation[]).map((tool, index) => <button key={tool} className={activePanel === tool ? 'active' : ''} title={operationNames[tool]} onClick={() => { setOperation(tool); setActivePanel(tool); }}><span>{['↗', '✂', '✦', '◐', '◇','IA'][index]}</span><small>{tool==='reconstruct'?'IA':operationNames[tool]}</small></button>)}<button className={activePanel === 'history' ? 'active' : ''} title="Histórico" onClick={() => setActivePanel('history')}><span>☷</span><small>Histórico</small></button></nav></aside>
    <section className="side-panel" onClickCapture={event=>{const target=(event.target as HTMLElement).closest<HTMLButtonElement>('.history-state');if(!target)return;const buttons=Array.from(target.parentElement!.querySelectorAll('.history-state'));const index=buttons.indexOf(target);const ordered=[...versions].sort((a,b)=>(a.versionNumber??1)-(b.versionNumber??1));const item=ordered[index];if(!item)return;if(event.shiftKey){event.preventDefault();event.stopPropagation();setSelectedVersions(ordered.slice(Math.min(selectionAnchor.current,index),Math.max(selectionAnchor.current,index)+1).filter(value=>value.parentId).map(value=>value.id));}else if(event.ctrlKey||event.metaKey){event.preventDefault();event.stopPropagation();if(item.parentId)setSelectedVersions(current=>current.includes(item.id)?current.filter(id=>id!==item.id):[...current,item.id]);selectionAnchor.current=index;}else{setSelectedVersions(item.parentId?[item.id]:[]);selectionAnchor.current=index;}}}>
      <header><div><p>SUPERIMAGEM STUDIO</p><h1>{activePanel === 'history' ? 'Histórico' : operationNames[operation]}</h1></div><span className={`local-badge ${gpuLabel === 'DESCONECTADO' ? 'offline' : ''}`}>● {gpuLabel}</span></header>
      {activePanel !== 'history' && <div className="field"><label>IMAGEM</label><button className="file-button" onClick={() => fileInput.current?.click()}>{icons.upload} {artifact ? artifact.filename : 'Escolher imagem'}</button><input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={e => upload(e.target.files?.[0])}/>{artifact && <small>{artifact.width} × {artifact.height}px · {artifact.hasAlpha ? 'transparência será preservada' : 'imagem opaca'}</small>}</div>}
      {activePanel === 'upscale' && <><div className="field"><label>NÍVEL DE AUMENTO</label><div className="scale-grid three"><button className={scale === 2 ? 'selected' : ''} onClick={() => setScale(2)}><b>2×</b><small>4× pixels</small></button><button className={scale === 4 ? 'selected' : ''} onClick={() => setScale(4)}><b>4×</b><small>16× pixels</small></button><button className={scale === 8 ? 'selected' : ''} onClick={() => setScale(8)}><b>8×</b><small>64× pixels</small></button></div></div><div className="field"><label>TIPO DE IMAGEM</label><div className="segmented model-options"><button className={model === 'general' ? 'selected' : ''} onClick={() => setModel('general')}>Foto / geral</button><button className={model === 'illustration' ? 'selected' : ''} onClick={() => setModel('illustration')}>Ilustração / estampa</button><button className={model === 'vector-refine' ? 'selected' : ''} onClick={() => setModel('vector-refine')}>Refinar logo / letras</button></div><small>{model === 'general' ? 'RealESRGAN x4plus oficial para fotos e uso geral.' : model === 'illustration' ? 'RealESRGAN Anime 6B para traços, estampas e cores chapadas.' : 'SwinIR Classical oficial: recuperação fiel de bordas, letras e logotipos antes de vetorizar.'}</small></div>{scale === 8 && <div className="field"><label>MODO 8×</label><div className="segmented"><button className={detailMode === 'faithful' ? 'selected' : ''} onClick={() => setDetailMode('faithful')}>Fiel</button><button className={detailMode === 'detailed' ? 'selected' : ''} onClick={() => setDetailMode('detailed')}>Detalhado</button></div></div>}</>}
      {activePanel === 'remove-background' && <div className="field"><label>MOTOR DE RECORTE</label><div className="segmented"><button className={backgroundEngine === 'birefnet-hr' ? 'selected' : ''} onClick={() => setBackgroundEngine('birefnet-hr')}>BiRefNet HR</button><button className={backgroundEngine === 'ben2' ? 'selected' : ''} onClick={() => setBackgroundEngine('ben2')}>BEN2</button></div><small>Escolha o motor ou use a comparação automática abaixo.</small><label className="check-row"><input type="checkbox" checked={refineEdges} disabled={backgroundEngine!=='ben2'&&backgroundEngine!=='auto'&&backgroundEngine!=='hybrid'} onChange={e => setRefineEdges(e.target.checked)}/> Refinamento interno do BEN2</label><small>Para refinamento real do alpha, use ViTMatte nas opções especializadas.</small></div>}
      {activePanel === 'clean' && <div className="field"><label>TIPO DE LIMPEZA</label><div className="segmented"><button className={cleanStyle === 'illustration' ? 'selected' : ''} onClick={() => setCleanStyle('illustration')}>Ilustração</button><button className={cleanStyle === 'photo' ? 'selected' : ''} onClick={() => setCleanStyle('photo')}>Foto</button></div><label>FORÇA · {cleanStrength}%</label><input type="range" min="0" max="100" value={cleanStrength} onChange={e => setCleanStrength(+e.target.value)}/><small>Remove ruído, limpa pixels transparentes e aplica contraste conservador.</small></div>}
      {activePanel === 'black-white' && <div className="field"><label>CONVERSÃO</label><div className="segmented"><button className={thresholdMode === 'adaptive' ? 'selected' : ''} onClick={() => setThresholdMode('adaptive')}>Adaptativo</button><button className={thresholdMode === 'otsu' ? 'selected' : ''} onClick={() => setThresholdMode('otsu')}>Otsu</button><button className={thresholdMode === 'manual' ? 'selected' : ''} onClick={() => setThresholdMode('manual')}>Manual</button></div>{thresholdMode === 'manual' && <><label>LIMIAR · {threshold}</label><input type="range" min="0" max="255" value={threshold} onChange={e => setThreshold(+e.target.value)}/></>}<small>Prepara arte limpa para silk, estampa e vetorização.</small></div>}
      {activePanel === 'vectorize' && <div className="field"><label>TIPO DE VETOR</label><div className="segmented"><button className={vectorMode === 'color' ? 'selected' : ''} onClick={() => setVectorMode('color')}>Colorido</button><button className={vectorMode === 'bw' ? 'selected' : ''} onClick={() => setVectorMode('bw')}>Preto e branco</button></div>{vectorMode === 'color' && <><label>MÁXIMO DE CORES · {maxColors}</label><input type="range" min="2" max="64" value={maxColors} onChange={e => setMaxColors(+e.target.value)}/></>}<small>{vectorMode === 'color' ? 'VTracer com curvas spline e recorte sem emendas.' : 'Potrace dedicado a formas monocromáticas limpas.'}</small></div>}
      {activePanel !== 'history' && <div className="field active-version compact-active"><div className="compact-active-head"><label>ENTRADA DESTA OPERAÇÃO</label><div className="history-nav"><button disabled={navigationIndex <= 0} onClick={() => void navigateVersion(-1)} title="Voltar para a versão vista antes">←</button><button disabled={navigationIndex < 0 || navigationIndex >= navigationTrail.length - 1} onClick={() => void navigateVersion(1)} title="Avançar novamente">→</button></div></div><div className="version-current"><span>{artifact?.versionNumber ?? (versions.findIndex(item => item.id === artifact?.id) + 1 || 0)}</span><div><strong>{artifact ? (artifact.parentId ? `V${artifact.versionNumber ?? '?'}` : 'Original') : 'Nenhuma imagem'}</strong><small>{artifact ? versionLabel(artifact) : 'Escolha uma imagem'}</small></div></div><button className="open-history" onClick={() => setActivePanel('history')}>☷ Abrir histórico · {versions.length} {versions.length === 1 ? 'versão' : 'versões'}</button></div>}
      {activePanel === 'history' && <div className="history-panel"><div className="history-mode"><b>Histórico não linear ativo</b><small>Escolha qualquer estado. As versões posteriores são preservadas como outro ramo.</small></div>{versions.length === 0 ? <div className="history-empty">Envie uma imagem para iniciar o histórico.</div> : <div className="history-list">{[...versions].sort((a, b) => versionNumber(a) - versionNumber(b)).map((item, index) => { const number = versionNumber(item, index + 1); const parent = versions.find(candidate => candidate.id === item.parentId); const active = item.id === artifact?.id; const depth = versionDepth(item); return <button key={item.id} className={`history-state ${active ? 'active' : ''} ${depth ? 'branched' : ''}`} style={{ '--history-indent': `${depth * 12}px` } as React.CSSProperties} onClick={() => void selectVersion(item)}><i/><span className="history-number">{number === 1 ? 'O' : number}</span><span className="history-info"><strong>{number === 1 ? 'Original' : `V${number} · ${versionLabel(item)}`}</strong><small>{item.width} × {item.height}px{parent ? ` · nasceu da ${parent.parentId ? `V${parent.versionNumber}` : 'Original'}` : ''}</small></span>{active && <em>ATIVA</em>}</button>; })}</div>}<div className="history-selection"><span>Entrada selecionada</span><b>{artifact ? (artifact.parentId ? `V${artifact.versionNumber}` : 'Original') : 'Nenhuma'}</b><small>{artifact?.parentId ? `A próxima operação criará uma nova versão a partir da V${artifact.versionNumber}.` : 'A próxima operação partirá da imagem original.'}</small></div><div className="version-actions history-actions"><button disabled={!artifact?.parentId} onClick={undo}>↶ Versão-pai</button><button disabled={!original || artifact?.id === original.id} onClick={reset}>Ir à original</button><button className="delete-version" disabled={!artifact?.parentId} onClick={() => void deleteVersion()}>Excluir versão ativa</button></div></div>}
      {activePanel !== 'history' && <div className="field output-summary"><p className="dimension">Próxima saída: <strong>{outputDimensions}</strong></p><small>A operação usará exatamente a entrada indicada acima.</small></div>}
      {activePanel !== 'history' && <div className="field format-field"><label>BAIXAR VERSÃO ATIVA</label>{artifact?.mime === 'image/svg+xml' ? <div className="format-switch"><button className="selected">SVG</button></div> : <div className="format-switch"><button className={exportFormat === 'png' ? 'selected' : ''} onClick={() => setExportFormat('png')}>PNG</button><button className={exportFormat === 'jpeg' ? 'selected' : ''} onClick={() => setExportFormat('jpeg')}>JPEG</button><button className={exportFormat === 'webp' ? 'selected' : ''} disabled={Boolean(artifact && Math.max(artifact.width, artifact.height) > 16383)} onClick={() => setExportFormat('webp')}>WebP</button></div>}<small>{artifact?.mime === 'image/svg+xml' ? 'Vetor editável e escalável sem perda.' : exportFormat === 'png' ? 'Sem perdas e com transparência preservada.' : exportFormat === 'jpeg' ? 'Arquivo compacto com fundo branco e qualidade 95.' : 'Arquivo compacto Q95, preservando transparência.'}</small></div>}
      {activePanel === 'upscale' && <div className="field"><label>REDUZIR RUÍDO · {denoise}%</label><input type="range" min="0" max="100" value={denoise} onChange={e => setDenoise(+e.target.value)}/><label>NITIDEZ EXTRA · {sharpness}%</label><input type="range" min="0" max="100" value={sharpness} onChange={e => setSharpness(+e.target.value)}/><small>Tratamento suave de luminância; transparência preservada. Zero desativa.</small></div>}
      {activePanel === 'reconstruct' && <div className="field"><label className="check-row"><input type="checkbox" checked={useAI} onChange={e => setUseAI(e.target.checked)}/> Usar IA · recriar imagem</label><select value={restorationEngine} onChange={e=>setRestorationEngine(e.target.value as typeof restorationEngine)}><option value="flux">FLUX Klein · reconstrução de arte</option><option value="diffbir">DiffBIR · restauração</option></select><small>Gera um candidato no mesmo tamanho da entrada, sem executar upscale em seguida. Pode alterar fontes: confira antes de aceitar. Para preservar letras use Ampliar sem reconstrução.</small></div>}
      {artifact?.candidate && <div className="field"><small>{artifact.warning}</small><button disabled={running} onClick={async () => { try { const accepted: ArtifactInfo = await (await api(`/api/artifacts/${artifact.id}/accept`, { method: 'POST' })).json(); await Promise.allSettled(candidates.filter(item => item.id !== accepted.id).map(item => api(`/api/artifacts/${item.id}`, { method: 'DELETE' }))); setCandidates([]); setArtifact(accepted); setVersions(current => [...current, accepted]); recordNavigation(accepted.id); } catch (e) { setError(String(e)); } }}>✓ Aceitar candidato selecionado</button><button disabled={running} onClick={async () => { if (!operationSource) return; try { const ids = new Set([artifact.id, ...candidates.map(item => item.id)]); await Promise.all([...ids].map(id => api(`/api/artifacts/${id}`, { method: 'DELETE' }))); setCandidates([]); await selectVersion(operationSource, false); } catch (e) { setError(String(e)); } }}>Descartar candidatos</button><button onClick={async () => { if (!operationSource) return; try { const url = await loadPreviewImage(operationSource.id); revoke(beforeUrl); revoke(beforeInspectUrl); setBeforeUrl(url); setBeforeInspectUrl(''); setView('original'); } catch (e) { setError(String(e)); } }}>Ver imagem de entrada</button></div>}
      {activePanel === 'black-white' && <div className="field"><label>PREPARAÇÃO DE LOGOS</label><div className="segmented"><button className={thresholdMode === 'auto' ? 'selected' : ''} onClick={() => setThresholdMode('auto')}>Automático</button><button className={thresholdMode === 'sauvola' ? 'selected' : ''} onClick={() => setThresholdMode('sauvola')}>Sauvola</button></div><small>Automático avalia iluminação e escolhe Otsu ou Sauvola. Confira acentos e furos antes de vetorizar.</small></div>}
      {activePanel === 'remove-background' && <div className="field"><button className={backgroundEngine === 'auto' ? 'selected' : ''} onClick={() => setBackgroundEngine('auto')}>Automático · comparar BEN2 / BiRefNet</button><small>Execução sequencial; aceite o candidato preferido antes de continuar. Nenhuma fusão automática.</small></div>}
      {candidates.length > 0 && <div className="field"><label>CANDIDATOS DE RECORTE</label>{candidates.map(item => <button key={item.id} onClick={async () => { try { const url = await loadPreviewImage(item.id); revoke(afterUrl); setAfterUrl(url); setArtifact(item); setView('result'); } catch (e) { setError(String(e)); } }}>{item.engine}</button>)}{candidates.length === 2 && <button onClick={async () => { try { const [a,b] = await Promise.all(candidates.map(item => loadPreviewImage(item.id, true))); revoke(beforeUrl); revoke(afterUrl); revoke(beforeInspectUrl); revoke(afterInspectUrl); setBeforeUrl(a); setBeforeInspectUrl(''); setAfterUrl(b); setAfterInspectUrl(''); setArtifact(candidates[1]); setView('compare'); } catch (e) { setError(String(e)); } }}>Comparar A/B · esquerda BEN2 / direita BiRefNet</button>}{job?.warnings?.map(text => <small key={text}>{text}</small>)}</div>}
      {artifact?.hasAlpha && !artifact.candidate && <div className="field"><button onClick={async()=>{try{const sourceId=artifact.maskSourceId??artifact.parentId??artifact.id; const [sourceUrl,maskUrl]=await Promise.all([loadPreviewImage(sourceId),loadPreviewImage(artifact.id)]);setMaskEditor({sourceUrl,maskUrl});}catch(e){setError(String(e));}}}>Pincéis · Manter / Remover</button></div>}
      {artifact?.fidelity && <details className="field"><summary>Verificação de contornos · {Math.round(artifact.fidelity.edgeAgreement*100)}%</summary><small>{artifact.fidelity.message}</small>{artifact.fidelity.review==='divergence'&&<strong>Divergência significativa: confira letras e desenhos.</strong>}</details>}
      {artifact?.vectorReport&&<div className="field"><small>SVG: {artifact.vectorReport.paths} caminhos · {artifact.vectorReport.commands} comandos · {(artifact.vectorReport.bytes/1024).toFixed(1)} KiB</small></div>}
      <details className="field"><summary>Cache de modelos no Google Drive</summary><small>Monte o Drive no Colab. Caminho: MyDrive/SuperImagemCache. A ativação não copia resultados nem envia imagens.</small><button onClick={async()=>{try{const result=await(await api('/api/cache',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:true})})).json();setError(`Cache ativo: ${result.path}`);}catch(e){setError(String(e));}}}>Ativar cache Drive</button><button onClick={async()=>{try{await api('/api/cache',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:false})});setError('Cache temporário da sessão ativado.');}catch(e){setError(String(e));}}}>Usar cache temporário</button></details>
      {activePanel==='history'&&<div className="field"><small>Ctrl: seleção separada · Shift: intervalo. {selectedVersions.length} selecionadas.</small><button onClick={()=>setSelectedVersions(versions.filter(item=>item.parentId).map(item=>item.id))}>Selecionar todas as versões</button><button onClick={()=>setSelectedVersions([])}>Limpar seleção</button><div style={{display:'flex',gap:4,flexWrap:'wrap'}}>{selectedVersions.map(id=><span key={id} style={{border:'1px solid #ff684a',padding:4,borderRadius:4}}>V{versions.find(item=>item.id===id)?.versionNumber}</span>)}</div><button disabled={!selectedVersions.length||running} onClick={async()=>{try{const result=await(await api('/api/artifacts/delete-batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:selectedVersions})})).json();const deleted=new Set<string>(result.deletedIds);setVersions(items=>items.filter(item=>!deleted.has(item.id)));setSelectedVersions([]);if(original){setNavigationTrail([original.id]);navigationIndexRef.current=0;setNavigationIndex(0);await selectVersion(original,false);}}catch(e){setError(String(e));}}}>🗑 Excluir selecionadas</button>{error&&<small>{error}</small>}</div>}
      {activePanel==='remove-background'&&<div className="field"><label>MOTORES ESPECIALIZADOS · SOB DEMANDA</label><select value={backgroundEngine} onChange={e=>setBackgroundEngine(e.target.value as typeof backgroundEngine)}><option value="auto">Automático · BEN2 / BiRefNet A/B</option><option value="ben2">BEN2</option><option value="birefnet-hr">BiRefNet HR-Matting</option><option value="finegrain">Finegrain · objeto selecionado</option><option value="fey">FeyNoBg · cabelo e fios</option><option value="lucida">Lucida · logo / ilustração</option><option value="hybrid">Híbrido BEN2 + BiRefNet · experimental</option></select><button disabled={!artifact||running} onClick={async()=>{if(!artifact)return;try{setBoxUrl(await loadPreviewImage(artifact.id));}catch(e){setError(String(e));}}}>Desenhar caixas · {boxes.length} regiões</button><label className="check-row"><input type="checkbox" checked={analysisContrast} onChange={e=>setAnalysisContrast(e.target.checked)}/> Contraste somente para análise BEN2/BiRefNet</label><label className="check-row"><input type="checkbox" checked={advancedMatting} onChange={e=>setAdvancedMatting(e.target.checked)}/> Refinar alpha com ViTMatte</label><small>ViTMatte preserva áreas certas e estima transparência na faixa incerta. Os motores extras ainda precisam de validação T4.</small></div>}
      {activePanel==='vectorize'&&<div className="field"><label>MOTOR SVG</label><select value={vectorEngine} onChange={e=>setVectorEngine(e.target.value as typeof vectorEngine)}><option value="auto">Automático · Potrace P&B / VTracer cor</option><option value="potrace" disabled={vectorMode!=='bw'}>Potrace · preto e branco</option><option value="vtracer">VTracer</option><option value="inkvec">Inkvec · arte plana / logos</option></select><label>SIMPLIFICAÇÃO POTRACE · {curveTolerance}</label><input type="range" min=".01" max=".8" step=".01" value={curveTolerance} onChange={e=>setCurveTolerance(+e.target.value)}/><small>Menor tolerância preserva mais detalhes; maior reduz complexidade. Execute motores diferentes a partir da mesma versão para comparar no histórico.</small></div>}
      {activePanel==='upscale'&&<div className="field"><label className="check-row"><input type="checkbox" checked={prepareBW} onChange={e=>setPrepareBW(e.target.checked)}/> Preparar P&B após melhorar</label><button disabled={!artifact||artifact.candidate||running} onClick={()=>{enhanceRequest.current=true;void execute();}}>✦ Melhorar automaticamente · sem reconstruir</button><small>Reconstrução generativa foi movida para a aba IA.</small></div>}
      {activePanel==='remove-background'&&<div className="field"><label>QUALIDADE DO RECORTE</label><select value={cutoutQuality} onChange={e=>setCutoutQuality(e.target.value as typeof cutoutQuality)}><option value="normal">Normal · BiRefNet 1024</option><option value="detailed">Detalhada · BiRefNet 1536</option><option value="max">Máxima · BiRefNet 2048</option></select><small>Outros motores usam as transformações próprias. Refinamento interno do BEN2 atua só no BEN2; para matting adicional ative ViTMatte.</small></div>}
      {(activePanel==='history'||activePanel==='vectorize')&&versions.length>1&&<div className="field"><label>COMPARAR DUAS VERSÕES / MOTORES</label><select value={compareVersion} onChange={e=>setCompareVersion(e.target.value)}><option value="">Escolher versão para o lado esquerdo</option>{versions.filter(item=>item.id!==artifact?.id).map(item=><option key={item.id} value={item.id}>{item.parentId?`V${item.versionNumber}`:'Original'} · {item.engine??'Entrada'}</option>)}</select><button disabled={!compareVersion||!artifact||running} onClick={async()=>{const reference=versions.find(item=>item.id===compareVersion);if(!reference||!artifact)return;try{const [left,right]=await Promise.all([loadPreviewImage(reference.id,true),loadPreviewImage(artifact.id,true)]);revoke(beforeUrl);revoke(afterUrl);revoke(beforeInspectUrl);revoke(afterInspectUrl);setBeforeUrl(left);setAfterUrl(right);setBeforeInspectUrl('');setAfterInspectUrl('');setOperationSource(reference);setView('compare');setFit(true);}catch(e){setError(String(e));}}}>Comparar com a versão ativa</button></div>}
      {activePanel==='reconstruct'&&<div className="field"><label>TEXTO EXATO DA LOGO · OPCIONAL</label><input type="text" maxLength={300} value={logoText} onChange={e=>setLogoText(e.target.value)} placeholder="Ex.: Firma Autógrafa Digital"/><small>Ajuda a orientar a escrita, mas não garante a mesma fonte. Confira sempre o candidato.</small></div>}
      <div className="panel-spacer"/>
      {activePanel !== 'history' && error && <div className="error">{error}</div>}{activePanel !== 'history' && job?.error && <div className="error">{job.error.message}</div>}
      {activePanel !== 'history' && job && <details className="job-state" open={running || undefined}><summary><span>{phaseLabel[job.phase] ?? job.phase}</span><b>{progress === null ? '•••' : `${progress}%`}</b></summary><div className="tile-progress" aria-label={`Progresso ${progress ?? 0}%`}>{Array.from({ length: squares }, (_, i) => <i key={i} className={i < completedSquares ? 'done' : ''}/>)}</div>{job.metrics && <small>{(job.metrics.durationMs / 1000).toFixed(1)}s · pico {(job.metrics.peakRssBytes / 1073741824).toFixed(1)} GB</small>}</details>}
      {activePanel !== 'history' && <div className="actions">{running ? <button className="danger" disabled={!job} onClick={() => {if(job)void api(`/api/jobs/${job.id}/cancel`, { method: 'POST' }).catch(e=>setError(String(e)));}}>Cancelar processamento</button> : <><button className="primary" disabled={!artifact || artifact.candidate || artifact.mime === 'image/svg+xml' || (operation==='reconstruct'&&!useAI)} onClick={execute}>{operationNames[operation]} versão ativa{operation === 'upscale' || operation === 'remove-background' ? ' na GPU' : ''}</button>{artifact && <button className="download-png" onClick={download}>↓ Baixar versão ativa · {artifact.mime === 'image/svg+xml' ? 'SVG' : exportFormat === 'jpeg' ? 'JPEG' : exportFormat.toUpperCase()}</button>}</>}</div>}
    </section>
    {maskEditor&&artifact&&<MaskEditor {...maskEditor} onClose={()=>{revoke(maskEditor.sourceUrl);revoke(maskEditor.maskUrl);setMaskEditor(null);}} onApply={async strokes=>{const value:JobRecord=await(await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({artifactId:artifact.id,operation:'mask-correct',options:{strokes}})})).json();setOperationSource(artifact);setJob(value);revoke(maskEditor.sourceUrl);revoke(maskEditor.maskUrl);setMaskEditor(null);}}/>}
    {boxUrl&&<BoxSelector url={boxUrl} onClose={()=>{revoke(boxUrl);setBoxUrl('');}} onApply={value=>{setBoxes(value);setBackgroundEngine('finegrain');revoke(boxUrl);setBoxUrl('');}}/>}
    <main className="workspace">
      {running&&<div className="processing-overlay" role="status" aria-live="polite"><div className="processing-card"><div className="processing-spinner"/><strong>{({installing_flux:'Instalando dependências da IA',loading_flux_transformer:'Baixando/carregando FLUX',loading_flux_encoder:'Carregando codificador de texto',loading_flux_pipeline:'Carregando componentes da IA',encoding_flux_reference:'Preparando imagem de referência',reconstructing_logo:'Recriando imagem',decoding_flux_result:'Finalizando e restaurando enquadramento',installing_diffbir:'Preparando DiffBIR',restoring_diffbir:'Restaurando imagem'} as Record<string,string>)[job?.phase??'']??phaseLabel[job?.phase??'']??'Processando imagem'}</strong><span>{progress===null?'Etapa sem porcentagem disponível':`${progress}% desta etapa`}</span><span>Tempo: {Math.max(0,Math.floor((now-(job?.startedAtEpoch??now/1000)*1000)/1000))} s · tarefa {job?.id??'iniciando'}</span><small>{job?.heartbeatEpoch?`Processo ativo · atualização há ${Math.max(0,Math.floor(now/1000-job.heartbeatEpoch))} s`:'Acompanhe downloads e mensagens no log do Colab.'}</small><button disabled={!job} onClick={()=>{if(job)void api(`/api/jobs/${job.id}/cancel`,{method:'POST'});}}>Cancelar esta tarefa</button></div></div>}
      <div className="workspace-toolbar"><div className="view-switch"><button className={view === 'original' ? 'selected' : ''} onClick={() => setView('original')}>{afterUrl ? 'Antes' : 'Versão ativa'}</button><button className={view === 'result' ? 'selected' : ''} disabled={!afterUrl} onClick={() => setView('result')}>Resultado</button><button className={view === 'compare' ? 'selected' : ''} disabled={!afterUrl} onClick={() => setView('compare')}>Comparar</button></div><div className="resolution-badge">{view === 'original' || !afterUrl ? `${beforeArtifact?.width ?? 0} × ${beforeArtifact?.height ?? 0}` : `${artifact?.width ?? 0} × ${artifact?.height ?? 0}`}</div><div className="background-switch"><button className={viewBg === 'checker' ? 'selected' : ''} onClick={() => setViewBg('checker')} title="Transparência">▦</button><button className={viewBg === 'white' ? 'selected' : ''} onClick={() => setViewBg('white')}>Branco</button><button className={viewBg === 'black' ? 'selected' : ''} onClick={() => setViewBg('black')}>Preto</button></div></div>
      <div className="viewer-controls"><button onClick={() => changeZoom(zoom - 25)}>{icons.zoomOut}</button><strong>{fit ? 'Ajustado' : `${Number((zoom / 100).toFixed(2))}×`}</strong><button onClick={() => changeZoom(zoom + 25)}>{icons.zoomIn}</button><button onClick={() => setFit(true)}>Ajustar</button><button onClick={fitWidth}>Ajustar largura</button><span>{panModifier ? 'MÃO ATIVA · arraste para mover' : view === 'compare' ? 'Arraste a linha · Ctrl: mover imagem' : 'Roda: zoom · arraste: mover'}</span></div>
      <div ref={canvasRef} className={`canvas ${viewBg} ${!fit && (view !== 'compare' || panModifier) ? 'pannable' : ''} ${panModifier ? 'pan-modifier' : ''} ${panning ? 'panning' : ''}`} onPointerDown={startPan} onPointerMove={movePan} onPointerUp={stopPan} onPointerCancel={stopPan} onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={e => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files[0]); }}>
        {!beforeUrl ? <button className={`dropzone ${drag ? 'drag' : ''}`} onClick={() => fileInput.current?.click()}><span>{icons.upload}</span><strong>Escolha ou arraste uma imagem</strong><small>PNG, JPEG ou WebP · transparência preservada</small></button> : <div ref={stageRef} className="image-stage" style={{ width: stageWidth, flexShrink: 0 }}><img draggable={false} src={afterUrl && view === 'result' ? afterDisplayUrl : beforeDisplayUrl} style={{ clipPath: afterUrl && view === 'compare' ? `inset(0 ${100 - compare}% 0 0)` : undefined }} alt={view === 'original' ? 'Versão usada como entrada' : 'Resultado processado'}/>{afterUrl && view === 'compare' && <div className="after-layer" style={{ clipPath: `inset(0 0 0 ${compare}%)` }}><img draggable={false} src={afterDisplayUrl} alt="Resultado processado"/></div>}{afterUrl && view === 'compare' && <><input className="compare-slider" aria-label="Comparação antes e depois" type="range" min="0" max="100" value={compare} onChange={e => setCompare(+e.target.value)}/><div className="divider" role="separator" aria-label="Mover divisor da comparação" aria-valuenow={Math.round(compare)} style={{ left: `${compare}%` }} onPointerDown={startCompare} onPointerMove={moveCompare} onPointerUp={stopCompare} onPointerCancel={stopCompare}><span className="compare-handle">‹ ›</span></div><span className="before-label">ENTRADA</span><span className="after-label">{operationNames[operation].toUpperCase()}</span></>}</div>}
      </div>
      <footer><span>{inspectionReady ? 'Comparação HD ativa · original e IA em resoluções diferentes reais.' : fit ? 'Preview leve ajustado à tela. O download mantém a resolução completa.' : 'Carregando detalhe para inspeção; o arquivo baixado permanece completo.'}</span><span>{running ? 'GPU processando' : `${gpuLabel}${buildVersion ? ` · v${buildVersion}` : ''} · 1 trabalho por vez`}</span></footer>
    </main>
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
