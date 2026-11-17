import React, { useEffect, useRef, useState } from 'react';

export type MaskStroke = { mode: 'keep' | 'remove'; size: number; hardness: number; opacity: number; points: number[][] };
export function MaskEditor({ sourceUrl, maskUrl, onApply, onClose }: { sourceUrl: string; maskUrl: string; onApply: (strokes: MaskStroke[]) => Promise<void>; onClose: () => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const original = useRef<HTMLImageElement | null>(null);
  const current = useRef<HTMLImageElement | null>(null);
  const active = useRef<MaskStroke | null>(null);
  const [strokes, setStrokes] = useState<MaskStroke[]>([]);
  const [redo, setRedo] = useState<MaskStroke[]>([]);
  const [mode, setMode] = useState<'keep' | 'remove'>('keep');
  const [size, setSize] = useState(.04);
  const [opacity, setOpacity] = useState(1);
  const [hardness, setHardness] = useState(.7);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const redraw = () => {
    const element = canvas.current, base = current.current, input = original.current;
    if (!element || !base || !input) return;
    const ctx = element.getContext('2d')!;
    ctx.clearRect(0, 0, element.width, element.height);
    ctx.drawImage(base, 0, 0, element.width, element.height);
    for (const stroke of [...strokes, ...(active.current ? [active.current] : [])]) {
      ctx.save(); ctx.globalAlpha = stroke.opacity;
      for (let i = 0; i < stroke.points.length; i++) {
        const previous = stroke.points[Math.max(0, i-1)], point = stroke.points[i];
        const distance = Math.hypot((point[0]-previous[0])*element.width, (point[1]-previous[1])*element.height);
        const radius = stroke.size * Math.max(element.width, element.height) / 2;
        const count = Math.max(1, Math.ceil(distance / Math.max(1, radius/3)));
        for (let j = 0; j <= count; j++) {
          const x = (previous[0]+(point[0]-previous[0])*j/count)*element.width;
          const y = (previous[1]+(point[1]-previous[1])*j/count)*element.height;
          ctx.save(); ctx.beginPath(); ctx.arc(x,y,radius,0,Math.PI*2); ctx.clip();
          if (stroke.mode === 'keep') ctx.drawImage(input,0,0,element.width,element.height);
          else { ctx.globalCompositeOperation='destination-out'; ctx.fillRect(x-radius,y-radius,radius*2,radius*2); }
          ctx.restore();
        }
      }
      ctx.restore();
    }
  };
  useEffect(() => {
    let cancelled = false;
    const load = (url: string) => new Promise<HTMLImageElement>((resolve,reject) => { const image = new Image(); image.onload=()=>resolve(image); image.onerror=()=>reject(new Error('Falha ao carregar prévia da máscara')); image.src=url; });
    void Promise.all([load(sourceUrl),load(maskUrl)]).then(([a,b]) => { if(cancelled) return; original.current=a; current.current=b; if(canvas.current) { canvas.current.width=b.naturalWidth; canvas.current.height=b.naturalHeight; } redraw(); }).catch(e=>setError(String(e)));
    return () => { cancelled=true; };
  }, [sourceUrl,maskUrl]);
  useEffect(redraw,[strokes]);
  const point = (event: React.PointerEvent<HTMLCanvasElement>) => { const box=event.currentTarget.getBoundingClientRect(); return [Math.max(0,Math.min(1,(event.clientX-box.left)/box.width)),Math.max(0,Math.min(1,(event.clientY-box.top)/box.height))]; };
  const end = () => { const stroke=active.current; if(stroke) { active.current=null; setStrokes(items=>[...items,stroke]); setRedo([]); } };
  return <div style={{position:'fixed',inset:0,zIndex:1000,background:'#090b10',padding:20,overflow:'auto'}}>
    <h2>Corrigir máscara · Manter / Remover</h2>
    <div style={{display:'flex',gap:8,flexWrap:'wrap',alignItems:'center'}}>
      <button onClick={()=>setMode('keep')} aria-pressed={mode==='keep'}>Manter</button><button onClick={()=>setMode('remove')} aria-pressed={mode==='remove'}>Remover</button>
      <label>Tamanho<input type="range" min=".005" max=".2" step=".005" value={size} onChange={e=>setSize(+e.target.value)}/></label>
      <label>Dureza<input type="range" min="0" max="1" step=".05" value={hardness} onChange={e=>setHardness(+e.target.value)}/></label>
      <label>Opacidade<input type="range" min=".05" max="1" step=".05" value={opacity} onChange={e=>setOpacity(+e.target.value)}/></label>
      <button disabled={!strokes.length} onClick={()=>{setRedo(items=>[...items,strokes[strokes.length-1]]);setStrokes(items=>items.slice(0,-1));}}>↶ Desfazer</button>
      <button disabled={!redo.length} onClick={()=>{setStrokes(items=>[...items,redo[redo.length-1]]);setRedo(items=>items.slice(0,-1));}}>↷ Refazer</button>
      <button onClick={()=>{setStrokes([]);setRedo([]);}}>Restaurar máscara</button>
      <button disabled={busy||!strokes.length} onClick={async()=>{setBusy(true);try{await onApply(strokes);}catch(e){setError(String(e));}finally{setBusy(false);}}}>Aplicar correção</button><button disabled={busy} onClick={onClose}>Cancelar</button>
    </div>
    <p>Prévia aproximada do pincel. O resultado final calcula dureza e opacidade na resolução completa. Use a máscara aplicada para conferir bordas.</p>{error&&<p>{error}</p>}
    <canvas ref={canvas} style={{maxWidth:'100%',maxHeight:'75vh',touchAction:'none',cursor:'crosshair',background:'repeating-conic-gradient(#20212a 0% 25%,#13141b 0% 50%) 0/20px 20px'}} onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);active.current={mode,size,hardness,opacity,points:[point(e)]};redraw();}} onPointerMove={e=>{if(active.current){active.current.points.push(point(e));redraw();}}} onPointerUp={end} onPointerCancel={end}/>
  </div>;
}
