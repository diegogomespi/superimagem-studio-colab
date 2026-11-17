import React, {useRef,useState} from 'react';
export function BoxSelector({url,onApply,onClose,purpose='object'}:{purpose?:'object'|'inspection';url:string;onApply:(boxes:number[][])=>void;onClose:()=>void}) {
  const [boxes,setBoxes]=useState<number[][]>([]);
  const [draft,setDraft]=useState<number[]|null>(null);
  const start=useRef<number[]|null>(null);
  const point=(event:React.PointerEvent<HTMLDivElement>)=>{
    const rect=event.currentTarget.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];
  };
  const region=(end:number[])=>start.current?[Math.min(end[0],start.current[0]),Math.min(end[1],start.current[1]),Math.max(end[0],start.current[0]),Math.max(end[1],start.current[1])]:null;
  return <div className="box-modal" role="dialog" aria-modal="true" aria-label="Selecionar objeto para recorte">
    <header><div><h2>{purpose==='inspection'?'Área de comparação':'Selecionar objeto'}</h2><p>{purpose==='inspection'?'Arraste uma caixa ao redor do cabelo ou detalhe que deseja comparar. A primeira região será mostrada em todos os quadros da montagem; a imagem inteira será processada.':'Clique e arraste do canto superior ao inferior do objeto. Inclua cabelo, mãos e toda a logo. A caixa orienta o Finegrain; ela não é o contorno final.'}</p></div><button onClick={onClose}>✕ Fechar</button></header>
    <div className="box-workarea"><div className="box-image" onPointerDown={event=>{if(event.button!==0||boxes.length>=20)return;start.current=point(event);event.currentTarget.setPointerCapture(event.pointerId);}}
      onPointerMove={event=>{if(start.current)setDraft(region(point(event)));}}
      onPointerUp={event=>{const box=region(point(event));if(box&&box[2]-box[0]>.005&&box[3]-box[1]>.005)setBoxes(items=>[...items,box].slice(0,20));start.current=null;setDraft(null);}}
      onPointerCancel={()=>{start.current=null;setDraft(null);}}>
      <img src={url} alt="Arraste para indicar o objeto que deseja manter" draggable={false}/>
      {[...boxes,...(draft?[draft]:[])].map((box,index)=><div key={index} className="box-region" style={{left:`${box[0]*100}%`,top:`${box[1]*100}%`,width:`${(box[2]-box[0])*100}%`,height:`${(box[3]-box[1])*100}%`}}><span>{index+1}</span></div>)}
    </div></div>
    <footer><span>{boxes.length?`${boxes.length} região(ões) selecionada(s)`:'Nenhuma seleção: arraste sobre a imagem para começar.'}</span><button disabled={!boxes.length} onClick={()=>setBoxes(items=>items.slice(0,-1))}>↶ Desfazer caixa</button><button disabled={!boxes.length} onClick={()=>setBoxes([])}>Limpar</button><button className="box-apply" disabled={!boxes.length} onClick={()=>onApply(boxes)}>Usar seleção</button></footer>
  </div>;
}
