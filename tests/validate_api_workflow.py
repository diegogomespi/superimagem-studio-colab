"""API real com operações CPU; módulos CUDA substituídos apenas para importar sem torch."""
import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.modules['torch']=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
sys.modules['spandrel']=SimpleNamespace(ModelLoader=object)
from colab.app import app, jobs, job_futures, artifacts
from concurrent.futures import Future


def main():
    with TestClient(app) as client:
        assert client.get('/api/health').json()['engineAvailable'] is False
        cuda=sys.modules['torch'].cuda
        cuda.is_available=lambda:True
        cuda.get_device_properties=lambda _:SimpleNamespace(name='T4 teste',total_memory=16*1024**3)
        cuda.mem_get_info=lambda:(10*1024**3,16*1024**3)
        sys.modules['torch'].version=SimpleNamespace(cuda='teste')
        assert client.get('/api/health').json()['engineAvailable'] is True
        def lost_gpu():raise RuntimeError('GPU caiu no teste')
        cuda.mem_get_info=lost_gpu
        assert client.get('/api/health').json()['engineAvailable'] is False
        cuda.is_available=lambda:False
        image=np.full((64,96,4),255,np.uint8)
        cv2.putText(image,'A',(20,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0,255),2)
        _,encoded=cv2.imencode('.png',image)
        response=client.post('/api/artifacts',files={'file':('logo.png',io.BytesIO(encoded.tobytes()),'image/png')})
        assert response.status_code==200,response.text
        original=response.json()
        missing_box=client.post('/api/jobs',json={'artifactId':original['id'],'operation':'remove-background','options':{'backgroundEngine':'finegrain','boxes':[]}})
        assert missing_box.status_code==400 and 'arraste' in missing_box.text
        cuda.is_available=lambda:True
        oversized={**original,'id':'oversized-test','width':10048,'height':10064}
        artifacts[oversized['id']]=oversized
        count=len(jobs)
        blocked=client.post('/api/jobs',json={'artifactId':oversized['id'],'operation':'upscale','options':{'scale':4}})
        assert blocked.status_code==400 and len(jobs)==count,blocked.text
        del artifacts[oversized['id']]
        cuda.is_available=lambda:False
        def process(source,operation,options):
            response=client.post('/api/jobs',json={'artifactId':source,'operation':operation,'options':options})
            assert response.status_code==202,response.text
            job_id=response.json()['id']
            for _ in range(100):
                job=client.get('/api/jobs/'+job_id).json()
                if job['status'] in ('done','error'):break
                time.sleep(.03)
            assert job['status']=='done',job
            return client.get('/api/artifacts/'+job['resultArtifactId']).json()
        second=process(original['id'],'black-white',{'thresholdMode':'otsu'})
        third=process(second['id'],'mask-correct',{'strokes':[{'mode':'remove','size':.1,'points':[[.5,.5]]}]})
        assert 'fidelity' in third
        cleaned=process(original['id'],'clean',{'cleanStyle':'illustration','cleanStrength':100,'cleanDenoise':40,'cleanContrast':30,'cleanSharpness':25,'cleanEdges':50})
        assert cleaned['engine']=='OpenCV-clean-illustration-v2',cleaned
        assert (cleaned['width'],cleaned['height'])==(original['width'],original['height'])
        # Caminho HTTP da montagem/pacote; inferência substituída explicitamente.
        import zipfile
        import colab.cutout_benchmark as benchmark
        real_benchmark=benchmark.run_benchmark
        def fake_benchmark(source,destination,options,job,root):
            cv2.imwrite(str(destination),np.full((64,128,3),110,np.uint8))
            with zipfile.ZipFile(destination.with_suffix('.zip'),'w') as archive:archive.writestr('report.json','{}')
            return {'tests':[{'id':'fey','status':'ok'}]}
        benchmark.run_benchmark=fake_benchmark
        cuda.is_available=lambda:True
        cuda.empty_cache=lambda:None
        try:
            montage=process(original['id'],'cutout-test',{'testPresets':['fey']})
            assert montage['operation']=='cutout-test' and montage['width']==128
            response=client.get('/api/artifacts/'+montage['id']+'/download?format=zip')
            assert response.status_code==200 and response.headers['content-type']=='application/zip'
        finally:
            cuda.is_available=lambda:False
            benchmark.run_benchmark=real_benchmark
        # Exclusão em lote valida dependências antes de apagar qualquer item.
        response=client.post('/api/artifacts/delete-batch',json={'ids':[second['id']]})
        assert response.status_code==409,response.text
        assert client.get('/api/artifacts/'+second['id']).status_code==200
        response=client.post('/api/artifacts/delete-batch',json={'ids':[second['id'],third['id']]})
        assert response.status_code==200,response.text
        assert client.get('/api/artifacts/'+original['id']).status_code==200
        assert client.post('/api/artifacts/delete-batch',json={'ids':[original['id']]}).status_code==400
        jobs['ghost']=dict(id='ghost',artifactId=original['id'],status='processing',phase='inference')
        assert client.get('/api/jobs/active').json()['job'] is None
        assert jobs['ghost']['status']=='interrupted'
        jobs['live']=dict(id='live',artifactId=original['id'],status='processing',phase='reconstructing_logo')
        live=Future();live.set_running_or_notify_cancel();job_futures['live']=live
        assert client.get('/api/jobs/active').json()['job']['id']=='live'
        blocked=client.post('/api/jobs',json={'artifactId':original['id'],'operation':'black-white'})
        assert blocked.status_code==409 and blocked.json()['activeJob']['id']=='live'
        jobs['live']['status']='done';live.set_result(None)
        print('OK: API HTTP real, upload, P&B, pincel, métricas, dependências e exclusão atômica de ramo. CUDA não testada.')


if __name__=='__main__':main()
