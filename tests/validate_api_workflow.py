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
from colab.app import app


def main():
    with TestClient(app) as client:
        image=np.full((64,96,4),255,np.uint8)
        cv2.putText(image,'A',(20,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0,255),2)
        _,encoded=cv2.imencode('.png',image)
        response=client.post('/api/artifacts',files={'file':('logo.png',io.BytesIO(encoded.tobytes()),'image/png')})
        assert response.status_code==200,response.text
        original=response.json()
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
        # Exclusão em lote valida dependências antes de apagar qualquer item.
        response=client.post('/api/artifacts/delete-batch',json={'ids':[second['id']]})
        assert response.status_code==409,response.text
        assert client.get('/api/artifacts/'+second['id']).status_code==200
        response=client.post('/api/artifacts/delete-batch',json={'ids':[second['id'],third['id']]})
        assert response.status_code==200,response.text
        assert client.get('/api/artifacts/'+original['id']).status_code==200
        assert client.post('/api/artifacts/delete-batch',json={'ids':[original['id']]}).status_code==400
        print('OK: API HTTP real, upload, P&B, pincel, métricas, dependências e exclusão atômica de ramo. CUDA não testada.')


if __name__=='__main__':main()
