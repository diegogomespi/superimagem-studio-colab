"""Contratos de perfis/trimap/fusão. Não substitui teste de qualidade na T4."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from colab.matting_tools import make_trimap,guided_alpha
from colab.cutout_benchmark import PRESETS,validate_presets

mask=np.zeros((160,160),np.uint8);mask[40:120,40:120]=255
narrow=make_trimap(mask,2);wide=make_trimap(mask,10)
assert set(np.unique(wide))=={0,128,255}
assert np.count_nonzero(wide==128)>np.count_nonzero(narrow==128)
assert wide[80,80]==255 and wide[0,0]==0
detail=np.full_like(mask,190)
combined=guided_alpha(mask,detail,4)
assert combined[80,80]==255 and combined[0,0]==0 and combined[40,80]==190
keys=[key for key in PRESETS if key.startswith('research-')]
assert len(keys)==8
validate_presets(keys)
assert PRESETS['research-fey'][2]=={}
assert PRESETS['research-fey-color'][2]=={'foregroundCorrection':True}
assert PRESETS['research-whole'][2]['mattingMode']=='whole'
assert PRESETS['research-wide'][2]['trimapRatio']>PRESETS['research-narrow'][2]['trimapRatio']
print('OK: 8 perfis distintos, trimaps com 3 estados, banda variável e fusão local preservando núcleo/fundo.')
