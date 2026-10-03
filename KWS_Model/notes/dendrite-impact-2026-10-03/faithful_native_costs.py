"""Offline native architecture cost audit: no checkpoints, no licensed PAI imports.

Constructs each unchanged recorded model configuration, runs one dummy batch,
counts Conv1d/Conv2d/Linear through hooks. DTNet's forward bypasses its module
hooks; its number is an ideal causal recurrence MAC estimate, not MPS compute.
All MACs exclude normalization, activations, pooling and bias additions.
"""
from pathlib import Path
import csv,json,sys
import yaml,torch
ROOT=Path(__file__).resolve().parents[3]
KWS=ROOT/'KWS_Model';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(KWS/'src'))
from kws.models.registry import build_model
rows=list(csv.DictReader((OUT/'faithful-native-students-inventory.csv').open()))
grouped={r['model_config']:r for r in rows}
results=[]
for path,r in sorted(grouped.items()):
 cfg=yaml.safe_load((KWS/path).read_text())
 torch.manual_seed(0)
 m=build_model(cfg,(32,101),12).cpu().eval()
 params=sum(p.numel() for p in m.parameters())
 if params!=int(r['params']):raise ValueError((path,params,r['params']))
 if cfg.get('family')=='dtnet':
  macs=0
  if cfg.get('input_transform','none')=='idct':macs+=32*32*101
  for layer in m.layers:
   n,k,f=layer.neurons,layer.branches,layer.fan_in
   macs+=101*n*k*(f+1) # branch affine and soma mixing
   if layer.branch_rho is not None:macs+=101*n*k*2 # a*v+(1-a)*u
   if layer.soma_rho is not None:macs+=101*n*2
  macs+=m.fc.in_features*m.fc.out_features
  method='ideal_streaming_recurrence_plus_IDCT; not actual dense training matmul cost'
 else:
  total=[0];hs=[]
  def hook(module,inputs,out):
   kernel=module.in_features if isinstance(module,torch.nn.Linear) else (module.in_channels//module.groups)*__import__('math').prod(module.kernel_size)
   total[0]+=out.numel()*kernel
  for module in m.modules():
   if isinstance(module,(torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Linear)):hs.append(module.register_forward_hook(hook))
  with torch.no_grad():m(torch.zeros(1,1,32,101))
  for h in hs:h.remove()
  macs=total[0];method='dummy_forward_Conv1d_Conv2d_Linear_hooks'
 results.append(dict(model_config=path,params=params,macs=macs,method=method))
with (OUT/'faithful-native-model-costs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
print('Profiled',len(results),'unchanged native configurations; parameter counts all agree with logged epochs.')
