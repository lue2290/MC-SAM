"""Count the current ViT-H model on the meta device, without checkpoint downloads."""
import ast, json, hashlib
from pathlib import Path
from collections import defaultdict
import torch
import segment_anything.build_sam as unused
import importlib
builder=importlib.import_module('segment_anything.build_sam')
from segment_anything import sam_model_registry
from segment_anything.modeling.mcsam_integrated import create_integrated_model, create_optimizer_for_integrated_model
root=Path(__file__).resolve().parent
# Reuse the exact SAM construction, omitting only the subsequent checkpoint I/O.
tree=ast.parse((root/'segment_anything/build_sam.py').read_text(encoding='utf-8-sig'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_build_sam')
cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='checkpoint' for x in n.targets))
fn.body=fn.body[:cut]+[ast.Return(value=ast.Name(id='sam',ctx=ast.Load()))]
ns=dict(vars(builder));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'sam_without_checkpoint','exec'),ns)
sam_model_registry['vit_h']=lambda checkpoint=None:ns['_build_sam'](1280,32,16,[7,15,23,31],None)
with torch.device('meta'):
 model=create_integrated_model(None,model_type='vit_h',device='meta')
rows=defaultdict(lambda:{'total':0,'trainable':0,'frozen':0,'tensors':0})
def group(name):
 if name.startswith('image_encoder.shared_manifold_adapter.'):return 'MCA'
 if name.startswith('image_encoder.blip_feature_adjust.'):return 'BLIP feature alignment'
 if name.startswith('image_encoder.'):return 'SAM image encoder (original)'
 return name.split('.')[0]
for name,p in model.named_parameters():
 r=rows[group(name)];r['total']+=p.numel();r['trainable' if p.requires_grad else 'frozen']+=p.numel();r['tensors']+=1
opt=create_optimizer_for_integrated_model(model,lr=5e-5)
optimizer_ids={id(p) for g in opt.param_groups for p in g['params']}
assert optimizer_ids=={id(p) for p in model.parameters() if p.requires_grad}
report={'configuration':{'backbone':'vit_h','image_size':1024,'n_streams':4,'cspg_temperature':1.0,'cspg_iters':5},'scope':'Registered integrated model parameters; excludes separately loaded BLIP and Mamba; no checkpoint loaded, no training performed.','source_sha256':hashlib.sha256((root/'segment_anything/modeling/mcsam_integrated.py').read_bytes()).hexdigest(),'modules':dict(rows)}
for key in ['total','trainable','frozen']:report[key]=sum(r[key] for r in rows.values())
report['trainable_percent']=100*report['trainable']/report['total']
report['optimizer_groups']=[{'name':g['name'],'scalars':sum(p.numel() for p in g['params']),'tensors':len(g['params']),'lr':g['lr']} for g in opt.param_groups]
report['rankdice_parameters']=sum(p.numel() for p in model.rankdice_module.parameters())
(root/'parameter_counts_current.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
