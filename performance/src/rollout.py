#!/usr/bin/env python3
"""GT(340.0 s)からteacher forcingなしで100-step rolloutを評価する。"""
from __future__ import annotations
import argparse,csv,math,sys
from pathlib import Path
from typing import Any
import numpy as np, torch, yaml
ROOT=Path(__file__).resolve().parents[2]
for d in (ROOT/'src',Path(__file__).resolve().parent):
    if str(d) not in sys.path: sys.path.insert(0,str(d))
from dataset import PairDataset
from model import build_model
from test import load_checkpoint,resolve_path
from test_dataloader import create_test_dataloader,load_config as load_loader_config
from train import select_device

def load_config(path:Path)->dict[str,Any]:
    c=yaml.safe_load(path.read_text())
    if not isinstance(c,dict): raise ValueError('Rollout config must be a mapping')
    for k in ('test_dataloader_config','model_config','checkpoint','device'):
        if not isinstance(c.get(k),str): raise ValueError(f'{k} must be a string')
    for k in ('start_time','end_time','steps','tolerance','reynolds'):
        if k not in c.get('rollout',{}): raise ValueError(f'Missing rollout.{k}')
    return c

def time_pairs(ds,setting):
    d=torch.load(ds.files[0],map_location='cpu',weights_only=False); t=d['times']; n=int(setting['steps']); tol=float(setting['tolerance'])
    start,end=float(setting['start_time']),float(setting['end_time'])
    if tuple(t.shape)!=(n,2) or len(ds)!=n: raise ValueError(f'Expected {n} time pairs: {ds.files[0]}')
    expected=torch.linspace(start+(end-start)/n,end,n,dtype=t.dtype)
    if abs(float(t[0,0])-start)>tol or not torch.allclose(t[:,1],expected,atol=tol,rtol=0) or not torch.allclose(t[:-1,1],t[1:,0],atol=tol,rtol=0):
        raise ValueError('Time alignment/off-by-one failure: pred[0]=340.1 and pred[99]=350.0 required')
    return t

def calc(pred,gt,mask,names):
    out={}
    for idx,name in [(None,'overall'),*enumerate(names)]:
        p=pred if idx is None else pred[idx:idx+1]; g=gt if idx is None else gt[idx:idx+1]; use=mask.unsqueeze(0).expand_as(p); p,g=p[use],g[use]
        e=float((p-g).square().sum()); energy=float(g.square().sum())
        out[f'mse_{name}']=e/g.numel(); out[f'relative_l2_{name}']=math.sqrt(e/energy)
    return out

def wandb_run(cfg,reynolds):
    w=cfg.get('wandb',{})
    if not w.get('enabled',False): return None
    import wandb
    return wandb.init(project=w['project'],entity=w.get('entity'),name=f"{w.get('name_prefix','rollout')}-Re{reynolds:g}",group=w.get('group'),job_type='rollout',mode=w.get('mode','online'),tags=['fno','rollout',f'Re{reynolds:g}'],config=cfg)

def visualize(saved,out,count):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    pred,gt=saved['prediction'],saved['ground_truth']; err=saved['metrics']['relative_l2_overall']; worst=int(torch.tensor(err).argmax())
    ids=set(np.linspace(0,len(pred)-1,count,dtype=int).tolist()); ids.add(worst)
    for i in sorted(ids):
        tag='max_error' if i==worst else f'step{i+1:03d}'
        for ch,name in enumerate(saved['metadata']['output_channels']):
            fields=(gt[i,ch].numpy(),pred[i,ch].numpy(),np.abs(pred[i,ch].numpy()-gt[i,ch].numpy()))
            fig,ax=plt.subplots(1,3,figsize=(14,4))
            for a,f,title in zip(ax,fields,('Ground Truth','Prediction','Absolute Error')):
                im=a.imshow(f,origin='lower',aspect='auto'); a.set_title(title); a.set_axis_off(); fig.colorbar(im,ax=a)
            fig.suptitle(f"Re={saved['Re']:g}, t={saved['time'][i]:.1f}, {name}"); fig.tight_layout(); fig.savefig(out/f'{tag}_{name}.png',dpi=180); plt.close(fig)


def export_prediction_vti(saved, output_dir):
    """PredictionだけをRe別VTI時系列とPVD collectionへ保存する。"""
    import pyvista as pv
    prediction=saved['prediction']; ground_truth=saved['ground_truth']; meta=saved['metadata']; bounds=meta['bounds']; ny,nx=prediction.shape[-2:]
    spacing=((bounds[1]-bounds[0])/(nx-1),(bounds[3]-bounds[2])/(ny-1),1.0)
    origin=(bounds[0],bounds[2],float(meta.get('z',0.0)))
    output_dir.mkdir(parents=True,exist_ok=True); entries=[]
    for step,time_value in enumerate(saved['time']):
        grid=pv.ImageData(dimensions=(nx,ny,1),spacing=spacing,origin=origin)
        for channel,name in enumerate(meta['output_channels']):
            pred=prediction[step,channel].numpy(); truth=ground_truth[step,channel].numpy()
            grid.point_data[f'{name}_prediction']=pred.ravel(order='C')
            grid.point_data[f'{name}_ground_truth']=truth.ravel(order='C')
            grid.point_data[f'{name}_absolute_error']=np.abs(pred-truth).ravel(order='C')
        grid.field_data['Re']=np.array([saved['Re']],dtype=np.float64)
        grid.field_data['time']=np.array([float(time_value)],dtype=np.float64)
        filename=f'prediction_{step+1:04d}.vti'; grid.save(output_dir/filename)
        entries.append((float(time_value),filename))
    lines=['<?xml version="1.0"?>','<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">','  <Collection>']
    lines += [f'    <DataSet timestep="{t:.12g}" group="" part="0" file="{name}"/>' for t,name in entries]
    lines += ['  </Collection>','</VTKFile>']
    (output_dir/f"prediction_Re{saved['Re']:g}.pvd").write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run_case(ds,reynolds,model,device,cfg,out):
    setting=cfg['rollout']; times=time_pairs(ds,setting); state=ds[0][0].unsqueeze(0).to(device); mask_i=ds.input_channels.index('mask'); input_i=[ds.input_channels.index(n) for n in ds.output_channels]
    preds,gts,rows=[],[],[]; run=wandb_run(cfg,reynolds)
    try:
        with torch.no_grad():
            for step in range(int(setting['steps'])):
                target=ds[step][1].to(device); pred=model(state)[0]
                if not torch.isfinite(pred).all(): raise ValueError(f'NaN/Inf at Re={reynolds:g}, step={step+1}')
                mask=state[0,mask_i].bool(); pred=pred.masked_fill(~mask.unsqueeze(0),0); pp=ds.inverse_transform_output(pred,mask); pg=ds.inverse_transform_output(target,mask)
                row={'Re':reynolds,'step':step+1,'time':float(times[step,1]),'elapsed_time':float(times[step,1]-times[0,0]),**calc(pp,pg,mask,ds.output_channels)}; rows.append(row)
                if run: run.log(row,step=step+1)
                preds.append(pp.cpu()); gts.append(pg.cpu()); state=state.clone(); state[0,input_i]=pred
    finally:
        if run: run.finish()
    raw=torch.load(ds.files[0],map_location='cpu',weights_only=False); source_meta=raw['meta']
    saved={'prediction':torch.stack(preds),'ground_truth':torch.stack(gts),'time':times[:,1],'Re':reynolds,'mask':state[0,mask_i].cpu(),'metrics':{k:[r[k] for r in rows] for k in rows[0] if k.startswith(('mse_','relative_l2_'))},'metadata':{'initial_time':float(times[0,0]),'teacher_forcing':False,'scale':'physical','output_channels':list(ds.output_channels),'bounds':source_meta['bounds'],'z':source_meta.get('z',0.0),'nx':source_meta['nx'],'ny':source_meta['ny']}}
    torch.save(saved,out/f'rollout_Re{reynolds:g}.pt')
    if cfg['output'].get('save_prediction_only_pt',True):
        torch.save({'prediction':saved['prediction'],'time':saved['time'],'Re':reynolds,'metadata':saved['metadata']},out/f'rollout_prediction_Re{reynolds:g}.pt')
    if cfg['output'].get('export_vti',True): export_prediction_vti(saved,out/'vti'/f'Re{reynolds:g}')
    plots=out/'plots'/f'Re{reynolds:g}'; plots.mkdir(parents=True,exist_ok=True); visualize(saved,plots,int(cfg['output']['representative_count'])); return rows

def rollout(config_path):
    path=Path(config_path).resolve(); cfg=load_config(path); lp=resolve_path(cfg['test_dataloader_config'],path); lc=load_loader_config(lp); full=create_test_dataloader(lp).dataset; assert isinstance(full,PairDataset)
    requested=[float(x) for x in cfg['rollout']['reynolds']]; missing=set(requested)-set(full.reynolds_by_file)
    if missing: raise ValueError(f'Re not in test loader: {sorted(missing)}')
    out=resolve_path(cfg['output']['root_directory'],path)/lc['evaluation']['type']/'rollout'; out.mkdir(parents=True,exist_ok=True); device=select_device(cfg['device']); model=build_model(resolve_path(cfg['model_config'],path)).to(device); load_checkpoint(resolve_path(cfg['checkpoint'],path),model,device); model.eval()
    rows=[]
    for re in requested: rows+=run_case(full.select_reynolds([re]),re,model,device,cfg,out)
    with (out/cfg['output']['metrics_csv']).open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f'rollout completed: Re={requested}, results={out}')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,default=Path(__file__).parents[1]/'configs'/'rollout.yaml'); rollout(p.parse_args().config)
if __name__=='__main__': main()
