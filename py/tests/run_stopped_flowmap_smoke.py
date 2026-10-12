"""Bounded full-size V100 smoke using cached TS data and retained checkpoints.

This checks optimizer/EMA updates, both heads, inference, and checkpoint resume.
It does not measure convergence or establish molecular sampling quality.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from configs.ace_a_nme import get_config
from launchers import learn
from common import loss_args, state_utils, updates


def sha256(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def run(args):
    torch.set_num_threads(2)
    device=torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('This smoke requires a local CUDA GPU')
    if not 1<=args.steps<=1000:
        raise ValueError('Bounded smoke requires 1..1000 steps')
    results=[]
    for mode in ('pretrain_flowmap','dlogpdt','hutchinson'):
        if args.family == 'esd_cond_div_only' and mode == 'pretrain_flowmap':
            continue
        torch.manual_seed(2026)
        cfg=get_config(0,output_folder=str(Path(args.output).parent))
        cfg.problem.com_augmentation=args.com_augmentation
        cfg.network.com_aware=args.com_augmentation
        cfg.optimization.bs=args.batch_size
        cfg.optimization.total_steps=args.steps
        cfg.optimization.diag_fraction=.5
        cfg.network.divergence_mode=mode
        cfg.network.from_scratch=False
        cfg.network.predict_divergence=mode!='pretrain_flowmap'
        cfg.network.vectorized_divergence=mode=='dlogpdt'
        cfg.network.divergence_rescale=1. if mode=='dlogpdt' else float(cfg.problem.d)
        cfg.network.div_loss_weighting_type=args.div_loss_weighting
        cfg.network.load_path=args.velocity_checkpoint if mode=='pretrain_flowmap' else args.flow_checkpoint
        cfg.teacher.load_path='' if args.family=='esd_cond' else args.velocity_checkpoint
        cfg.training.loss_type=args.family
        with cfg.training.ignore_type():
            cfg.training.psd_type='uniform' if args.family=='psd' else None
        cfg.training.time_distr='uniform'
        cfg.logging.visual_freq=0
        cfg.logging.save_freq=0
        cfg.logging.output_name=f'contract_smoke_{args.family}_{mode}'
        cfg,statics,state=learn.setup_state(cfg,device,seed=2026)
        net=state.net
        initial={k:v.detach().clone() for k,v in net.named_parameters()}
        step_fn=updates.setup_train_step(cfg)
        ema_update=updates.setup_ema_update(cfg)
        gen_cpu=torch.Generator().manual_seed(41)
        gen_cuda=torch.Generator(device=device).manual_seed(43)
        # Hold out a deterministic training batch for before/after diagnostics.
        fixed=loss_args.get_loss_fn_args(cfg,statics,gen_cpu,device)
        def teachers():
            return (statics.teacher_params,statics.teacher_params) if statics.teacher_params is not None else (None,state.self_teacher())
        def diagnostic():
            td,to=teachers()
            loss,metrics=statics.loss(td,to,*fixed, density_ema_params=state.ema_teacher(),
                                      generator=torch.Generator(device=device).manual_seed(29))
            return {'loss':float(loss.detach()),**{k:float(v) for k,v in metrics.items()}}
        before=diagnostic()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start=time.perf_counter()
        peak_grad=0.
        for i in range(args.steps):
            batch=loss_args.get_loss_fn_args(cfg,statics,gen_cpu,device)
            td,to=teachers()
            value,metrics=step_fn(state,statics.loss,td,to,batch,gen_cuda)
            if not state.last_update_succeeded or not torch.isfinite(value):
                raise AssertionError(f'{args.family}/{mode} rejected optimizer step {i}: {metrics}')
            ema_update(state)
            peak_grad=max(peak_grad,metrics['train/grad_norm'])
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        assert state.step==args.steps
        after=diagnostic()
        changes={k:float((v-initial[k]).abs().max()) for k,v in net.named_parameters()}
        assert max(changes.values())>0
        if cfg.network.predict_divergence:
            assert changes['dit.div_final_linear.weight']>0, 'density head did not learn'
        base=statics.sample_rho0(16,gen_cpu,device)
        with torch.no_grad():
            samples,logp,nfe=net.sample_forward(base,n_steps=2)
            assert torch.isfinite(samples).all()
            logp_info=None
            if logp is not None:
                inverse=net.sample_inverse(samples,n_steps=2)
                assert torch.isfinite(logp).all() and all(torch.isfinite(v).all() for v in inverse[:3])
                logp_info={'min':float(logp.min()),'max':float(logp.max())}
        # Save separate smoke output, reload params+EMA+optimizer, and continue.
        checkpoint=Path(args.checkpoint_dir)/f'{args.family}-{mode}.pt'
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        state_utils.save_state(state,str(checkpoint))
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        state.load_state_dict(saved,load_optimizer=True)
        for fac,params in saved['ema_params'].items():
            for name,w in params.items():
                torch.testing.assert_close(state.ema_params[fac][name],w,rtol=0,atol=0)
        td,to=teachers()
        step_fn(state,statics.loss,td,to,fixed,gen_cuda)
        ema_update(state)
        assert state.step==args.steps+1 and state.last_update_succeeded
        result=dict(mode=mode,family=args.family,steps=args.steps,restored_steps=state.step,
                    batch_size=args.batch_size,div_loss_weighting=args.div_loss_weighting,parameters=sum(p.numel() for p in net.parameters()),
                    elapsed_seconds=elapsed,ms_per_step=1000*elapsed/args.steps,
                    max_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_grad_norm=peak_grad,before=before,after=after,forward_logp=logp_info,
                    velocity_head_max_update=changes['dit.final_linear.weight'],
                    density_head_max_update=changes.get('dit.div_final_linear.weight'),
                    checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint))
        results.append(result)
        print(json.dumps(result),flush=True)
        Path(args.output).write_text(json.dumps(dict(device=torch.cuda.get_device_name(0),
            torch_version=torch.__version__,source_velocity=args.velocity_checkpoint,
            source_flow=args.flow_checkpoint,results=results),indent=2)+'\n')
        del statics,state,net,initial,saved
        torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    from common.com_lift import add_com_cli_arguments
    add_com_cli_arguments(parser)
    parser.add_argument('--family',choices=['lsd','esd','psd','esd_cond','esd_cond_div_only'],required=True)
    parser.add_argument('--velocity_checkpoint',required=True)
    parser.add_argument('--flow_checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint_dir',required=True)
    parser.add_argument('--div_loss_weighting',choices=['uniform','linear','square'],default='linear')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=64)
    run(parser.parse_args())
