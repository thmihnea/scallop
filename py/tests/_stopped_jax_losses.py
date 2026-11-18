"""Independent per-sample JAX oracle for the selected correction contract.

Uses the unchanged vendored JAX NETWORK, not its historical loss functions.
Sources are stopped; LSD convex keeps both time corrections differentiable
as requested on 2026-09-10. Other policies retain the 2026-09-08 stopped update.
This is a test oracle only, not a second production training implementation.
"""
import math
import jax
import jax.numpy as jnp

sg=jax.lax.stop_gradient


def raw(X,p,s,t,x):
    result=X.apply(p,s,t,x,method='calc_phi',train=False,return_div=X.config.predict_divergence)
    return result if X.config.predict_divergence else (result,None)


def scale(X):
    return math.prod(X.config.input_dims) if X.config.vectorized_divergence else 1


def weighted(X,p,s,t,value):
    # Production fixes the old constant logvar=1 when weighting was disabled.
    w=X.apply(p,s,t,method='calc_weight') if X.config.use_weight else 0.
    return jnp.exp(-w)*value+w


def _calc_divergence_with_phi(p,t,x,z,data,label,rng,*,X,method='calc_b'):
    fn=lambda xx:X.apply(p,t,xx,method=method,train=False)
    full=getattr(X.config,'com_aware',False)
    project=lambda a:a if full else a-jnp.mean(a,axis=-2,keepdims=True)
    mode=X.config.divergence_mode
    if mode=='hutchinson':
        eps=project(jax.random.normal(rng['dropout'],x.shape))
        v,pullback=jax.vjp(fn,x)
        q=pullback(eps)[0]*eps
    else:
        v=fn(x)
        if mode in ('pretrain_v','pretrain_flowmap'):
            return sg(v),None
        c=1. if full else 1.-1./x.shape[-2]
        eta=data-z
        if mode=='dlogpdt':
            q=((v-eta)*project(z)/X.config.rescale**2-c)/(1-t)
        elif mode=='cond_trace':
            q=(-(v-eta)*eta*t/X.config.rescale**2-c)/(1-t)
        else:
            raise ValueError(mode)
    return sg(v),sg(q if X.config.vectorized_divergence else jnp.sum(q))


def diagonal_term(p,teacher,z,data,label,t,rng,*,interp,X):
    x=interp.calc_It(t,z,data)
    u,d=raw(X,p,t,t,x)
    if X.config.divergence_mode=='pretrain_v':
        target_u=interp.calc_It_dot(t,z,data)
        target_d=None
    else:
        ref=p if X.config.from_scratch or teacher is None else teacher
        target_u,target_d=_calc_divergence_with_phi(ref,t,x,z,data,label,rng,X=X)
        if X.config.from_scratch or teacher is None:
            target_u=interp.calc_It_dot(t,z,data)
    value=jnp.sum((u-sg(target_u))**2)
    if d is not None:
        target_d=sg(target_d/X.config.divergence_rescale)
        factor={'uniform':1.,'linear':1-t,'square':(1-t)**2}[X.config.div_loss_weighting_type]
        error = d**2-2*d*target_d if X.config.div_loss_weighting_type == 'linear' and not X.config.use_weight else (d-target_d)**2
        value=value+jnp.sum(error)*factor/scale(X)
    return weighted(X,p,t,t,value),{}


def lsd_term(p,teacher,z,data,label,s,t,rng,*,interp,X,stopgrad_type='stopped',density_ema_params=None):
    x=interp.calc_It(s,z,data)
    fn=lambda tt:raw(X,p,s,tt,x)
    # Keep None out of a differentiated pytree when no density head exists.
    if X.config.predict_divergence:
        (u,d),(du,dd)=jax.jvp(fn,(t,),(jnp.ones_like(t),))
    else:
        u,du=jax.jvp(lambda tt:fn(tt)[0],(t,),(jnp.ones_like(t),))
        d=None
    y=sg(x+(t-s)*u)
    b=X.apply(teacher,t,y,method='calc_b',train=False)
    residual_u = u+(t-s)*du-sg(b) if stopgrad_type == 'convex' else u-sg(b-(t-s)*du)
    value=jnp.sum(residual_u**2)
    if d is not None:
        ema=p if density_ema_params is None else density_ema_params
        g=raw(X,ema,t,t,y)[1]
        residual_d = d+(t-s)*dd-sg(g) if stopgrad_type == 'convex' else d-sg(g-(t-s)*dd)
        value=value+jnp.sum(residual_d**2)/scale(X)
    return weighted(X,p,s,t,value),{}


def esd_term(p,teacher,z,data,label,s,t,rng,*,interp,X,stopgrad_type='stopped',density_ema_params=None):
    x=interp.calc_It(s,z,data)
    b=sg(X.apply(teacher,s,x,method='calc_b',train=False))
    if X.config.predict_divergence:
        (u,d),(du,dd)=jax.jvp(lambda xx,ss:raw(X,p,ss,t,xx),(x,s),(b,jnp.ones_like(s)))
    else:
        u,du=jax.jvp(lambda xx,ss:raw(X,p,ss,t,xx)[0],(x,s),(b,jnp.ones_like(s)))
        d=None
    value=jnp.sum((u-sg(b+(t-s)*du))**2)
    if d is not None:
        ema=p if density_ema_params is None else density_ema_params
        g=raw(X,ema,s,s,x)[1]
        value=value+jnp.sum((d-sg(g+(t-s)*dd))**2)/scale(X)
    return weighted(X,p,s,t,value)


def psd_term(p,teacher,z,data,label,s,t,a,h,rng,*,interp,X,psd_type,stopgrad_type='stopped',density_ema_params=None):
    x=interp.calc_It(s,z,data)
    u,d=raw(X,p,s,t,x)
    w=.5 if psd_type=='midpoint' else h
    def targets(params):
        u1,d1=raw(X,params,s,a,x)
        y=x+(a-s)*u1
        u2,d2=raw(X,params,a,t,y)
        return (1-w)*u1+w*u2,None if d1 is None else (1-w)*d1+w*d2
    target_u=targets(teacher)[0]
    value=jnp.sum((u-sg(target_u))**2)
    if d is not None:
        ema=p if density_ema_params is None else density_ema_params
        target_d=targets(ema)[1]
        value=value+jnp.sum((d-sg(target_d))**2)/scale(X)
    return weighted(X,p,s,t,value)


def setup_loss(cfg,net,interp):
    def loss(p,teacher_diag,teacher_off,z,data,label,s,t,u,h,keys):
        n=len(z)
        nd=max(1,int(n*cfg.optimization.diag_fraction))
        value=0.
        if nd:
            def diag(zz,xx,tt,key):
                return diagonal_term(p,teacher_diag,zz,xx,None,tt,{'dropout':key},interp=interp,X=net)[0]
            value=jnp.sum(jax.vmap(diag)(z[:nd],data[:nd],t[:nd],keys[:nd]))
        if nd<n:
            def off(zz,xx,ss,tt,key):
                return lsd_term(p,teacher_off,zz,xx,None,ss,tt,{'dropout':key},
                                interp=interp,X=net,stopgrad_type=cfg.training.stopgrad_type)[0]
            value=value+jnp.sum(jax.vmap(off)(z[nd:],data[nd:],s[nd:],t[nd:],keys[nd:]))
        return value/n,{}
    return loss
