"""Small optional float64 Torch kernels; CPU remains the reference backend."""
import numpy as np


def torch_device(device):
    # Input: cpu/auto/cuda device; output: available Torch device; module function.
    import torch
    selected='cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device
    if str(selected).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested but is unavailable in this Python environment')
    return torch.device(selected)


def cosine_rows(a,b,device='cpu'):
    # Input: broadcastable vectors and device; output: float64 row cosines, with zero for zero norms.
    if device=='cpu':
        a,b=np.asarray(a),np.asarray(b); denom=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        return np.divide((a*b).sum(-1),denom,out=np.zeros_like(denom,dtype=float),where=denom>0)
    import torch
    with torch.inference_mode():
        a=torch.as_tensor(a,dtype=torch.float64,device=torch_device(device))
        b=torch.as_tensor(b,dtype=torch.float64,device=a.device)
        denom=torch.linalg.vector_norm(a,dim=-1)*torch.linalg.vector_norm(b,dim=-1)
        return torch.where(denom>0,(a*b).sum(-1)/denom.clamp_min(torch.finfo(a.dtype).tiny),0.).cpu().numpy()


def matrix_product(a,b,device='cpu',block_size=512):
    # Input: two matrices of row vectors/device; output: blocked a @ b.T on the chosen backend.
    if device=='cpu': return a@b.T
    import torch
    with torch.inference_mode():
        dev=torch_device(device); right=torch.as_tensor(b,dtype=torch.float64,device=dev)
        result=np.empty((len(a),len(b)))
        for start in range(0,len(a),block_size):
            left=torch.as_tensor(a[start:start+block_size],dtype=torch.float64,device=dev)
            result[start:start+len(left)]=(left@right.T).cpu().numpy()
        return result


def quantile_pair_values(features,variances,names,left,right,name_left,name_right,device,chunk_size=1024):
    # Input: reusable column-state features and pair indices; output: raw value distances/name cosines.
    import torch
    with torch.inference_mode():
        dev=torch_device(device)
        q=torch.as_tensor(features,dtype=torch.float64,device=dev)
        v=torch.as_tensor(variances,dtype=torch.float64,device=dev)
        n=torch.as_tensor(names,dtype=torch.float64,device=dev)
        distances=[]; cosines=[]
        def cos(a,b):
            denom=torch.linalg.vector_norm(a,dim=-1)*torch.linalg.vector_norm(b,dim=-1)
            return torch.where(denom>0,(a*b).sum(-1)/denom.clamp_min(torch.finfo(a.dtype).tiny),0.)
        for start in range(0,len(left),chunk_size):
            end=start+chunk_size
            a=torch.as_tensor(left[start:end],device=dev); b=torch.as_tensor(right[start:end],device=dev)
            w=torch.exp(-torch.abs(v[a]-v[b])); w=w/w.sum(-1,keepdim=True)
            distances.append((w*torch.clamp(1.-cos(q[a],q[b]),min=0.)).sum(-1).cpu().numpy())
            i=torch.as_tensor(name_left[start:end],device=dev); j=torch.as_tensor(name_right[start:end],device=dev)
            cosines.append(cos(n[i],n[j]).cpu().numpy())
        return np.concatenate(distances),np.concatenate(cosines)


def knn_impute(matrix,k,device='auto',block_size=256):
    # Input: scaled numeric matrix with NaNs; output: sklearn-compatible distance-weighted KNN fill.
    import torch
    x=np.asarray(matrix,dtype=float); out=x.copy(); missing=np.isnan(x)
    if np.isinf(x).any() or np.all(missing,axis=0).any():
        raise ValueError('KNN backend requires no infinities or entirely missing columns')
    queries=np.flatnonzero(missing.any(axis=1))
    if not len(queries): return out
    with torch.inference_mode():
        dev=torch_device(device); values=torch.as_tensor(np.nan_to_num(x),dtype=torch.float64,device=dev)
        masks=torch.as_tensor(~missing,dtype=torch.float64,device=dev)
        for start in range(0,len(queries),block_size):
            rows=queries[start:start+block_size]; a=values[rows]; ma=masks[rows]
            common=ma@masks.T
            squared=(a*a)@masks.T+ma@(values*values).T-2.*a@values.T
            d=torch.sqrt(squared.clamp_min(0.)*x.shape[1]/common.clamp_min(1.))
            d=torch.where(common>0,d,torch.nan).cpu().numpy()
            for j in range(x.shape[1]):
                receivers=np.flatnonzero(missing[rows,j])
                donors=np.flatnonzero(~missing[:,j])
                if not len(receivers): continue
                distances=d[np.ix_(receivers,donors)]
                for local,receiver in enumerate(receivers):
                    dist=distances[local]
                    if np.isnan(dist).all(): out[rows[receiver],j]=np.nanmean(x[:,j]); continue
                    chosen=np.argpartition(dist,min(k,len(donors))-1)[:min(k,len(donors))]
                    dd=dist[chosen]; vv=x[donors[chosen],j]
                    if np.any(dd==0): weights=(dd==0).astype(float)
                    else: weights=np.divide(1.,dd,out=np.zeros_like(dd),where=np.isfinite(dd))
                    out[rows[receiver],j]=np.dot(weights,vv)/weights.sum()
    return out
