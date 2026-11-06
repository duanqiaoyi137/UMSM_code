"""Float64 BayesianRidge estimator for sklearn's unchanged MICE scheduler.

Uses the same empirical-Bayes updates, priors and stopping rule as sklearn.
Posterior sampling, feature order and bounds stay in IterativeImputer.
"""
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.utils.validation import check_X_y,check_array,check_is_fitted

from ..numerics import torch_device


class TorchBayesianRidge(RegressorMixin,BaseEstimator):
    def __init__(self,device='auto',max_iter=300,tol=1e-3,alpha_1=1e-6,alpha_2=1e-6,
                 lambda_1=1e-6,lambda_2=1e-6,alpha_init=None,lambda_init=None,fit_intercept=True):
        # Input: BayesianRidge priors/stopping settings/device; output: unfitted sklearn-compatible estimator.
        self.device,self.max_iter,self.tol=device,max_iter,tol
        self.alpha_1,self.alpha_2,self.lambda_1,self.lambda_2=alpha_1,alpha_2,lambda_1,lambda_2
        self.alpha_init,self.lambda_init,self.fit_intercept=alpha_init,lambda_init,fit_intercept

    def fit(self,X,y):
        # Input: complete numeric training predictors/observed targets; output: posterior estimator; sklearn override.
        import torch
        X,y=check_X_y(X,y,dtype=np.float64); self.n_features_in_=X.shape[1]
        with torch.inference_mode():
            dev=torch_device(self.device)
            x=torch.as_tensor(X,dtype=torch.float64,device=dev); target=torch.as_tensor(y,dtype=torch.float64,device=dev)
            n,p=x.shape
            offset=x.mean(0) if self.fit_intercept else torch.zeros(p,dtype=x.dtype,device=dev)
            yoffset=target.mean() if self.fit_intercept else torch.zeros((),dtype=x.dtype,device=dev)
            variance=target.var(correction=0); x=x-offset; target=target-yoffset
            alpha=1./(variance+np.finfo(float).eps) if self.alpha_init is None else torch.tensor(self.alpha_init,dtype=x.dtype,device=dev)
            penalty=torch.tensor(1. if self.lambda_init is None else self.lambda_init,dtype=x.dtype,device=dev)
            kwargs={'driver':'gesvd'} if dev.type=='cuda' else {}
            u,s,vfull=torch.linalg.svd(x,full_matrices=(n<p),**kwargs)
            eigen=s*s; v=vfull[:len(s)]; xty=x.T@target
            def update(a,l):
                if n>p: coef=v.T@((v@xty)/(eigen+l/a))
                else: coef=x.T@(u@((u.T@target)/(eigen+l/a)))
                return coef,torch.sum((target-x@coef)**2)
            previous=None
            for iteration in range(self.max_iter):
                coef,sse=update(alpha,penalty)
                gamma=torch.sum(alpha*eigen/(penalty+alpha*eigen))
                penalty=(gamma+2*self.lambda_1)/(torch.sum(coef**2)+2*self.lambda_2)
                alpha=(n-gamma+2*self.alpha_1)/(sse+2*self.alpha_2)
                if previous is not None and torch.sum(torch.abs(previous-coef)).item()<self.tol: break
                previous=coef
            self.n_iter_=iteration+1; coef,_=update(alpha,penalty)
            full_eigen=torch.zeros(p,dtype=x.dtype,device=dev); full_eigen[:len(s)]=eigen
            covariance=vfull.T@(vfull/(alpha*full_eigen+penalty)[:,None])
            self._coef,self._cov,self._offset,self._intercept=coef,covariance,offset,yoffset-offset@coef
            self._alpha=alpha; self.device_=str(dev)
            self.coef_=coef.cpu().numpy(); self.sigma_=covariance.cpu().numpy()
            self.X_offset_=offset.cpu().numpy(); self.intercept_=float(self._intercept.cpu())
            self.alpha_,self.lambda_=float(alpha.cpu()),float(penalty.cpu())
            if not np.isfinite(self.sigma_).all() or not np.isfinite(self.coef_).all():
                raise FloatingPointError('Nonfinite Bayesian posterior; the imputation method was not substituted')
        return self

    def predict(self,X,return_std=False):
        # Input: complete query predictors; output: predictive mean and optional std; sklearn override.
        import torch
        check_is_fitted(self,'coef_'); X=check_array(X,dtype=np.float64)
        if X.shape[1]!=self.n_features_in_: raise ValueError('Feature count changed')
        with torch.inference_mode():
            x=torch.as_tensor(X,dtype=torch.float64,device=self._coef.device)
            mean=x@self._coef+self._intercept
            if not return_std: return mean.cpu().numpy()
            centered=x-self._offset
            variance=((centered@self._cov)*centered).sum(1)+1./self._alpha
            # Ill-conditioned designs can leave a negative round-off in the
            # quadratic form of the covariance.  The posterior mean is unchanged;
            # clamp the numerical artifact before taking the square root.
            if not torch.isfinite(variance).all():
                raise FloatingPointError('Nonfinite posterior predictive variance')
            return mean.cpu().numpy(),torch.sqrt(variance.clamp_min(0.)).cpu().numpy()


def mice_estimator(backend,device):
    # Input: selected numerical backend/device; output: estimator or sklearn's default sentinel.
    if backend=='sklearn': return None
    if backend=='torch': return TorchBayesianRidge(device=device)
    raise ValueError('numerical_backend must be sklearn or torch')
