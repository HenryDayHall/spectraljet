#%%%
import sys
import itertools
from typing import Tuple, Any, Union, List
from numpy import linalg
import torch.jit
import functools
import torch

import 

#%%%











#%%%
ndim = 1
name = "Sphere"
reversible = False
#EPS = {torch.float32: 1e-4, torch.float64: 1e-7}

def broadcast_shapes(*shapes: Tuple[int]) -> Tuple[int]:
    #Apply numpy broadcasting rules to shapes
        result = []
        for dims in itertools.zip_longest(*map(reversed, shapes), fillvalue=1):
            dim: int = 1
            for d in dims:
                if dim != 1 and d != 1 and d != dim:
                    raise ValueError("Shapes can't be broadcasted")
                elif d > dim:
                    dim = d
            result.append(dim)
        return tuple(reversed(result))


class Sphere(Manifold):

    def __init__(self, K=1.0):
        super(Sphere, self).__init__()
        if torch.is_tensor(K):
            self.K = K
        else:
            self.K = torch.tensor(K)
        self.EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u
        inner = (u * v).sum(-1, keepdim=keepdim)
        target_shape = broadcast_shapes(x.shape[:-1] + (1,) * keepdim, inner.shape)
        return inner.expand(target_shape)
    
    
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        x = self._project_on_subspace(x)
        return x / x.norm(dim=-1, keepdim=True)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        return self._project_on_subspace(u)

    def exp(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.projx(x + u)
        cond = norm_u > self.EPS[norm_u.dtype]
        return torch.where(cond, exp, retr)

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.projx(x + u)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.proju(y, v)

    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist.gt(self.EPS[x.dtype])
        result = torch.where(
            cond, u * dist / u.norm(dim=-1, keepdim=True).clamp_min(self.EPS[x.dtype]), u
        )
        return result

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        inner = self.inner(x, x, y, keepdim=keepdim).clamp(
            -1 + self.EPS[x.dtype], 1 - self.EPS[x.dtype]
        )
        return torch.acos(inner)

    #egrad2rgrad = proju

    def __str__(self):
        return 'sphere'
    
    def dim_to_sh(self, dim):
        if hasattr(dim, '__iter__'):
            return dim[-1]
        else:
            return dim
    
    def sh_to_dim(self, sh):
        if hasattr(sh, '__iter__'):
            return sh[-1]
        else:
            return sh
    
    def squeeze_tangent(self, x):
        return x
    
    def unsqueeze_tangent(self, x):
        return x
    
    def zero(self, *shape):
        return torch.zeros(*shape)

    def zero_tan(self, *shape):
        return torch.zeros(*shape)

    def zero_like(self, x):
        return torch.zeros_like(x)

    def zero_tan_like(self, x):
        return torch.zeros_like(x)



#%%%

# Variable Instantiation
man = Sphere()


#%%
x = torch.rand(5, 3)
x *= torch.rand(5, 1) / x.norm(dim=-1, keepdim=True)
x = torch.nn.Parameter(x)
w = torch.nn.Parameter(torch.rand(5)) #use ones or pass in None for non-weighted mean

# computation
y = frechet_mean(x, man, w)

# differentiation
y.sum().backward()
#print(x.grad, w.grad)

print(y)

# %%
