from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel, Field, root_validator, validator

IntMat = List[List[int]]

def _is_gf2_matrix(m: IntMat) -> bool:
    return all((c in (0,1) for row in m for c in row))

def _shape(m: IntMat) -> tuple[int,int]:
    return (len(m), len(m[0]) if m and isinstance(m[0], list) else 0)

class Blocks(BaseModel):
    __root__: Dict[str, IntMat]
    @validator('__root__')
    def check_gf2(cls, v):
        for k, mat in v.items():
            if not _is_gf2_matrix(mat):
                raise ValueError(f"blocks['{k}'] not GF(2)")
            if len({len(r) for r in mat if isinstance(r, list)}) > 1:
                raise ValueError(f"blocks['{k}'] rows have inconsistent lengths")
        return v
    def degrees(self) -> List[int]:
        return sorted(int(k) for k in self.__root__.keys())
    def get(self, k: int) -> IntMat:
        return self.__root__[str(k)]

class Boundaries(BaseModel):
    blocks: Blocks

class Shapes(BaseModel):
    n: Dict[str, int] = Field(...)
    @validator('n')
    def check_positive(cls, v):
        for k, val in v.items():
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"Shapes.n['{k}'] must be nonnegative int")
        return v

class CMap(BaseModel):
    blocks: Blocks

class Support(BaseModel):
    masks: Dict[str, IntMat]

class Pairings(BaseModel):
    data: dict

class Reps(BaseModel):
    data: dict

class TriangleDegree(BaseModel):
    A: IntMat
    B: IntMat
    J: IntMat

class TriangleSchema(BaseModel):
    by_degree: Dict[str, TriangleDegree]
    @root_validator(pre=True)
    def reshape_input(cls, values):
        if 'by_degree' in values: return values
        return {'by_degree': values}
    def get(self, k: int) -> TriangleDegree:
        return self.by_degree[str(k)]

def check_cmap_square_against_shapes(cmap: CMap, shapes: Shapes):
    for k, mat in cmap.blocks.__root__.items():
        n = shapes.n.get(k)
        if n is None: raise ValueError(f"Shapes missing degree '{k}'")
        r, c = _shape(mat)
        if (r, c) != (n, n):
            raise ValueError(f"CMap.blocks['{k}'] has shape {r}x{c}; expected {n}x{n}")

def check_boundaries_against_shapes(bounds: Boundaries, shapes: Shapes):
    for k, mat in bounds.blocks.__root__.items():
        k_int = int(k)
        n_k = shapes.n.get(k)
        n_km1 = shapes.n.get(str(k_int - 1))
        if n_k is None or n_km1 is None:
            raise ValueError(f"Shapes missing n for degrees {k} or {k_int-1}")
        r = len(mat)
        c = len(mat[0]) if r>0 else n_k  # assume expected cols when zero rows
        if n_km1 == 0:
            if r != 0:
                raise ValueError(f"∂_{k} should have 0 rows (n_{k-1}=0); got {r}")
            continue
        if (r, c) != (n_km1, n_k):
            raise ValueError(f"∂_{k} has shape {r}x{c}; expected {n_km1}x{n_k}")

def check_support_against_cmap(support: Support, cmap: CMap):
    """Ensure each mask has same shape as the corresponding CMap block.
       Accept either n×n or empty list when n==0."""
    for k, mask in support.masks.items():
        mat = cmap.blocks.__root__.get(k)
        if mat is None:
            raise ValueError(f"Support degree '{k}' not present in CMap")
        # shapes must match; allow [] only if the corresponding C block is 0×0
        rM = len(mat); cM = len(mat[0]) if rM>0 else 0
        rS = len(mask); cS = len(mask[0]) if rS>0 else cM
        if rM == 0 and rS == 0:
            continue
        if (rM, cM) != (rS, cS):
            raise ValueError(f"Support mask at degree {k} has shape {rS}x{cS}; expected {rM}x{cM}")
