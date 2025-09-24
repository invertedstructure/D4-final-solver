
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
    @validator('masks')
    def check_masks(cls, v):
        for k, mat in v.items():
            if not all(c in (0,1) for row in mat for c in row):
                raise ValueError(f"Support.masks['{k}'] must be 0/1")
        return v

class Pairings(BaseModel):
    data: dict

class Reps(BaseModel):
    data: dict

class TriangleDegree(BaseModel):
    A: IntMat
    B: IntMat
    J: IntMat
    @root_validator
    def check_gf2_dims(cls, values):
        for name in ('A','B','J'):
            mat = values.get(name)
            if not _is_gf2_matrix(mat):
                raise ValueError(f"Triangle.{name} must be GF(2)")
        return values

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
        r, c = _shape(mat)
        if (r, c) != (n_km1, n_k):
            raise ValueError(f"∂_{k} has shape {r}x{c}; expected {n_km1}x{n_k}")

def check_support_against_cmap(support: Support, cmap: CMap):
    for k, mask in support.masks.items():
        mat = cmap.blocks.__root__.get(k)
        if mat is None:
            raise ValueError(f"Support degree '{k}' not present in CMap")
        from .linalg_gf2 import shape as _shape2
        if _shape2(mask) != _shape2(mat):
            raise ValueError(f"Support mask at degree {k} has wrong shape")
