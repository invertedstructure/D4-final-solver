from __future__ import annotations
from typing import List

IntMat = List[List[int]]

def shape(M: IntMat) -> tuple[int,int]:
    return (len(M), len(M[0]) if M else 0)

def eye(n: int) -> IntMat:
    return [[1 if i==j else 0 for j in range(n)] for i in range(n)]

def add(A: IntMat, B: IntMat) -> IntMat:
    r, c = shape(A)
    return [[(A[i][j] ^ B[i][j]) for j in range(c)] for i in range(r)]

def mul(A: IntMat, B: IntMat) -> IntMat:
    r, k = shape(A)
    k2, c = shape(B)
    assert k == k2, f"mul shape mismatch: {shape(A)} x {shape(B)}"
    out = [[0]*c for _ in range(r)]
    for i in range(r):
        for t in range(k):
            if A[i][t] == 0: 
                continue
            rowA = A[i][t]
            Bt = B[t]
            for j in range(c):
                out[i][j] ^= (rowA & Bt[j])
    return out

def zeros(r: int, c: int) -> IntMat:
    return [[0]*c for _ in range(r)]

def is_zero(M: IntMat) -> bool:
    return all(v == 0 for row in M for v in row)
