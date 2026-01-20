"""Smoke regression tests for Phase‑1 Step‑A contract.

Run:
  python -m unittest test_stepa_phase1_smoke.py

These tests are intentionally small and dependency-free (unittest only).
They are *not* exhaustive; they assert the frozen invariants on a tiny toy suite.
"""

from __future__ import annotations

import unittest

import stepa_v1 as S


class TestStepAPhase1Smoke(unittest.TestCase):
    def test_deterministic_cert_and_fp(self) -> None:
        A = [[1, 0], [0, 0]]
        c1 = S.stepA_cert(A)
        c2 = S.stepA_cert(A)
        self.assertEqual(c1, c2)
        fp1 = S.fpv1_sha256(c1)
        fp2 = S.fpv1_sha256(c2)
        self.assertEqual(fp1, fp2)
        self.assertTrue(fp1.get("defined"))
        self.assertIsInstance(fp1.get("sha256"), str)

    def test_maps_preserve_fp_when_defined(self) -> None:
        A = [[1, 0], [0, 0]]
        c0 = S.stepA_cert(A)
        self.assertTrue(c0.get("defined"))
        fp0 = S.fpv1_sha256(c0)

        for entry in S.default_map_suite_entries():
            mp = S.map_entry_to_concrete_ops(entry, n_cols=S.shape(A)[1])
            A1 = S.apply_stepA_map(A, mp)
            # Type gate should pass for admissible maps.
            tg = S.stepA_type_gate(A, A1)
            self.assertEqual(tg.get("status"), "PASS")
            # Fingerprint invariance (Cor.2) for this toy suite.
            c1 = S.stepA_cert(A1)
            fp1 = S.fpv1_sha256(c1)
            self.assertEqual(fp0.get("sha256"), fp1.get("sha256"))

    def test_component_transport_matches_recompute(self) -> None:
        A = [[1, 0], [0, 0]]
        c0 = S.stepA_cert(A)
        y = c0["y"]
        comp0 = S.A_comp_with_y(A, y)

        for entry in S.default_map_suite_entries():
            mp = S.map_entry_to_concrete_ops(entry, n_cols=S.shape(A)[1])
            A1 = S.apply_stepA_map(A, mp)
            comp_hat = S.transport_A_comp(A, y, comp0, mp)
            comp1 = S.A_comp_with_y(A1, y)
            self.assertEqual(comp_hat, comp1)


if __name__ == "__main__":
    unittest.main()
