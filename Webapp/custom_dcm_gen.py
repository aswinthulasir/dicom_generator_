"""
DICOM Phantom Generator – MULTI PATIENT
=================================================
• User selects modality: CT (axial) or MR (sagittal T1)
• Fixed study dates (5 series, 1 per patient per day):
      2026-04-01 to 2026-04-05
• All series have DIFFERENT Patient IDs, DIFFERENT StudyInstanceUIDs, and DIFFERENT Dates

Usage:
    pip install pydicom numpy scipy
    python create_body_dicom_patient2.py

Opens cleanly in: 3D Slicer, Horos, OsiriX, MicroDicom, RadiAnt
"""

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from scipy.ndimage import gaussian_filter
import datetime
import os
import random
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Hounsfield Unit (HU) reference values for CT
# ──────────────────────────────────────────────────────────────────────────────
HU = dict(
    air           = -1024,
    lung          = -700,
    lung_vessel   = -150,
    fat           = -80,
    water         = 0,
    soft_tissue   = 40,
    muscle        = 50,
    liver         = 62,
    spleen        = 50,
    kidney_cortex = 40,
    kidney_medull = 20,
    pancreas      = 42,
    gallbladder   = 12,
    bladder       = 8,
    blood         = 42,
    bowel_air     = -400,
    bowel_fluid   = 20,
    bone_cancel   = 250,
    bone_cortex   = 750,
    aorta_lumen   = 42,
    csf           = 10,
)

# MR T1 signal intensity reference (0–1023 scale, unsigned)
MR_SIG = dict(
    air         = 0,
    fat         = 900,
    muscle      = 480,
    soft_tissue = 430,
    liver       = 510,
    spleen      = 460,
    kidney      = 440,
    fluid_csf   = 60,
    blood       = 400,
    bone_marrow = 780,
    bone_cortex = 55,
    disc        = 90,
    lung        = 55,
    bladder     = 65,
    cord        = 520,
)

# ──────────────────────────────────────────────────────────────────────────────
# Study schedule: 4 studies, 1 patient, paired series UIDs
# ──────────────────────────────────────────────────────────────────────────────
# Study pair A (studies 1 & 2): 3 series each, shared SeriesInstanceUIDs
# Study pair B (studies 3 & 4): 4 series each, shared SeriesInstanceUIDs
# All 4 studies share the SAME PatientID but have DIFFERENT StudyInstanceUIDs
# and DIFFERENT dates.
# ──────────────────────────────────────────────────────────────────────────────

PATIENT_ID   = "PAT-200001"
PATIENT_NAME = "PHANTOM^PATIENT"

STUDY_SCHEDULE = [
    # (StudyDate,    StudyTime,  n_series,  pair_group)
    ("20260401", "090000", 3, "A"),   # Study 1  – pair A
    ("20260403", "100000", 3, "A"),   # Study 2  – pair A  (same 3 series UIDs)
    ("20260405", "090000", 4, "B"),   # Study 3  – pair B
    ("20260407", "100000", 4, "B"),   # Study 4  – pair B  (same 4 series UIDs)
]

# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def ellipse(shape, center, axes):
    rows, cols = shape
    cy, cx = center
    ry, rx = max(axes[0], 1), max(axes[1], 1)
    y, x = np.ogrid[:rows, :cols]
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1


def ring(shape, center, outer_axes, inner_axes):
    return ellipse(shape, center, outer_axes) & ~ellipse(shape, center, inner_axes)


def paint(image, mask, value, noise_std=5):
    if not mask.any():
        return
    n = np.random.normal(value, noise_std, image.shape)
    image[mask] = n[mask]


# ──────────────────────────────────────────────────────────────────────────────
# CT pixel generator – AXIAL
# ──────────────────────────────────────────────────────────────────────────────

def generate_ct_slice(slice_idx: int, num_slices: int,
                      rows: int = 512, cols: int = 512) -> np.ndarray:
    image = np.full((rows, cols), HU['air'], dtype=np.float32)
    sh  = (rows, cols)
    pos = slice_idx / max(num_slices - 1, 1)
    cy, cx = rows // 2, cols // 2

    if pos < 0.12:
        bry, brx = int(rows * 0.23), int(cols * 0.19)
    elif pos < 0.48:
        bry, brx = int(rows * 0.30), int(cols * 0.25)
    elif pos < 0.72:
        bry, brx = int(rows * 0.33), int(cols * 0.28)
    else:
        bry, brx = int(rows * 0.31), int(cols * 0.27)

    body_mask = ellipse(sh, (cy, cx), (bry, brx))
    paint(image, body_mask, HU['soft_tissue'], noise_std=6)
    skin = body_mask & ~ellipse(sh, (cy, cx), (bry - 4, brx - 4))
    paint(image, skin, HU['soft_tissue'] + 10, noise_std=8)

    fat_ry, fat_rx     = bry - 5, brx - 5
    fat_in_ry          = fat_ry - int(rows * 0.022)
    fat_in_rx          = fat_rx - int(cols * 0.022)
    paint(image, ring(sh, (cy, cx), (fat_ry, fat_rx), (fat_in_ry, fat_in_rx)),
          HU['fat'], noise_std=15)

    msc_out_ry, msc_out_rx = fat_in_ry, fat_in_rx
    msc_in_ry = msc_out_ry - int(rows * 0.030)
    msc_in_rx = msc_out_rx - int(cols * 0.025)
    paint(image, ring(sh, (cy, cx), (msc_out_ry, msc_out_rx), (msc_in_ry, msc_in_rx)),
          HU['muscle'], noise_std=10)
    paint(image, ellipse(sh, (cy, cx), (msc_in_ry, msc_in_rx)), HU['fat'] + 30, noise_std=8)

    sp_cy = cy + int(bry * 0.62)
    sp_cx = cx
    sp_r  = int(rows * 0.042)
    paint(image, ellipse(sh, (sp_cy, sp_cx), (sp_r, sp_r)), HU['bone_cancel'], noise_std=35)
    paint(image, ring(sh, (sp_cy, sp_cx), (sp_r, sp_r), (sp_r - 4, sp_r - 4)),
          HU['bone_cortex'], noise_std=40)
    canal_r = max(int(sp_r * 0.38), 4)
    paint(image, ellipse(sh, (sp_cy, sp_cx), (canal_r, canal_r)), HU['csf'], noise_std=4)
    paint(image, ellipse(sh, (sp_cy + sp_r + 6, sp_cx), (9, 6)), HU['bone_cortex'] - 100, noise_std=40)
    for side in (-1, 1):
        paint(image, ellipse(sh, (sp_cy, sp_cx + side * (sp_r + 14)), (5, 9)),
              HU['bone_cortex'] - 100, noise_std=40)
        paint(image, ellipse(sh, (sp_cy - 4, sp_cx + side * int(cols * 0.075)),
                             (int(rows * 0.058), int(cols * 0.042))), HU['muscle'], noise_std=10)

    ao_cy = sp_cy - sp_r - 13
    ao_cx = cx - int(cols * 0.014)
    ao_r  = int(rows * 0.020)
    paint(image, ellipse(sh, (ao_cy, ao_cx), (ao_r, ao_r)), HU['aorta_lumen'], noise_std=6)
    paint(image, ring(sh, (ao_cy, ao_cx), (ao_r, ao_r), (ao_r - 3, ao_r - 3)),
          HU['muscle'] + 20, noise_std=8)
    paint(image, ellipse(sh, (ao_cy, cx + int(cols * 0.022)),
                         (int(rows * 0.016), int(cols * 0.020))), HU['blood'] - 5, noise_std=6)

    # ── Region-specific anatomy ───────────────────────────────────────────────
    if pos < 0.12:
        t = pos / 0.12
        l_ry = int(rows * (0.08 + 0.10 * t))
        l_rx = int(cols * (0.07 + 0.08 * t))
        for side in (-1, 1):
            paint(image, ellipse(sh, (cy - int(rows * 0.02), cx + side * int(cols * 0.11)),
                                 (l_ry, l_rx)), HU['lung'], noise_std=30)
        for side in (-1, 1):
            paint(image, ellipse(sh, (cy - int(rows * 0.05), cx + side * int(cols * 0.14)),
                                 (7, 20)), HU['bone_cortex'] - 50, noise_std=40)

    elif pos < 0.48:
        t = (pos - 0.12) / 0.36
        lung_ry   = int(rows * (0.17 + 0.03 * min(t * 2, 1.0) - 0.02 * max(t - 0.5, 0)))
        lung_rx_l = int(cols * (0.10 + 0.03 * min(t * 2, 1.0)))
        lung_rx_r = int(cols * (0.11 + 0.03 * min(t * 2, 1.0)))
        lcx = cx - int(cols * 0.13)
        rcx = cx + int(cols * 0.14)
        lcy = cy - int(rows * 0.02)
        l_lung = ellipse(sh, (lcy, lcx), (lung_ry, lung_rx_l))
        r_lung = ellipse(sh, (lcy, rcx), (lung_ry, lung_rx_r))
        paint(image, l_lung, HU['lung'], noise_std=35)
        paint(image, r_lung, HU['lung'], noise_std=35)
        for cx_l, mask_l in ((lcx, l_lung), (rcx, r_lung)):
            for _ in range(18):
                vy = lcy + random.randint(-int(rows * 0.13), int(rows * 0.13))
                vx = cx_l + random.randint(-int(cols * 0.08), int(cols * 0.08))
                paint(image, ellipse(sh, (vy, vx), (3, 3)) & mask_l,
                      HU['lung_vessel'], noise_std=25)
        heart_t = max(0.0, (t - 0.12) / 0.88)
        if t > 0.12:
            h_ry = int(rows * 0.115 * min(heart_t * 2, 1.0))
            h_rx = int(cols * 0.095 * min(heart_t * 2, 1.0))
            h_cy = cy + int(rows * 0.025)
            h_cx = cx - int(cols * 0.025)
            if h_ry > 5 and h_rx > 5:
                paint(image, ellipse(sh, (h_cy, h_cx), (h_ry, h_rx)), HU['muscle'], noise_std=8)
                lv_cy = h_cy + int(rows * 0.015)
                lv_cx = h_cx - int(cols * 0.018)
                lv_ry, lv_rx = int(h_ry * 0.50), int(h_rx * 0.42)
                lv_in = (max(lv_ry - int(rows * 0.018), 3), max(lv_rx - int(cols * 0.014), 3))
                paint(image, ring(sh, (lv_cy, lv_cx), (lv_ry, lv_rx), lv_in),
                      HU['muscle'] + 10, noise_std=8)
                paint(image, ellipse(sh, (lv_cy, lv_cx), lv_in), HU['blood'], noise_std=6)
                rv_cx = h_cx + int(cols * 0.055)
                rv_ry, rv_rx = int(h_ry * 0.46), int(h_rx * 0.38)
                rv_mask = ellipse(sh, (h_cy - int(rows * 0.005), rv_cx), (rv_ry, rv_rx))
                rv_wall = rv_mask & ~ellipse(sh, (h_cy - int(rows * 0.005), rv_cx),
                                             (max(rv_ry - 6, 3), max(rv_rx - 5, 3)))
                paint(image, rv_wall, HU['muscle'] + 5, noise_std=8)
                paint(image, rv_mask & ~rv_wall, HU['blood'] - 5, noise_std=6)
                la_mask = ellipse(sh, (h_cy - int(rows * 0.04), h_cx - int(cols * 0.01)),
                                  (int(h_ry * 0.35), int(h_rx * 0.38)))
                paint(image, la_mask, HU['blood'] - 5, noise_std=6)
        paint(image, ellipse(sh, (cy - int(rows * 0.01), cx),
                             (int(rows * 0.025), int(cols * 0.013))),
              HU['bone_cancel'] + 80, noise_std=40)
        for ang in [0.20, 0.35, 0.50, 0.65, 0.80]:
            theta = np.pi * ang
            for side in (-1, 1):
                rib_cy = int(cy + bry * 0.62 * np.sin(theta - 0.35))
                rib_cx = int(cx + side * brx * 0.82 * np.cos(theta * 0.6))
                paint(image, ellipse(sh, (rib_cy, rib_cx), (7, 10)),
                      HU['bone_cortex'] - 100, noise_std=50)
                paint(image, ellipse(sh, (rib_cy, rib_cx), (4, 6)),
                      HU['bone_cancel'] - 50, noise_std=30)
        if pos < 0.22:
            t_mask = ellipse(sh, (cy - int(rows * 0.04), cx),
                             (int(rows * 0.028), int(cols * 0.018)))
            paint(image, t_mask, HU['air'] + 50, noise_std=20)
            paint(image, ring(sh, (cy - int(rows * 0.04), cx),
                              (int(rows * 0.028), int(cols * 0.018)),
                              (int(rows * 0.020), int(cols * 0.012))),
                  HU['soft_tissue'], noise_std=10)
        elif pos < 0.32:
            for side in (-1, 1):
                paint(image, ellipse(sh, (cy - int(rows * 0.04),
                                         cx + side * int(cols * 0.04)),
                                     (int(rows * 0.018), int(cols * 0.013))),
                      HU['air'] + 50, noise_std=20)
        paint(image, ellipse(sh, (sp_cy - sp_r - 10, cx + 8), (6, 5)),
              HU['air'] + 200, noise_std=80)

    elif pos < 0.63:
        t = (pos - 0.48) / 0.15
        if t < 0.5:
            base_size = int(rows * 0.07 * (1 - t * 2))
            for side in (-1, 1):
                paint(image, ellipse(sh, (cy - int(rows * 0.11),
                                         cx + side * int(cols * 0.14)),
                                     (base_size, int(cols * 0.08))),
                      HU['lung'], noise_std=40)
        liv_cy = cy + int(rows * 0.015)
        liv_cx = cx + int(cols * 0.10)
        l_ry   = int(rows * (0.18 - 0.02 * t))
        l_rx   = int(cols * (0.16 - 0.02 * t))
        liver_mask = ellipse(sh, (liv_cy, liv_cx), (l_ry, l_rx))
        paint(image, liver_mask, HU['liver'], noise_std=8)
        for _ in range(8):
            hvy = liv_cy + random.randint(-int(rows * 0.10), int(rows * 0.10))
            hvx = liv_cx + random.randint(-int(cols * 0.09), int(cols * 0.09))
            paint(image, ellipse(sh, (hvy, hvx), (4, 6)) & liver_mask,
                  HU['blood'] - 10, noise_std=5)
        gb_cy = liv_cy + int(rows * 0.09)
        gb_cx = liv_cx - int(cols * 0.06)
        paint(image, ellipse(sh, (gb_cy, gb_cx),
                             (int(rows * 0.038), int(cols * 0.028))),
              HU['gallbladder'], noise_std=5)
        paint(image, ring(sh, (gb_cy, gb_cx),
                          (int(rows * 0.038), int(cols * 0.028)),
                          (int(rows * 0.026), int(cols * 0.018))),
              HU['soft_tissue'] + 5, noise_std=6)
        spl_cy = cy - int(rows * 0.025)
        spl_cx = cx - int(cols * 0.135)
        paint(image, ellipse(sh, (spl_cy, spl_cx),
                             (int(rows * (0.095 - 0.01 * t)),
                              int(cols * (0.075 - 0.01 * t)))),
              HU['spleen'], noise_std=8)
        st_cy = cy - int(rows * 0.045)
        st_cx = cx - int(cols * 0.06)
        st_mask = ellipse(sh, (st_cy, st_cx),
                          (int(rows * 0.075), int(cols * 0.065)))
        paint(image, st_mask, HU['bowel_air'] + 50, noise_std=80)
        paint(image, ring(sh, (st_cy, st_cx),
                          (int(rows * 0.075), int(cols * 0.065)),
                          (int(rows * 0.058), int(cols * 0.048))),
              HU['soft_tissue'] + 5, noise_std=8)
        paint(image, ellipse(sh, (cy + int(rows * 0.025), cx - int(cols * 0.015)),
                             (int(rows * 0.030), int(cols * 0.090))),
              HU['pancreas'], noise_std=8)
        for side in (-1, 1):
            for i in range(3):
                rib_cy = cy + int(rows * (0.08 + i * 0.07))
                rib_cx = cx + side * int(cols * (0.22 - i * 0.01))
                paint(image, ellipse(sh, (rib_cy, rib_cx), (7, 9)),
                      HU['bone_cortex'] - 80, noise_std=50)
                paint(image, ellipse(sh, (rib_cy, rib_cx), (4, 5)),
                      HU['bone_cancel'] - 30, noise_std=30)

    elif pos < 0.78:
        t = (pos - 0.63) / 0.15
        for side in (-1, 1):
            k_cy = cy + int(rows * (0.045 + 0.01 * t)) - side * int(rows * 0.015)
            k_cx = cx + side * int(cols * 0.125)
            k_ry = int(rows * (0.075 - 0.005 * t))
            k_rx = int(cols * (0.048 - 0.003 * t))
            paint(image, ellipse(sh, (k_cy, k_cx), (k_ry, k_rx)),
                  HU['kidney_cortex'], noise_std=8)
            paint(image, ellipse(sh, (k_cy, k_cx),
                                 (int(k_ry * 0.68), int(k_rx * 0.68))),
                  HU['kidney_medull'], noise_std=8)
            paint(image, ellipse(sh, (k_cy, k_cx),
                                 (int(k_ry * 0.32), int(k_rx * 0.32))),
                  HU['fat'] + 20, noise_std=10)
        for side in (-1, 1):
            paint(image, ellipse(sh, (cy + int(rows * 0.075),
                                      cx + side * int(cols * 0.082)),
                                 (int(rows * 0.065), int(cols * 0.040))),
                  HU['muscle'], noise_std=10)
        for side in (-1, 1):
            col_cx  = cx + side * int(cols * 0.155)
            col_cy  = cy + int(rows * 0.02)
            col_mask = ellipse(sh, (col_cy, col_cx),
                               (int(rows * 0.055), int(cols * 0.042)))
            paint(image, col_mask,
                  HU['bowel_air'] + random.randint(50, 200), noise_std=60)
            paint(image, ring(sh, (col_cy, col_cx),
                              (int(rows * 0.055), int(cols * 0.042)),
                              (int(rows * 0.038), int(cols * 0.028))),
                  HU['soft_tissue'], noise_std=8)
        for _ in range(10):
            by   = cy + random.randint(-int(rows * 0.07), int(rows * 0.06))
            bx   = cx + random.randint(-int(cols * 0.09), int(cols * 0.09))
            b_ry = random.randint(12, 22)
            b_rx = random.randint(10, 18)
            hu_c = random.choice([HU['bowel_air'], HU['bowel_air'] + 200,
                                   HU['bowel_fluid'], HU['bowel_fluid'] + 10])
            b_mask = ellipse(sh, (by, bx), (b_ry, b_rx))
            paint(image, b_mask, hu_c, noise_std=25)
            paint(image, ring(sh, (by, bx), (b_ry, b_rx),
                              (max(b_ry - 5, 3), max(b_rx - 4, 3))),
                  HU['soft_tissue'], noise_std=8)
        for side in (-1, 1):
            paint(image, ellipse(sh, (cy - int(rows * 0.04),
                                      cx + side * int(cols * 0.055)),
                                 (int(rows * 0.065), int(cols * 0.038))),
                  HU['muscle'], noise_std=10)

    else:
        t = (pos - 0.78) / 0.22
        for side in (-1, 1):
            il_cx = cx + side * int(cols * (0.155 + 0.02 * t))
            il_cy = cy + int(rows * 0.04)
            il_ry = int(rows * (0.11 + 0.02 * t))
            il_rx = int(cols * (0.10 + 0.02 * t))
            paint(image, ellipse(sh, (il_cy, il_cx), (il_ry, il_rx)),
                  HU['bone_cancel'] + 50, noise_std=40)
            paint(image, ring(sh, (il_cy, il_cx), (il_ry, il_rx),
                              (il_ry - int(rows * 0.015), il_rx - int(cols * 0.013))),
                  HU['bone_cortex'], noise_std=40)
        bl_ry = int(rows * (0.10 - 0.05 * t))
        bl_rx = int(cols * (0.09 - 0.04 * t))
        if bl_ry > 10:
            bl_cy  = cy - int(rows * (0.02 + 0.03 * t))
            bl_mask = ellipse(sh, (bl_cy, cx), (bl_ry, bl_rx))
            paint(image, bl_mask, HU['bladder'], noise_std=4)
            paint(image, ring(sh, (bl_cy, cx), (bl_ry, bl_rx),
                              (bl_ry - int(rows * 0.012), bl_rx - int(cols * 0.010))),
                  HU['soft_tissue'] + 10, noise_std=6)
        rect_cy  = cy + int(rows * 0.055)
        rect_mask = ellipse(sh, (rect_cy, cx),
                            (int(rows * 0.042), int(cols * 0.042)))
        paint(image, rect_mask,
              HU['bowel_air'] + random.randint(100, 300), noise_std=60)
        paint(image, ring(sh, (rect_cy, cx),
                          (int(rows * 0.042), int(cols * 0.042)),
                          (int(rows * 0.026), int(cols * 0.026))),
              HU['soft_tissue'], noise_std=8)
        for side in (-1, 1):
            paint(image, ellipse(sh, (cy + int(rows * 0.025),
                                      cx + side * int(cols * 0.090)),
                                 (int(rows * 0.058), int(cols * 0.042))),
                  HU['muscle'], noise_std=10)
        if t > 0.55:
            fh_t = (t - 0.55) / 0.45
            fh_r = int(rows * 0.065 * fh_t)
            if fh_r > 8:
                for side in (-1, 1):
                    fh_cx = cx + side * int(cols * (0.185 + 0.01 * fh_t))
                    fh_cy = cy + int(rows * 0.06)
                    paint(image, ellipse(sh, (fh_cy, fh_cx), (fh_r, fh_r)),
                          HU['bone_cancel'] + 80, noise_std=40)
                    paint(image, ring(sh, (fh_cy, fh_cx), (fh_r, fh_r),
                                      (fh_r - int(rows * 0.010),
                                       fh_r - int(rows * 0.010))),
                          HU['bone_cortex'], noise_std=40)
        if t > 0.30:
            paint(image, ellipse(sh, (cy - int(rows * 0.005), cx),
                                 (int(rows * 0.028), int(cols * 0.048))),
                  HU['bone_cancel'] + 100, noise_std=40)

    image[~body_mask] = HU['air']
    image = np.clip(image, -1024, 3071)
    image = gaussian_filter(image, sigma=1.0)
    image[~body_mask] = HU['air']
    return image.astype(np.int16)


# ──────────────────────────────────────────────────────────────────────────────
# MR pixel generator – SAGITTAL T1-weighted
# ──────────────────────────────────────────────────────────────────────────────

def generate_mr_slice(slice_idx: int, num_slices: int,
                      rows: int = 512, cols: int = 512) -> np.ndarray:
    image = np.zeros((rows, cols), dtype=np.float32)
    sh    = (rows, cols)

    mid_idx    = (num_slices - 1) / 2.0
    centrality = 1.0 - abs(slice_idx - mid_idx) / max(mid_idx, 1)

    body_cy = rows // 2
    body_cx = cols // 2
    body_ry = int(rows * 0.460)
    body_rx = int(cols * (0.26 * centrality + 0.055))

    body_mask = ellipse(sh, (body_cy, body_cx), (body_ry, body_rx))
    paint(image, body_mask, MR_SIG['soft_tissue'], noise_std=22)

    fat_shell = body_mask & ~ellipse(sh, (body_cy, body_cx), (body_ry - 14, body_rx - 10))
    paint(image, fat_shell, MR_SIG['fat'], noise_std=38)

    paint(image,
          ellipse(sh, (int(rows * 0.58), int(cols * 0.29)),
                  (int(rows * 0.18), int(cols * 0.038))) & body_mask,
          MR_SIG['muscle'], noise_std=22)

    if centrality > 0.45:
        trachea_mask = ellipse(sh, (int(rows * 0.185), int(cols * 0.375)),
                               (int(rows * 0.078), int(cols * 0.024))) & body_mask
        paint(image, trachea_mask, MR_SIG['air'], noise_std=5)

    if centrality > 0.30:
        h_cy   = int(rows * 0.335)
        h_cx   = int(cols * 0.370)
        h_mask = ellipse(sh, (h_cy, h_cx),
                         (int(rows * 0.072), int(cols * 0.068))) & body_mask
        paint(image, h_mask, MR_SIG['blood'], noise_std=24)
        myo = ring(sh, (h_cy, h_cx),
                   (int(rows * 0.072), int(cols * 0.068)),
                   (int(rows * 0.050), int(cols * 0.046))) & body_mask
        paint(image, myo, MR_SIG['muscle'] + 28, noise_std=18)

    lung_mask = ellipse(sh, (int(rows * 0.318), int(cols * 0.425)),
                        (int(rows * 0.095), int(cols * 0.060))) & body_mask
    paint(image, lung_mask, MR_SIG['lung'], noise_std=18)

    if centrality > 0.22:
        liv_cy     = int(rows * 0.530)
        liv_cx     = int(cols * 0.385)
        liver_mask = ellipse(sh, (liv_cy, liv_cx),
                             (int(rows * 0.080), int(cols * 0.092))) & body_mask
        paint(image, liver_mask, MR_SIG['liver'], noise_std=22)
        for _ in range(6):
            hvy = liv_cy + random.randint(-int(rows * 0.05), int(rows * 0.05))
            hvx = liv_cx + random.randint(-int(cols * 0.05), int(cols * 0.05))
            paint(image, ellipse(sh, (hvy, hvx), (4, 5)) & liver_mask,
                  MR_SIG['blood'] - 28, noise_std=12)

    for _ in range(7):
        by    = int(rows * 0.63) + random.randint(-int(rows * 0.06), int(rows * 0.06))
        bx    = int(cols * 0.35) + random.randint(-int(cols * 0.05), int(cols * 0.05))
        b_sig = random.choice([MR_SIG['soft_tissue'], MR_SIG['fluid_csf'] + 60,
                                MR_SIG['fat'] - 200, MR_SIG['muscle']])
        b_mask = ellipse(sh, (by, bx),
                         (random.randint(10, 22), random.randint(8, 18))) & body_mask
        paint(image, b_mask, b_sig, noise_std=28)

    if centrality > 0.28:
        bl_cy   = int(rows * 0.818)
        bl_cx   = int(cols * 0.358)
        bl_mask = ellipse(sh, (bl_cy, bl_cx),
                          (int(rows * 0.038), int(cols * 0.046))) & body_mask
        paint(image, bl_mask, MR_SIG['bladder'], noise_std=8)
        bl_wall = ring(sh, (bl_cy, bl_cx),
                       (int(rows * 0.038), int(cols * 0.046)),
                       (int(rows * 0.026), int(cols * 0.030))) & body_mask
        paint(image, bl_wall, MR_SIG['soft_tissue'] + 32, noise_std=12)

    spine_cx = int(cols * (0.720 - 0.042 * (1 - centrality)))
    for pm_dx in (-int(cols * 0.060), int(cols * 0.056)):
        pm_mask = ellipse(sh, (int(rows * 0.500), spine_cx + pm_dx),
                          (int(rows * 0.375), int(cols * 0.038))) & body_mask
        paint(image, pm_mask, MR_SIG['muscle'], noise_std=22)
    epid_mask = ellipse(sh, (int(rows * 0.500), spine_cx + int(cols * 0.028)),
                        (int(rows * 0.368), int(cols * 0.018))) & body_mask
    paint(image, epid_mask, MR_SIG['fat'] - 75, noise_std=32)

    if centrality > 0.15:
        vis   = min(1.0, (centrality - 0.15) / 0.38)
        zones = [
            ("C",  7, 0.072, 0.225, 0.031),
            ("T", 12, 0.225, 0.595, 0.037),
            ("L",  5, 0.595, 0.792, 0.050),
            ("S",  3, 0.792, 0.898, 0.054),
        ]
        for zone_lbl, n_v, r_start, r_end, vbody_rx_frac in zones:
            row_start = int(rows * r_start)
            row_end   = int(rows * r_end)
            zone_h    = row_end - row_start
            vert_h    = zone_h // n_v
            disc_h    = max(2, int(vert_h * 0.22))
            body_h    = vert_h - disc_h
            vbody_ry  = max(body_h // 2 - 1, 3)
            vbody_rx  = max(int(cols * vbody_rx_frac), 4)
            for i in range(n_v):
                v_row_top = row_start + i * vert_h
                v_body_cy = v_row_top + body_h // 2
                disc_cy   = v_row_top + body_h + disc_h // 2
                vb_mask   = ellipse(sh, (v_body_cy, spine_cx), (vbody_ry, vbody_rx))
                if vb_mask.any():
                    paint(image, vb_mask, int(MR_SIG['bone_marrow'] * vis), noise_std=45)
                for ep_dy in (-vbody_ry, vbody_ry):
                    ep_mask = ellipse(sh, (v_body_cy + ep_dy, spine_cx), (2, vbody_rx - 2))
                    paint(image, ep_mask, MR_SIG['bone_cortex'], noise_std=10)
                if i < n_v - 1:
                    disc_ry  = max(disc_h // 2, 2)
                    disc_rx  = int(vbody_rx * 0.88)
                    disc_mask = ellipse(sh, (disc_cy, spine_cx), (disc_ry, disc_rx))
                    if disc_mask.any():
                        paint(image, disc_mask,
                              int(MR_SIG['disc'] * (0.65 + 0.35 * vis)), noise_std=18)
                if zone_lbl != "S":
                    sp_cx  = spine_cx + vbody_rx + int(cols * 0.026)
                    sp_mask = ellipse(sh, (v_body_cy, sp_cx),
                                      (max(vbody_ry - 2, 2), int(cols * 0.019)))
                    paint(image, sp_mask & body_mask,
                          int(MR_SIG['bone_marrow'] * 0.72 * vis), noise_std=35)

        canal_cx  = spine_cx - int(cols * 0.032)
        canal_ry  = int(rows * 0.355)
        canal_rx  = int(cols * 0.017)
        canal_mask = ellipse(sh, (int(rows * 0.485), canal_cx),
                             (canal_ry, canal_rx)) & body_mask
        paint(image, canal_mask, MR_SIG['fluid_csf'], noise_std=12)
        cord_rx   = max(int(canal_rx * 0.44), 3)
        cord_ry   = int(canal_ry * 0.84)
        cord_mask = ellipse(sh, (int(rows * 0.485), canal_cx),
                            (cord_ry, cord_rx)) & body_mask
        paint(image, cord_mask, MR_SIG['cord'], noise_std=18)
        lig_mask  = ellipse(sh, (int(rows * 0.485), canal_cx + canal_rx + 4),
                            (int(rows * 0.340), 3)) & body_mask
        paint(image, lig_mask, MR_SIG['bone_cortex'] + 30, noise_std=12)
        sac_mask  = ellipse(sh, (int(rows * 0.868), spine_cx - int(cols * 0.008)),
                            (int(rows * 0.048), int(cols * 0.056))) & body_mask
        paint(image, sac_mask, int(MR_SIG['bone_marrow'] * 0.84 * vis), noise_std=40)
        coc_mask  = ellipse(sh, (int(rows * 0.928), spine_cx + int(cols * 0.010)),
                            (int(rows * 0.018), int(cols * 0.024))) & body_mask
        paint(image, coc_mask, int(MR_SIG['bone_marrow'] * 0.68 * vis), noise_std=35)

    image[~body_mask] = 0
    image = gaussian_filter(image, sigma=1.2)
    image[~body_mask] = 0
    image = np.clip(image, 0, 1023)
    return image.astype(np.int16)


# ──────────────────────────────────────────────────────────────────────────────
# Image-plane metadata per modality
# ──────────────────────────────────────────────────────────────────────────────
_PLANE_META = {
    "CT": {
        "orientation":   [1, 0, 0,  0, 1, 0],
        "pixel_spacing": [0.7422, 0.7422],
        "plane":         "AXIAL",
        "body_part":     "CHEST",
        "sop_class":     "1.2.840.10008.5.1.4.1.1.2",
        "pixel_rep":     1,
        "wc":            40,
        "ww":            400,
    },
    "MR": {
        "orientation":   [0, 1, 0,  0, 0, -1],
        "pixel_spacing": [0.9375, 0.9375],
        "plane":         "SAGITTAL",
        "body_part":     "SPINE",
        "sop_class":     "1.2.840.10008.5.1.4.1.1.4",
        "pixel_rep":     0,
        "wc":            512,
        "ww":            800,
    },
}

PIXEL_GENERATORS = {
    "CT": generate_ct_slice,
    "MR": generate_mr_slice,
}


# ──────────────────────────────────────────────────────────────────────────────
# DICOM file writer
# ──────────────────────────────────────────────────────────────────────────────

def write_dicom(filepath, pixel_data, modality,
                patient_id, patient_name,
                study_uid, series_uid,
                series_number, instance_number,
                slice_location, slice_thickness,
                study_date, study_time):

    pm      = _PLANE_META[modality]
    ps      = pm["pixel_spacing"]
    sop_uid = generate_uid()
    rows_px, cols_px = pixel_data.shape

    img_pos = (
        [-(cols_px * ps[1] / 2.0), -(rows_px * ps[0] / 2.0), slice_location]
        if modality == "CT" else
        [slice_location, -(cols_px * ps[1] / 2.0),  (rows_px * ps[0] / 2.0)]
    )

    meta = Dataset()
    meta.MediaStorageSOPClassUID    = pm["sop_class"]
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID          = ExplicitVRLittleEndian
    meta.ImplementationClassUID     = generate_uid()
    meta.ImplementationVersionName  = "BodyPhantomP2_1.0"

    ds = FileDataset(filepath, {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR   = False

    ds.PatientName      = patient_name
    ds.PatientID        = patient_id
    ds.PatientBirthDate = "19720315"
    ds.PatientSex       = "M"

    ds.StudyInstanceUID       = study_uid
    ds.StudyDate              = study_date
    ds.StudyTime              = study_time
    ds.StudyID                = "1"
    ds.AccessionNumber        = f"ACC{random.randint(10000, 99999)}"
    ds.StudyDescription       = (
        f"{modality} – {study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
    )
    ds.ReferringPhysicianName = "Phantom^Generator"

    ds.SeriesInstanceUID  = series_uid
    ds.SeriesNumber       = series_number
    ds.SeriesDate         = study_date
    ds.SeriesTime         = study_time
    ds.SeriesDescription  = (
        f"{modality} {pm['plane']}  {study_date[:4]}-{study_date[4:6]}-{study_date[6:]}  "
        f"Ser{series_number}"
    )
    ds.Modality           = modality
    ds.BodyPartExamined   = pm["body_part"]

    ds.FrameOfReferenceUID        = generate_uid()
    ds.PositionReferenceIndicator = ""
    ds.Manufacturer               = "BodyPhantom Inc."
    ds.ManufacturerModelName      = "AnatomoSim 5000"
    ds.SoftwareVersions           = "3.2.0"

    ds.SOPClassUID    = pm["sop_class"]
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = instance_number
    ds.ImageType      = ["ORIGINAL", "PRIMARY", pm["plane"]]
    ds.ContentDate    = study_date
    ds.ContentTime    = study_time

    ds.PixelSpacing            = ps
    ds.ImageOrientationPatient = pm["orientation"]
    ds.ImagePositionPatient    = img_pos
    ds.SliceThickness          = slice_thickness
    ds.SliceLocation           = slice_location

    ds.SamplesPerPixel           = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows                      = rows_px
    ds.Columns                   = cols_px
    ds.BitsAllocated             = 16
    ds.BitsStored                = 16
    ds.HighBit                   = 15
    ds.PixelRepresentation       = pm["pixel_rep"]
    ds.PixelData                 = pixel_data.tobytes()
    ds.WindowCenter              = pm["wc"]
    ds.WindowWidth               = pm["ww"]

    if modality == "CT":
        ds.RescaleIntercept  = 0
        ds.RescaleSlope      = 1
        ds.RescaleType       = "HU"
        ds.KVP               = 120
        ds.ExposureTime      = 750
        ds.XRayTubeCurrent   = 250
        ds.ConvolutionKernel = "B30f"
    else:
        ds.ScanningSequence  = "SE"
        ds.SequenceVariant   = "NONE"
        ds.ScanOptions       = ""
        ds.MRAcquisitionType = "2D"
        ds.RepetitionTime    = 550.0
        ds.EchoTime          = 14.0
        ds.FlipAngle         = 90
        ds.NumberOfAverages  = 2

    pydicom.dcmwrite(filepath, ds)


# ──────────────────────────────────────────────────────────────────────────────
# Per-session generator  (one study → N series, with pre-generated UIDs)
# ──────────────────────────────────────────────────────────────────────────────

def generate_session(
    modality:           str,
    study_date:         str,
    study_time:         str,
    n_series:           int,
    series_uids:        list,       # pre-generated SeriesInstanceUIDs
    global_series_start: int,
    total_series:       int,
    patient_id:         str,
    patient_name:       str,
    study_label:        str,        # e.g. "Study 1 (Pair A)"
    output_dir:         str,
    image_size:         int   = 512,
    slices_per_series:  int   = 12,
    slice_thickness:    float = 5.0,
):
    """
    Write `n_series` DICOM series for a single study using pre-generated
    SeriesInstanceUIDs so that paired studies can share the same series UIDs.
    """
    pm       = _PLANE_META[modality]
    date_fmt = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
    gen      = PIXEL_GENERATORS[modality]

    study_uid = generate_uid()

    print(f"\n  ── {study_label}  │  {date_fmt}  │  {modality}/{pm['plane']}"
          f"  │  {n_series} series ──")
    print(f"     StudyInstanceUID : {study_uid}")

    for s in range(n_series):
        global_idx = global_series_start + s
        series_uid = series_uids[s]
        series_num = random.randint(100, 999)

        folder = os.path.join(
            output_dir,
            f"Study_{study_label.replace(' ', '_').replace('(', '').replace(')', '')}"
            f"_Series_{s + 1:02d}_{modality}_{pm['plane']}"
            f"_{date_fmt}_S{series_num}"
        )
        os.makedirs(folder, exist_ok=True)

        print(f"\n  [{global_idx}/{total_series}]"
              f"  {modality}/{pm['plane']}"
              f"  │  {date_fmt}"
              f"  │  SerNum={series_num}"
              f"  │  SeriesUID=...{series_uid[-12:]}"
              f"  │  {slices_per_series} slices")

        for sl in range(slices_per_series):
            filepath  = os.path.join(folder, f"slice_{sl + 1:04d}.dcm")
            slice_loc = (
                round(sl * slice_thickness, 3)
                if modality == "CT"
                else round(-(slices_per_series - 1) * slice_thickness / 2.0
                            + sl * slice_thickness, 3)
            )
            pixels = gen(sl, slices_per_series, image_size, image_size)
            write_dicom(
                filepath        = filepath,
                pixel_data      = pixels,
                modality        = modality,
                patient_id      = patient_id,
                patient_name    = patient_name,
                study_uid       = study_uid,
                series_uid      = series_uid,
                series_number   = series_num,
                instance_number = sl + 1,
                slice_location  = slice_loc,
                slice_thickness = slice_thickness,
                study_date      = study_date,
                study_time      = study_time,
            )
            print(f"    slice {sl + 1:>2}/{slices_per_series}"
                  f"  →  {os.path.basename(filepath)}", end="\r")

        print(f"    ✓  {slices_per_series} slices saved  →  {folder}          ")


# ──────────────────────────────────────────────────────────────────────────────
# Interactive modality selection
# ──────────────────────────────────────────────────────────────────────────────

MODALITY_INFO = {
    "CT": {
        "label":       "Computed Tomography (CT)",
        "description": "Axial cross-sections │ Hounsfield Units │ signed 16-bit",
        "plane":       "AXIAL",
    },
    "MR": {
        "label":       "Magnetic Resonance (MR) – T1 weighted",
        "description": "Sagittal slices │ signal 0-1023 │ fat=bright / CSF=dark",
        "plane":       "SAGITTAL",
    },
}


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   DICOM Phantom Generator – 4 STUDIES / 1 PATIENT         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  Layout:")
    print("    • 1 patient  (same PatientID for all studies)")
    print("    • Pair A – Study 1 & 2:  3 series each, shared SeriesInstanceUIDs")
    print("    • Pair B – Study 3 & 4:  4 series each, shared SeriesInstanceUIDs")
    print("    • Each study has a DIFFERENT StudyInstanceUID and date")
    print()


def select_modality() -> str:
    numeric_map = {"1": "CT", "2": "MR"}

    while True:
        print("  ┌──────────────────────────────────────────────────────────┐")
        print("  │  Select imaging modality for ALL series in this study:  │")
        print("  ├──────────────────────────────────────────────────────────┤")
        for key, info in MODALITY_INFO.items():
            print(f"  │  [{key}]  {info['label']:<44}  │")
        print(f"  │       CT → {_PLANE_META['CT']['plane']:<49}  │")
        print(f"  │       MR → {_PLANE_META['MR']['plane']:<49}  │")
        print("  └──────────────────────────────────────────────────────────┘")

        try:
            raw = input("\n  ➜  Enter modality [CT / MR]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Interrupted. Exiting.")
            sys.exit(0)

        choice = numeric_map.get(raw, raw.upper())
        if choice in MODALITY_INFO:
            info = MODALITY_INFO[choice]
            print()
            print(f"  ✔  Modality selected : {info['label']}")
            print(f"     Plane            : {info['plane']}")
            print(f"     Pixel type       : {info['description']}")
            print()
            return choice

        print(f"\n  ✘  '{raw}' is not recognised. Please type  CT  or  MR.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def create_patient2_study(
    modality:          str,
    output_dir:        str   = "body_dicoms_patient2",
    image_size:        int   = 512,
    slices_per_series: int   = 12,
    slice_thickness:   float = 5.0,
):
    """
    Generate 4 studies for ONE patient in the chosen modality.

    Layout
    ──────
      Pair A  (Studies 1 & 2):
        - 3 series per study
        - SAME SeriesInstanceUIDs across both studies (shared)
        - DIFFERENT StudyInstanceUIDs and dates

      Pair B  (Studies 3 & 4):
        - 4 series per study
        - SAME SeriesInstanceUIDs across both studies (shared)
        - DIFFERENT StudyInstanceUIDs and dates

      All 4 studies share the SAME PatientID.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pre-generate shared SeriesInstanceUIDs for each pair
    pair_a_series_uids = [generate_uid() for _ in range(3)]   # 3 series shared
    pair_b_series_uids = [generate_uid() for _ in range(4)]   # 4 series shared

    # Map pair groups to their pre-generated series UIDs
    pair_series_uids = {
        "A": pair_a_series_uids,
        "B": pair_b_series_uids,
    }

    total_series = sum(n for _, _, n, _ in STUDY_SCHEDULE)
    pm           = _PLANE_META[modality]

    print(f"  Patient ID   : {PATIENT_ID}")
    print(f"  Patient Name : {PATIENT_NAME}")
    print(f"  Modality     : {modality}  /  {pm['plane']}")
    print(f"  Total series : {total_series}  (across 4 studies)")
    print(f"  Matrix       : {image_size} × {image_size}  │  "
          f"{slices_per_series} slices/series  │  {slice_thickness} mm thick")
    print(f"  Output dir   : {output_dir}/")
    print()
    print("  Pair A series UIDs (shared by Study 1 & 2):")
    for i, uid in enumerate(pair_a_series_uids):
        print(f"    Series {i + 1}: {uid}")
    print("  Pair B series UIDs (shared by Study 3 & 4):")
    for i, uid in enumerate(pair_b_series_uids):
        print(f"    Series {i + 1}: {uid}")
    print("─" * 64)

    global_idx = 1
    for study_num, (study_date, study_time, n_series, pair_group) in enumerate(STUDY_SCHEDULE, 1):
        study_label = f"Study {study_num} (Pair {pair_group})"
        series_uids = pair_series_uids[pair_group]

        generate_session(
            modality             = modality,
            study_date           = study_date,
            study_time           = study_time,
            n_series             = n_series,
            series_uids          = series_uids,
            global_series_start  = global_idx,
            total_series         = total_series,
            patient_id           = PATIENT_ID,
            patient_name         = PATIENT_NAME,
            study_label          = study_label,
            output_dir           = output_dir,
            image_size           = image_size,
            slices_per_series    = slices_per_series,
            slice_thickness      = slice_thickness,
        )
        global_idx += n_series

    print()
    print("═" * 64)
    print(f"  ✓  All {total_series} {modality} series written to '{output_dir}/'")
    print(f"     Patient: {PATIENT_NAME} ({PATIENT_ID})")
    print()
    print("  Study layout:")
    g = 1
    for study_num, (study_date, _, n, pair_group) in enumerate(STUDY_SCHEDULE, 1):
        d = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
        print(f"    Study {study_num} (Pair {pair_group})  │  {d}"
              f"  →  {n} series  (global #{g:02d}–{g + n - 1:02d})")
        g += n
    print()
    print("  Pair A: Studies 1 & 2 share 3 SeriesInstanceUIDs")
    print("  Pair B: Studies 3 & 4 share 4 SeriesInstanceUIDs")
    print("═" * 64)
    print()
    print("  Load the root folder in your viewer.")
    print("  You will see 1 patient with 4 studies (different dates).")
    print("  Studies 1 & 2 each have 3 series with the same SeriesInstanceUIDs.")
    print("  Studies 3 & 4 each have 4 series with the same SeriesInstanceUIDs.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_banner()
    chosen = select_modality()
    create_patient2_study(
        modality          = chosen,
        output_dir        = "body_dicoms_patient2",
        image_size        = 512,
        slices_per_series = 12,
        slice_thickness   = 5.0,
    )