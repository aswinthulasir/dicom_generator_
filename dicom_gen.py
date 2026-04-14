# """
# Anatomically Realistic Human Body DICOM Phantom Generator
# ==========================================================
# Generates axial CT/MR DICOM series with pixel data that resembles
# real human body cross-sections (chest, abdomen, pelvis).

# Usage:
#     pip install pydicom numpy scipy
#     python create_body_dicom.py

# Opens cleanly in: 3D Slicer, Horos, OsiriX, MicroDicom, RadiAnt
# """

# import numpy as np
# import pydicom
# from pydicom.dataset import Dataset, FileDataset
# from pydicom.uid import generate_uid, ExplicitVRLittleEndian
# from scipy.ndimage import gaussian_filter
# import datetime
# import os
# import random

# # ──────────────────────────────────────────────────────────────────────────────
# # Hounsfield Unit (HU) reference values for CT
# # ──────────────────────────────────────────────────────────────────────────────
# HU = dict(
#     air          = -1024,
#     lung         = -700,
#     lung_vessel  = -150,
#     fat          = -80,
#     water        = 0,
#     soft_tissue  = 40,
#     muscle       = 50,
#     liver        = 62,
#     spleen       = 50,
#     kidney_cortex= 40,
#     kidney_medull= 20,
#     pancreas     = 42,
#     gallbladder  = 12,
#     bladder      = 8,
#     blood        = 42,
#     bowel_air    = -400,
#     bowel_fluid  = 20,
#     bone_cancel  = 250,   # cancellous (spongy) bone
#     bone_cortex  = 750,   # cortical (dense) bone
#     aorta_lumen  = 42,
#     csf          = 10,
# )

# # MR T1 signal intensity reference (0–1024 scale)
# MR = dict(
#     air         = 0,
#     fat         = 900,
#     muscle      = 500,
#     soft_tissue = 450,
#     liver       = 520,
#     spleen      = 480,
#     kidney      = 460,
#     fluid       = 80,
#     blood       = 420,
#     bone_marrow = 800,
#     bone_cortex = 60,
#     lung        = 60,
#     bladder     = 70,
# )


# # ──────────────────────────────────────────────────────────────────────────────
# # Geometry helpers
# # ──────────────────────────────────────────────────────────────────────────────

# def ellipse(shape, center, axes):
#     """Boolean mask of a filled ellipse."""
#     rows, cols = shape
#     cy, cx = center
#     ry, rx = max(axes[0], 1), max(axes[1], 1)
#     y, x = np.ogrid[:rows, :cols]
#     return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1


# def ring(shape, center, outer_axes, inner_axes):
#     """Boolean mask of an elliptical ring (annulus)."""
#     return ellipse(shape, center, outer_axes) & ~ellipse(shape, center, inner_axes)


# def paint(image, mask, value, noise_std=5):
#     """Write value (+ optional Gaussian noise) into image where mask is True."""
#     if not mask.any():
#         return
#     n = np.random.normal(value, noise_std, image.shape)
#     image[mask] = n[mask]


# # ──────────────────────────────────────────────────────────────────────────────
# # Anatomical slice generator – CT
# # ──────────────────────────────────────────────────────────────────────────────

# def generate_ct_slice(slice_idx: int, num_slices: int, rows: int = 512, cols: int = 512) -> np.ndarray:
#     """
#     Return a (rows × cols) int16 array with realistic CT HU values for one
#     axial body slice. slice_idx=0 → upper chest; slice_idx=num_slices-1 → pelvis.
#     """
#     image = np.full((rows, cols), HU['air'], dtype=np.float32)
#     sh = (rows, cols)
#     pos = slice_idx / max(num_slices - 1, 1)   # 0 (chest apex) … 1 (pelvis)
#     cy, cx = rows // 2, cols // 2

#     # ── 1. Body contour (shape varies by region) ──────────────────────────────
#     if pos < 0.12:                          # lung apex – narrow
#         bry, brx = int(rows * 0.23), int(cols * 0.19)
#     elif pos < 0.48:                        # chest
#         bry, brx = int(rows * 0.30), int(cols * 0.25)
#     elif pos < 0.72:                        # abdomen – widest
#         bry, brx = int(rows * 0.33), int(cols * 0.28)
#     else:                                   # pelvis
#         bry, brx = int(rows * 0.31), int(cols * 0.27)

#     body_mask = ellipse(sh, (cy, cx), (bry, brx))

#     # Soft tissue baseline inside body
#     paint(image, body_mask, HU['soft_tissue'], noise_std=6)

#     # ── 2. Skin (outermost 4 px) ──────────────────────────────────────────────
#     skin = body_mask & ~ellipse(sh, (cy, cx), (bry - 4, brx - 4))
#     paint(image, skin, HU['soft_tissue'] + 10, noise_std=8)

#     # ── 3. Subcutaneous fat ───────────────────────────────────────────────────
#     fat_ry, fat_rx = bry - 5, brx - 5
#     fat_in_ry = fat_ry - int(rows * 0.022)
#     fat_in_rx = fat_rx - int(cols * 0.022)
#     fat_mask = ring(sh, (cy, cx), (fat_ry, fat_rx), (fat_in_ry, fat_in_rx))
#     paint(image, fat_mask, HU['fat'], noise_std=15)

#     # ── 4. Outer muscle wall ──────────────────────────────────────────────────
#     msc_out_ry, msc_out_rx = fat_in_ry, fat_in_rx
#     msc_in_ry  = msc_out_ry - int(rows * 0.030)
#     msc_in_rx  = msc_out_rx - int(cols * 0.025)
#     muscle_wall = ring(sh, (cy, cx), (msc_out_ry, msc_out_rx), (msc_in_ry, msc_in_rx))
#     paint(image, muscle_wall, HU['muscle'], noise_std=10)

#     # Inner cavity baseline (fat/connective tissue)
#     inner = ellipse(sh, (cy, cx), (msc_in_ry, msc_in_rx))
#     paint(image, inner, HU['fat'] + 30, noise_std=8)

#     # ── 5. Spine ─────────────────────────────────────────────────────────────
#     sp_cy = cy + int(bry * 0.62)
#     sp_cx = cx
#     sp_r  = int(rows * 0.042)

#     # Vertebral body (cancellous bone)
#     vert = ellipse(sh, (sp_cy, sp_cx), (sp_r, sp_r))
#     paint(image, vert, HU['bone_cancel'], noise_std=35)

#     # Cortical shell
#     paint(image, ring(sh, (sp_cy, sp_cx), (sp_r, sp_r), (sp_r - 4, sp_r - 4)),
#           HU['bone_cortex'], noise_std=40)

#     # Spinal canal (CSF)
#     canal_r = max(int(sp_r * 0.38), 4)
#     paint(image, ellipse(sh, (sp_cy, sp_cx), (canal_r, canal_r)), HU['csf'], noise_std=4)

#     # Spinous process (posterior)
#     sp_proc_cy = sp_cy + sp_r + 6
#     paint(image, ellipse(sh, (sp_proc_cy, sp_cx), (9, 6)), HU['bone_cortex'] - 100, noise_std=40)

#     # Transverse processes
#     for side in (-1, 1):
#         tp_cx = sp_cx + side * (sp_r + 14)
#         paint(image, ellipse(sh, (sp_cy, tp_cx), (5, 9)), HU['bone_cortex'] - 100, noise_std=40)

#     # Paraspinal (erector spinae) muscles
#     for side in (-1, 1):
#         pm_cx = sp_cx + side * int(cols * 0.075)
#         paint(image, ellipse(sh, (sp_cy - 4, pm_cx), (int(rows * 0.058), int(cols * 0.042))),
#               HU['muscle'], noise_std=10)

#     # ── 6. Major vessels ──────────────────────────────────────────────────────
#     ao_cy = sp_cy - sp_r - 13
#     ao_cx = cx - int(cols * 0.014)
#     ao_r  = int(rows * 0.020)

#     # Aorta lumen
#     paint(image, ellipse(sh, (ao_cy, ao_cx), (ao_r, ao_r)), HU['aorta_lumen'], noise_std=6)
#     # Aortic wall
#     paint(image, ring(sh, (ao_cy, ao_cx), (ao_r, ao_r), (ao_r - 3, ao_r - 3)),
#           HU['muscle'] + 20, noise_std=8)

#     # IVC (slightly right, elliptical)
#     ivc_cx = cx + int(cols * 0.022)
#     paint(image, ellipse(sh, (ao_cy, ivc_cx), (int(rows * 0.016), int(cols * 0.020))),
#           HU['blood'] - 5, noise_std=6)

#     # ══════════════════════════════════════════════════════════════════════════
#     # Region-specific anatomy
#     # ══════════════════════════════════════════════════════════════════════════

#     # ── LUNG APEX  (pos 0.00 – 0.12) ─────────────────────────────────────────
#     if pos < 0.12:
#         t = pos / 0.12    # 0→1 across this zone
#         l_ry = int(rows * (0.08 + 0.10 * t))
#         l_rx = int(cols * (0.07 + 0.08 * t))
#         for side in (-1, 1):
#             lcx = cx + side * int(cols * 0.11)
#             paint(image, ellipse(sh, (cy - int(rows * 0.02), lcx), (l_ry, l_rx)),
#                   HU['lung'], noise_std=30)
#         # Clavicle cross-sections
#         for side in (-1, 1):
#             ccx = cx + side * int(cols * 0.14)
#             ccy = cy - int(rows * 0.05)
#             paint(image, ellipse(sh, (ccy, ccx), (7, 20)), HU['bone_cortex'] - 50, noise_std=40)

#     # ── CHEST  (pos 0.12 – 0.48) ─────────────────────────────────────────────
#     elif pos < 0.48:
#         # Lung volumes grow then slightly taper toward diaphragm
#         t = (pos - 0.12) / 0.36    # 0→1 across chest
#         lung_ry = int(rows * (0.17 + 0.03 * min(t * 2, 1.0) - 0.02 * max(t - 0.5, 0)))
#         lung_rx_l = int(cols * (0.10 + 0.03 * min(t * 2, 1.0)))
#         lung_rx_r = int(cols * (0.11 + 0.03 * min(t * 2, 1.0)))
#         lcx = cx - int(cols * 0.13)   # left lung centre
#         rcx = cx + int(cols * 0.14)   # right lung centre
#         lcy = cy - int(rows * 0.02)

#         # Left lung
#         l_lung = ellipse(sh, (lcy, lcx), (lung_ry, lung_rx_l))
#         paint(image, l_lung, HU['lung'], noise_std=35)

#         # Right lung (slightly larger)
#         r_lung = ellipse(sh, (lcy, rcx), (lung_ry, lung_rx_r))
#         paint(image, r_lung, HU['lung'], noise_std=35)

#         # Pulmonary vessels (bright dots scattered inside lungs)
#         for cx_lung, mask_lung in ((lcx, l_lung), (rcx, r_lung)):
#             for _ in range(18):
#                 vy = lcy + random.randint(-int(rows * 0.13), int(rows * 0.13))
#                 vx = cx_lung + random.randint(-int(cols * 0.08), int(cols * 0.08))
#                 v_mask = ellipse(sh, (vy, vx), (3, 3)) & mask_lung
#                 paint(image, v_mask, HU['lung_vessel'], noise_std=25)

#         # Heart (present from ~20% into chest region)
#         heart_t = max(0.0, (t - 0.12) / 0.88)
#         if t > 0.12:
#             h_ry = int(rows * 0.115 * min(heart_t * 2, 1.0))
#             h_rx = int(cols * 0.095 * min(heart_t * 2, 1.0))
#             h_cy = cy + int(rows * 0.025)
#             h_cx = cx - int(cols * 0.025)
#             if h_ry > 5 and h_rx > 5:
#                 # Pericardium + myocardium background
#                 heart_mask = ellipse(sh, (h_cy, h_cx), (h_ry, h_rx))
#                 paint(image, heart_mask, HU['muscle'], noise_std=8)

#                 # Left ventricle (muscular, thick wall)
#                 lv_cy = h_cy + int(rows * 0.015)
#                 lv_cx = h_cx - int(cols * 0.018)
#                 lv_ry, lv_rx = int(h_ry * 0.50), int(h_rx * 0.42)
#                 lv_wall = ring(sh, (lv_cy, lv_cx), (lv_ry, lv_rx),
#                                (max(lv_ry - int(rows * 0.018), 3), max(lv_rx - int(cols * 0.014), 3)))
#                 lv_lumen = ellipse(sh, (lv_cy, lv_cx),
#                                    (max(lv_ry - int(rows * 0.018), 3), max(lv_rx - int(cols * 0.014), 3)))
#                 paint(image, lv_wall,  HU['muscle'] + 10, noise_std=8)
#                 paint(image, lv_lumen, HU['blood'], noise_std=6)

#                 # Right ventricle (thin wall, crescent shape)
#                 rv_cx = h_cx + int(cols * 0.055)
#                 rv_ry, rv_rx = int(h_ry * 0.46), int(h_rx * 0.38)
#                 rv_mask = ellipse(sh, (h_cy - int(rows * 0.005), rv_cx), (rv_ry, rv_rx))
#                 rv_wall = rv_mask & ~ellipse(sh, (h_cy - int(rows * 0.005), rv_cx),
#                                              (max(rv_ry - 6, 3), max(rv_rx - 5, 3)))
#                 rv_lumen = rv_mask & ~rv_wall
#                 paint(image, rv_wall,  HU['muscle'] + 5, noise_std=8)
#                 paint(image, rv_lumen, HU['blood'] - 5, noise_std=6)

#                 # Left atrium (posterior)
#                 la_cy = h_cy - int(rows * 0.04)
#                 la_mask = ellipse(sh, (la_cy, h_cx - int(cols * 0.01)),
#                                   (int(h_ry * 0.35), int(h_rx * 0.38)))
#                 paint(image, la_mask, HU['blood'] - 5, noise_std=6)

#         # Sternum (anterior midline)
#         st_cy = cy - int(rows * 0.01)
#         paint(image, ellipse(sh, (st_cy, cx), (int(rows * 0.025), int(cols * 0.013))),
#               HU['bone_cancel'] + 80, noise_std=40)

#         # Ribs (5 pairs, posterolateral, oval cross-sections)
#         rib_angles = [0.20, 0.35, 0.50, 0.65, 0.80]
#         for ang in rib_angles:
#             theta = np.pi * ang
#             for side in (-1, 1):
#                 rib_cy = int(cy + bry * 0.62 * np.sin(theta - 0.35))
#                 rib_cx = int(cx + side * brx * 0.82 * np.cos(theta * 0.6))
#                 rib_mask = ellipse(sh, (rib_cy, rib_cx), (7, 10))
#                 paint(image, rib_mask, HU['bone_cortex'] - 100, noise_std=50)
#                 # Rib medullary cavity
#                 paint(image, ellipse(sh, (rib_cy, rib_cx), (4, 6)),
#                       HU['bone_cancel'] - 50, noise_std=30)

#         # Trachea / main bronchi (air-filled, only upper chest)
#         if pos < 0.22:
#             t_mask = ellipse(sh, (cy - int(rows * 0.04), cx), (int(rows * 0.028), int(cols * 0.018)))
#             paint(image, t_mask, HU['air'] + 50, noise_std=20)
#             # Tracheal wall
#             t_wall = ring(sh, (cy - int(rows * 0.04), cx),
#                           (int(rows * 0.028), int(cols * 0.018)),
#                           (int(rows * 0.020), int(cols * 0.012)))
#             paint(image, t_wall, HU['soft_tissue'], noise_std=10)
#         elif pos < 0.32:
#             for side in (-1, 1):
#                 br_cx = cx + side * int(cols * 0.04)
#                 br_mask = ellipse(sh, (cy - int(rows * 0.04), br_cx),
#                                   (int(rows * 0.018), int(cols * 0.013)))
#                 paint(image, br_mask, HU['air'] + 50, noise_std=20)

#         # Esophagus (posterior mediastinum, slightly left of midline)
#         eso_mask = ellipse(sh, (sp_cy - sp_r - 10, cx + 8), (6, 5))
#         paint(image, eso_mask, HU['air'] + 200, noise_std=80)

#     # ── UPPER ABDOMEN  (pos 0.48 – 0.63) ─────────────────────────────────────
#     elif pos < 0.63:
#         t = (pos - 0.48) / 0.15   # 0→1

#         # Lower lung bases (disappear as we move down)
#         if t < 0.5:
#             base_size = int(rows * 0.07 * (1 - t * 2))
#             for side in (-1, 1):
#                 lcx = cx + side * int(cols * 0.14)
#                 paint(image, ellipse(sh, (cy - int(rows * 0.11), lcx), (base_size, int(cols * 0.08))),
#                       HU['lung'], noise_std=40)

#         # Liver (right side, large)
#         l_ry = int(rows * (0.18 - 0.02 * t))
#         l_rx = int(cols * (0.16 - 0.02 * t))
#         liv_cy = cy + int(rows * 0.015)
#         liv_cx = cx + int(cols * 0.10)
#         liver_mask = ellipse(sh, (liv_cy, liv_cx), (l_ry, l_rx))
#         paint(image, liver_mask, HU['liver'], noise_std=8)
#         # Hepatic vessels (darker branching tubes inside liver)
#         for _ in range(8):
#             hvy = liv_cy + random.randint(-int(rows * 0.10), int(rows * 0.10))
#             hvx = liv_cx + random.randint(-int(cols * 0.09), int(cols * 0.09))
#             hv_m = ellipse(sh, (hvy, hvx), (4, 6)) & liver_mask
#             paint(image, hv_m, HU['blood'] - 10, noise_std=5)

#         # Gallbladder (under liver)
#         gb_cy = liv_cy + int(rows * 0.09)
#         gb_cx = liv_cx - int(cols * 0.06)
#         gb_mask = ellipse(sh, (gb_cy, gb_cx), (int(rows * 0.038), int(cols * 0.028)))
#         paint(image, gb_mask, HU['gallbladder'], noise_std=5)
#         gb_wall = ring(sh, (gb_cy, gb_cx),
#                        (int(rows * 0.038), int(cols * 0.028)),
#                        (int(rows * 0.026), int(cols * 0.018)))
#         paint(image, gb_wall, HU['soft_tissue'] + 5, noise_std=6)

#         # Spleen (left side)
#         sp_ry = int(rows * (0.095 - 0.01 * t))
#         sp_rx = int(cols * (0.075 - 0.01 * t))
#         spl_cy = cy - int(rows * 0.025)
#         spl_cx = cx - int(cols * 0.135)
#         spleen_mask = ellipse(sh, (spl_cy, spl_cx), (sp_ry, sp_rx))
#         paint(image, spleen_mask, HU['spleen'], noise_std=8)

#         # Stomach (left, largely air-filled)
#         st_cy = cy - int(rows * 0.045)
#         st_cx = cx - int(cols * 0.06)
#         st_mask = ellipse(sh, (st_cy, st_cx), (int(rows * 0.075), int(cols * 0.065)))
#         paint(image, st_mask, HU['bowel_air'] + 50, noise_std=80)
#         st_wall = ring(sh, (st_cy, st_cx),
#                        (int(rows * 0.075), int(cols * 0.065)),
#                        (int(rows * 0.058), int(cols * 0.048)))
#         paint(image, st_wall, HU['soft_tissue'] + 5, noise_std=8)

#         # Pancreas (horizontal, behind stomach)
#         pan_mask = ellipse(sh, (cy + int(rows * 0.025), cx - int(cols * 0.015)),
#                            (int(rows * 0.030), int(cols * 0.090)))
#         paint(image, pan_mask, HU['pancreas'], noise_std=8)

#         # Lower ribs
#         for side in (-1, 1):
#             for i in range(3):
#                 rib_cy = cy + int(rows * (0.08 + i * 0.07))
#                 rib_cx = cx + side * int(cols * (0.22 - i * 0.01))
#                 paint(image, ellipse(sh, (rib_cy, rib_cx), (7, 9)), HU['bone_cortex'] - 80, noise_std=50)
#                 paint(image, ellipse(sh, (rib_cy, rib_cx), (4, 5)), HU['bone_cancel'] - 30, noise_std=30)

#     # ── MID ABDOMEN  (pos 0.63 – 0.78) ───────────────────────────────────────
#     elif pos < 0.78:
#         t = (pos - 0.63) / 0.15

#         # Kidneys (bilateral posterior)
#         for side in (-1, 1):
#             k_cy = cy + int(rows * (0.045 + 0.01 * t)) - side * int(rows * 0.015)
#             k_cx = cx + side * int(cols * 0.125)
#             k_ry = int(rows * (0.075 - 0.005 * t))
#             k_rx = int(cols * (0.048 - 0.003 * t))
#             # Renal capsule / cortex
#             paint(image, ellipse(sh, (k_cy, k_cx), (k_ry, k_rx)), HU['kidney_cortex'], noise_std=8)
#             # Medulla
#             paint(image, ellipse(sh, (k_cy, k_cx), (int(k_ry * 0.68), int(k_rx * 0.68))),
#                   HU['kidney_medull'], noise_std=8)
#             # Renal sinus / pelvis (fat + collecting system)
#             paint(image, ellipse(sh, (k_cy, k_cx), (int(k_ry * 0.32), int(k_rx * 0.32))),
#                   HU['fat'] + 20, noise_std=10)

#         # Psoas muscles (lateral to spine)
#         for side in (-1, 1):
#             ps_cx = cx + side * int(cols * 0.082)
#             ps_cy = cy + int(rows * 0.075)
#             paint(image, ellipse(sh, (ps_cy, ps_cx), (int(rows * 0.065), int(cols * 0.040))),
#                   HU['muscle'], noise_std=10)

#         # Ascending / descending colon (bilateral, air-containing)
#         for side in (-1, 1):
#             col_cx = cx + side * int(cols * 0.155)
#             col_cy = cy + int(rows * 0.02)
#             col_mask = ellipse(sh, (col_cy, col_cx), (int(rows * 0.055), int(cols * 0.042)))
#             paint(image, col_mask, HU['bowel_air'] + random.randint(50, 200), noise_std=60)
#             col_wall = ring(sh, (col_cy, col_cx),
#                             (int(rows * 0.055), int(cols * 0.042)),
#                             (int(rows * 0.038), int(cols * 0.028)))
#             paint(image, col_wall, HU['soft_tissue'], noise_std=8)

#         # Small bowel loops (scattered, fluid/air mix)
#         for _ in range(10):
#             by = cy + random.randint(-int(rows * 0.07), int(rows * 0.06))
#             bx = cx + random.randint(-int(cols * 0.09), int(cols * 0.09))
#             b_ry = random.randint(12, 22)
#             b_rx = random.randint(10, 18)
#             content_hu = random.choice([HU['bowel_air'], HU['bowel_air'] + 200,
#                                          HU['bowel_fluid'], HU['bowel_fluid'] + 10])
#             b_mask = ellipse(sh, (by, bx), (b_ry, b_rx))
#             paint(image, b_mask, content_hu, noise_std=25)
#             b_wall = ring(sh, (by, bx), (b_ry, b_rx), (max(b_ry - 5, 3), max(b_rx - 4, 3)))
#             paint(image, b_wall, HU['soft_tissue'], noise_std=8)

#         # Abdominal rectus muscles (bilateral anterior)
#         for side in (-1, 1):
#             rm_cx = cx + side * int(cols * 0.055)
#             rm_cy = cy - int(rows * 0.04)
#             paint(image, ellipse(sh, (rm_cy, rm_cx), (int(rows * 0.065), int(cols * 0.038))),
#                   HU['muscle'], noise_std=10)

#     # ── PELVIS  (pos 0.78 – 1.00) ────────────────────────────────────────────
#     else:
#         t = (pos - 0.78) / 0.22   # 0→1

#         # Iliac wings (large curved bone masses, bilateral)
#         for side in (-1, 1):
#             il_cx = cx + side * int(cols * (0.155 + 0.02 * t))
#             il_cy = cy + int(rows * 0.04)
#             il_ry = int(rows * (0.11 + 0.02 * t))
#             il_rx = int(cols * (0.10 + 0.02 * t))
#             paint(image, ellipse(sh, (il_cy, il_cx), (il_ry, il_rx)),
#                   HU['bone_cancel'] + 50, noise_std=40)
#             il_shell = ring(sh, (il_cy, il_cx), (il_ry, il_rx),
#                             (il_ry - int(rows * 0.015), il_rx - int(cols * 0.013)))
#             paint(image, il_shell, HU['bone_cortex'], noise_std=40)

#         # Bladder (central, water density – shrinks as we go lower)
#         bl_ry = int(rows * (0.10 - 0.05 * t))
#         bl_rx = int(cols * (0.09 - 0.04 * t))
#         if bl_ry > 10:
#             bl_cy = cy - int(rows * (0.02 + 0.03 * t))
#             bl_mask = ellipse(sh, (bl_cy, cx), (bl_ry, bl_rx))
#             paint(image, bl_mask, HU['bladder'], noise_std=4)
#             bl_wall = ring(sh, (bl_cy, cx), (bl_ry, bl_rx),
#                            (bl_ry - int(rows * 0.012), bl_rx - int(cols * 0.010)))
#             paint(image, bl_wall, HU['soft_tissue'] + 10, noise_std=6)

#         # Rectum / sigmoid (posterior, air/stool)
#         rect_cy = cy + int(rows * 0.055)
#         rect_mask = ellipse(sh, (rect_cy, cx), (int(rows * 0.042), int(cols * 0.042)))
#         paint(image, rect_mask, HU['bowel_air'] + random.randint(100, 300), noise_std=60)
#         rect_wall = ring(sh, (rect_cy, cx),
#                          (int(rows * 0.042), int(cols * 0.042)),
#                          (int(rows * 0.026), int(cols * 0.026)))
#         paint(image, rect_wall, HU['soft_tissue'], noise_std=8)

#         # Iliacus / psoas muscles
#         for side in (-1, 1):
#             il_m_cx = cx + side * int(cols * 0.090)
#             il_m_cy = cy + int(rows * 0.025)
#             paint(image, ellipse(sh, (il_m_cy, il_m_cx), (int(rows * 0.058), int(cols * 0.042))),
#                   HU['muscle'], noise_std=10)

#         # Femoral heads (appear in lower pelvis)
#         if t > 0.55:
#             fh_t = (t - 0.55) / 0.45
#             fh_r = int(rows * 0.065 * fh_t)
#             if fh_r > 8:
#                 for side in (-1, 1):
#                     fh_cx = cx + side * int(cols * (0.185 + 0.01 * fh_t))
#                     fh_cy = cy + int(rows * 0.06)
#                     paint(image, ellipse(sh, (fh_cy, fh_cx), (fh_r, fh_r)),
#                           HU['bone_cancel'] + 80, noise_std=40)
#                     fh_shell = ring(sh, (fh_cy, fh_cx), (fh_r, fh_r),
#                                     (fh_r - int(rows * 0.010), fh_r - int(rows * 0.010)))
#                     paint(image, fh_shell, HU['bone_cortex'], noise_std=40)

#         # Pubic symphysis
#         if t > 0.30:
#             paint(image, ellipse(sh, (cy - int(rows * 0.005), cx), (int(rows * 0.028), int(cols * 0.048))),
#                   HU['bone_cancel'] + 100, noise_std=40)

#     # ── Clip to realistic CT range + zero-out outside body ───────────────────
#     image[~body_mask] = HU['air']
#     image = np.clip(image, -1024, 3071)

#     # Partial volume effect (slight Gaussian blur)
#     image = gaussian_filter(image, sigma=1.0)

#     # Re-zero outside body after blur
#     image[~body_mask] = HU['air']

#     return image.astype(np.int16)


# # ──────────────────────────────────────────────────────────────────────────────
# # Anatomical slice generator – MR (T1-weighted)
# # ──────────────────────────────────────────────────────────────────────────────

# def generate_mr_slice(slice_idx: int, num_slices: int, rows: int = 512, cols: int = 512) -> np.ndarray:
#     """
#     Return MR T1-weighted equivalent of the CT slice,
#     rescaled to 0–1024 unsigned 16-bit range.
#     Mapping: fat→bright, fluid→dark, muscle→intermediate.
#     """
#     # Generate CT slice first (HU values)
#     ct = generate_ct_slice(slice_idx, num_slices, rows, cols).astype(np.float32)

#     # Piecewise linear HU→MR T1 signal mapping
#     # HU -1024 (air)  → 0
#     # HU  -200 (fat)  → 900
#     # HU     0 (fluid)→ 80
#     # HU    50 (tissue)→ 480
#     # HU   700 (cortex)→ 60
#     mr = np.zeros_like(ct)

#     # Air (HU < -500) → near zero
#     m = ct < -500
#     mr[m] = np.interp(ct[m], [-1024, -500], [0, 40])

#     # Lung (-500 to -100) → low signal
#     m = (ct >= -500) & (ct < -100)
#     mr[m] = np.interp(ct[m], [-500, -100], [40, 200])

#     # Fat (-100 to -50) → very bright
#     m = (ct >= -100) & (ct < -50)
#     mr[m] = np.interp(ct[m], [-100, -50], [820, 950])

#     # Fat/soft transition (-50 to 0)
#     m = (ct >= -50) & (ct < 0)
#     mr[m] = np.interp(ct[m], [-50, 0], [950, 200])

#     # Fluid / soft tissue (0 to 80) → intermediate-dark
#     m = (ct >= 0) & (ct < 80)
#     mr[m] = np.interp(ct[m], [0, 80], [80, 520])

#     # Soft tissue / muscle (80 to 200)
#     m = (ct >= 80) & (ct < 200)
#     mr[m] = np.interp(ct[m], [80, 200], [520, 560])

#     # Cancellous bone (200 to 400) → bright (marrow fat)
#     m = (ct >= 200) & (ct < 400)
#     mr[m] = np.interp(ct[m], [200, 400], [700, 750])

#     # Cortical bone (400+) → very dark
#     m = ct >= 400
#     mr[m] = np.interp(ct[m], [400, 3071], [120, 50])

#     # Add MR noise
#     noise = np.random.normal(0, 12, mr.shape)
#     mr += noise

#     # Keep background (outside body) black
#     body_mask = ct > -800
#     mr[~body_mask] = 0

#     mr = np.clip(mr, 0, 1023)
#     return mr.astype(np.int16)


# # ──────────────────────────────────────────────────────────────────────────────
# # DICOM file writer
# # ──────────────────────────────────────────────────────────────────────────────

# def write_dicom(filepath: str, pixel_data: np.ndarray, modality: str,
#                 patient_id: str, patient_name: str,
#                 study_uid: str, series_uid: str,
#                 series_number: int, instance_number: int,
#                 slice_location: float, slice_thickness: float,
#                 study_date: str, study_time: str):

#     sop_uid = generate_uid()
#     sop_class = {
#         "CT": "1.2.840.10008.5.1.4.1.1.2",   # CT Image Storage
#         "MR": "1.2.840.10008.5.1.4.1.1.4",   # MR Image Storage
#     }[modality]

#     rows, cols = pixel_data.shape
#     pixel_rep  = 1 if modality == "CT" else 0   # CT: signed; MR: unsigned

#     # File meta
#     meta = Dataset()
#     meta.MediaStorageSOPClassUID    = sop_class
#     meta.MediaStorageSOPInstanceUID = sop_uid
#     meta.TransferSyntaxUID          = ExplicitVRLittleEndian
#     meta.ImplementationClassUID     = generate_uid()
#     meta.ImplementationVersionName  = "BodyPhantom_2.0"

#     ds = FileDataset(filepath, {}, file_meta=meta, preamble=b"\x00" * 128)
#     ds.is_little_endian = True
#     ds.is_implicit_VR   = False

#     # Patient
#     ds.PatientName      = patient_name
#     ds.PatientID        = patient_id
#     ds.PatientBirthDate = "19720315"
#     ds.PatientSex       = "M"

#     # Study
#     ds.StudyInstanceUID       = study_uid
#     ds.StudyDate              = study_date
#     ds.StudyTime              = study_time
#     ds.StudyID                = "1"
#     ds.AccessionNumber        = f"ACC{random.randint(1000, 9999)}"
#     ds.StudyDescription       = "Body Phantom Study"
#     ds.ReferringPhysicianName = "Phantom^Generator"

#     # Series
#     ds.SeriesInstanceUID  = series_uid
#     ds.SeriesNumber       = series_number
#     ds.SeriesDate         = study_date
#     ds.SeriesTime         = study_time
#     ds.SeriesDescription  = f"{modality} Body Series {series_number}"
#     ds.Modality           = modality
#     ds.BodyPartExamined   = "CHEST"

#     # Frame of reference
#     ds.FrameOfReferenceUID          = generate_uid()
#     ds.PositionReferenceIndicator   = ""

#     # Equipment
#     ds.Manufacturer          = "BodyPhantom Inc."
#     ds.ManufacturerModelName = "AnatomoSim 5000"
#     ds.SoftwareVersions      = "2.0.0"

#     # SOP
#     ds.SOPClassUID    = sop_class
#     ds.SOPInstanceUID = sop_uid
#     ds.InstanceNumber = instance_number
#     ds.ImageType      = ["ORIGINAL", "PRIMARY", "AXIAL"]
#     ds.ContentDate    = study_date
#     ds.ContentTime    = study_time

#     # Image plane
#     ds.PixelSpacing           = [0.7422, 0.7422]      # ~0.74 mm typical CT
#     ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
#     ds.ImagePositionPatient   = [-(cols * 0.7422 / 2), -(rows * 0.7422 / 2), slice_location]
#     ds.SliceThickness         = slice_thickness
#     ds.SliceLocation          = slice_location

#     # Pixel data
#     ds.SamplesPerPixel          = 1
#     ds.PhotometricInterpretation = "MONOCHROME2"
#     ds.Rows                     = rows
#     ds.Columns                  = cols
#     ds.BitsAllocated            = 16
#     ds.BitsStored               = 16
#     ds.HighBit                  = 15
#     ds.PixelRepresentation      = pixel_rep
#     ds.PixelData                = pixel_data.tobytes()

#     # Modality-specific tags
#     if modality == "CT":
#         ds.RescaleIntercept  = -1024
#         ds.RescaleSlope      = 1
#         ds.RescaleType       = "HU"
#         ds.KVP               = 120
#         ds.ExposureTime      = 750
#         ds.XRayTubeCurrent   = 250
#         ds.ConvolutionKernel = "B30f"
#         ds.WindowCenter      = 40
#         ds.WindowWidth       = 400

#     if modality == "MR":
#         ds.ScanningSequence  = "SE"
#         ds.SequenceVariant   = "NONE"
#         ds.ScanOptions       = ""
#         ds.MRAcquisitionType = "2D"
#         ds.RepetitionTime    = 550.0
#         ds.EchoTime          = 14.0
#         ds.FlipAngle         = 90
#         ds.NumberOfAverages  = 2
#         ds.WindowCenter      = 512
#         ds.WindowWidth       = 800

#     pydicom.dcmwrite(filepath, ds)


# # ──────────────────────────────────────────────────────────────────────────────
# # Main entry point
# # ──────────────────────────────────────────────────────────────────────────────

# def create_body_dicom(
#     output_dir: str = "body_dicoms",
#     num_series: int = None,
#     slices_per_series: int = None,
#     image_size: int = 512,
# ):
#     """
#     Generate anatomically realistic DICOM series of a human body phantom.

#     Parameters
#     ----------
#     output_dir       : root output folder
#     num_series       : number of series (default random 3–4)
#     slices_per_series: slices per series (default random 10–15)
#     image_size       : pixel matrix size (256 or 512 recommended)
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     num_series = num_series or random.randint(3, 4)
#     patient_id  = f"PAT-{random.randint(100000, 999999)}"
#     patient_name = "Body^Phantom^01"
#     study_uid   = generate_uid()
#     now         = datetime.datetime.now()
#     study_date  = now.strftime("%Y%m%d")
#     study_time  = now.strftime("%H%M%S")
#     slice_thickness = 5.0   # mm

#     print("=" * 60)
#     print("  Anatomical DICOM Phantom Generator")
#     print("=" * 60)
#     print(f"  Patient ID   : {patient_id}")
#     print(f"  Study UID    : {study_uid}")
#     print(f"  Series count : {num_series}")
#     print(f"  Matrix size  : {image_size} × {image_size}")
#     print("=" * 60)

#     for series_idx in range(num_series):
#         series_uid    = generate_uid()
#         series_number = random.randint(100, 999)
#         num_slices    = slices_per_series or random.randint(10, 15)
#         modality      = random.choice(["CT", "MR"])

#         series_dir = os.path.join(output_dir, f"Series_{series_idx + 1:02d}_{modality}_S{series_number}")
#         os.makedirs(series_dir, exist_ok=True)

#         print(f"\n  [{series_idx + 1}/{num_series}] {modality}  |  "
#               f"SeriesNumber={series_number}  |  {num_slices} slices")

#         for sl in range(num_slices):
#             filepath = os.path.join(series_dir, f"slice_{sl + 1:04d}.dcm")
#             slice_loc = round(sl * slice_thickness, 3)

#             if modality == "CT":
#                 pixels = generate_ct_slice(sl, num_slices, image_size, image_size)
#             else:
#                 pixels = generate_mr_slice(sl, num_slices, image_size, image_size)

#             write_dicom(
#                 filepath      = filepath,
#                 pixel_data    = pixels,
#                 modality      = modality,
#                 patient_id    = patient_id,
#                 patient_name  = patient_name,
#                 study_uid     = study_uid,
#                 series_uid    = series_uid,
#                 series_number = series_number,
#                 instance_number = sl + 1,
#                 slice_location  = slice_loc,
#                 slice_thickness = slice_thickness,
#                 study_date    = study_date,
#                 study_time    = study_time,
#             )
#             print(f"    slice {sl + 1:>2}/{num_slices}  →  {filepath}", end="\r")

#         print(f"    ✓  {num_slices} slices saved  →  {series_dir}          ")

#     print(f"\n{'=' * 60}")
#     print(f"  ✓  Done!  All series saved under '{output_dir}/'")
#     print(f"{'=' * 60}")
#     print("\n  Open with:  3D Slicer · Horos · OsiriX · MicroDicom · RadiAnt")
#     print("  Load the root folder — viewer will auto-group series.\n")


# # ──────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     create_body_dicom(
#         output_dir        = "body_dicoms",
#         # num_series      = 4,     # uncomment to fix
#         # slices_per_series = 12,  # uncomment to fix
#         image_size        = 512,
#     )


"""
Anatomically Realistic Human Body DICOM Phantom Generator
==========================================================
Generates axial CT/MR DICOM series with pixel data that resembles
real human body cross-sections (chest, abdomen, pelvis).

Usage:
    pip install pydicom numpy scipy
    python create_body_dicom.py

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

# MR T1 signal intensity reference (0–1024 scale)
MR = dict(
    air         = 0,
    fat         = 900,
    muscle      = 500,
    soft_tissue = 450,
    liver       = 520,
    spleen      = 480,
    kidney      = 460,
    fluid       = 80,
    blood       = 420,
    bone_marrow = 800,
    bone_cortex = 60,
    lung        = 60,
    bladder     = 70,
)

# ──────────────────────────────────────────────────────────────────────────────
# Modality Selection (interactive prompt)
# ──────────────────────────────────────────────────────────────────────────────

VALID_MODALITIES = {"CT", "MR"}

MODALITY_INFO = {
    "CT": {
        "label":       "Computed Tomography (CT)",
        "description": "Hounsfield Unit values, signed 16-bit pixels, bone/soft-tissue/air contrast",
        "use_cases":   "Chest, abdomen, pelvis, trauma, bone detail",
    },
    "MR": {
        "label":       "Magnetic Resonance (MR) – T1 weighted",
        "description": "Signal intensity 0–1023, unsigned 16-bit pixels, fat=bright / fluid=dark",
        "use_cases":   "Soft-tissue detail, musculoskeletal, liver, pelvis",
    },
}


def print_banner():
    """Print a welcome banner."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      Anatomical DICOM Body Phantom Generator  v2.0      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def print_modality_menu():
    """Display the modality selection menu."""
    print("  Select the imaging modality for ALL series in this study:")
    print()
    for key, info in MODALITY_INFO.items():
        print(f"  [{key}]  {info['label']}")
        print(f"         {info['description']}")
        print(f"         Best for: {info['use_cases']}")
        print()


def select_modality() -> str:
    """
    Interactively prompt the user to choose CT or MR.
    Accepts: 'CT', 'MR', 'ct', 'mr', '1' (CT), '2' (MR).
    Loops until a valid selection is made.

    Returns
    -------
    str
        Uppercase modality string: 'CT' or 'MR'.
    """
    numeric_map = {"1": "CT", "2": "MR"}

    while True:
        print_modality_menu()
        try:
            raw = input("  ➜  Enter modality [CT / MR]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Interrupted. Exiting.")
            sys.exit(0)

        choice = numeric_map.get(raw, raw.upper())

        if choice in VALID_MODALITIES:
            info = MODALITY_INFO[choice]
            print()
            print(f"  ✔  Selected: {info['label']}")
            print(f"     All series will be generated as {choice} images.")
            print()
            return choice

        print(f"\n  ✘  '{raw}' is not a valid option. Please type CT or MR.\n")
        print("  ─" * 30)
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def ellipse(shape, center, axes):
    """Boolean mask of a filled ellipse."""
    rows, cols = shape
    cy, cx = center
    ry, rx = max(axes[0], 1), max(axes[1], 1)
    y, x = np.ogrid[:rows, :cols]
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1


def ring(shape, center, outer_axes, inner_axes):
    """Boolean mask of an elliptical ring (annulus)."""
    return ellipse(shape, center, outer_axes) & ~ellipse(shape, center, inner_axes)


def paint(image, mask, value, noise_std=5):
    """Write value (+ optional Gaussian noise) into image where mask is True."""
    if not mask.any():
        return
    n = np.random.normal(value, noise_std, image.shape)
    image[mask] = n[mask]


# ──────────────────────────────────────────────────────────────────────────────
# Anatomical slice generator – CT
# ──────────────────────────────────────────────────────────────────────────────

def generate_ct_slice(slice_idx: int, num_slices: int, rows: int = 512, cols: int = 512) -> np.ndarray:
    """
    Return a (rows × cols) int16 array with realistic CT HU values for one
    axial body slice. slice_idx=0 → upper chest; slice_idx=num_slices-1 → pelvis.
    """
    image = np.full((rows, cols), HU['air'], dtype=np.float32)
    sh = (rows, cols)
    pos = slice_idx / max(num_slices - 1, 1)
    cy, cx = rows // 2, cols // 2

    # ── 1. Body contour ───────────────────────────────────────────────────────
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

    # Skin
    skin = body_mask & ~ellipse(sh, (cy, cx), (bry - 4, brx - 4))
    paint(image, skin, HU['soft_tissue'] + 10, noise_std=8)

    # Subcutaneous fat
    fat_ry, fat_rx = bry - 5, brx - 5
    fat_in_ry = fat_ry - int(rows * 0.022)
    fat_in_rx = fat_rx - int(cols * 0.022)
    fat_mask = ring(sh, (cy, cx), (fat_ry, fat_rx), (fat_in_ry, fat_in_rx))
    paint(image, fat_mask, HU['fat'], noise_std=15)

    # Outer muscle wall
    msc_out_ry, msc_out_rx = fat_in_ry, fat_in_rx
    msc_in_ry = msc_out_ry - int(rows * 0.030)
    msc_in_rx = msc_out_rx - int(cols * 0.025)
    muscle_wall = ring(sh, (cy, cx), (msc_out_ry, msc_out_rx), (msc_in_ry, msc_in_rx))
    paint(image, muscle_wall, HU['muscle'], noise_std=10)

    inner = ellipse(sh, (cy, cx), (msc_in_ry, msc_in_rx))
    paint(image, inner, HU['fat'] + 30, noise_std=8)

    # ── 2. Spine ──────────────────────────────────────────────────────────────
    sp_cy = cy + int(bry * 0.62)
    sp_cx = cx
    sp_r  = int(rows * 0.042)

    paint(image, ellipse(sh, (sp_cy, sp_cx), (sp_r, sp_r)), HU['bone_cancel'], noise_std=35)
    paint(image, ring(sh, (sp_cy, sp_cx), (sp_r, sp_r), (sp_r - 4, sp_r - 4)),
          HU['bone_cortex'], noise_std=40)

    canal_r = max(int(sp_r * 0.38), 4)
    paint(image, ellipse(sh, (sp_cy, sp_cx), (canal_r, canal_r)), HU['csf'], noise_std=4)

    sp_proc_cy = sp_cy + sp_r + 6
    paint(image, ellipse(sh, (sp_proc_cy, sp_cx), (9, 6)), HU['bone_cortex'] - 100, noise_std=40)

    for side in (-1, 1):
        tp_cx = sp_cx + side * (sp_r + 14)
        paint(image, ellipse(sh, (sp_cy, tp_cx), (5, 9)), HU['bone_cortex'] - 100, noise_std=40)

    for side in (-1, 1):
        pm_cx = sp_cx + side * int(cols * 0.075)
        paint(image, ellipse(sh, (sp_cy - 4, pm_cx), (int(rows * 0.058), int(cols * 0.042))),
              HU['muscle'], noise_std=10)

    # ── 3. Major vessels ──────────────────────────────────────────────────────
    ao_cy = sp_cy - sp_r - 13
    ao_cx = cx - int(cols * 0.014)
    ao_r  = int(rows * 0.020)

    paint(image, ellipse(sh, (ao_cy, ao_cx), (ao_r, ao_r)), HU['aorta_lumen'], noise_std=6)
    paint(image, ring(sh, (ao_cy, ao_cx), (ao_r, ao_r), (ao_r - 3, ao_r - 3)),
          HU['muscle'] + 20, noise_std=8)

    ivc_cx = cx + int(cols * 0.022)
    paint(image, ellipse(sh, (ao_cy, ivc_cx), (int(rows * 0.016), int(cols * 0.020))),
          HU['blood'] - 5, noise_std=6)

    # ══════════════════════════════════════════════════════════════════════════
    # Region-specific anatomy
    # ══════════════════════════════════════════════════════════════════════════

    if pos < 0.12:
        t = pos / 0.12
        l_ry = int(rows * (0.08 + 0.10 * t))
        l_rx = int(cols * (0.07 + 0.08 * t))
        for side in (-1, 1):
            lcx = cx + side * int(cols * 0.11)
            paint(image, ellipse(sh, (cy - int(rows * 0.02), lcx), (l_ry, l_rx)),
                  HU['lung'], noise_std=30)
        for side in (-1, 1):
            ccx = cx + side * int(cols * 0.14)
            ccy = cy - int(rows * 0.05)
            paint(image, ellipse(sh, (ccy, ccx), (7, 20)), HU['bone_cortex'] - 50, noise_std=40)

    elif pos < 0.48:
        t = (pos - 0.12) / 0.36
        lung_ry   = int(rows * (0.17 + 0.03 * min(t * 2, 1.0) - 0.02 * max(t - 0.5, 0)))
        lung_rx_l = int(cols * (0.10 + 0.03 * min(t * 2, 1.0)))
        lung_rx_r = int(cols * (0.11 + 0.03 * min(t * 2, 1.0)))
        lcx = cx - int(cols * 0.13)
        rcx = cx + int(cols * 0.14)
        lcy = cy - int(rows * 0.02)

        l_lung = ellipse(sh, (lcy, lcx), (lung_ry, lung_rx_l))
        paint(image, l_lung, HU['lung'], noise_std=35)
        r_lung = ellipse(sh, (lcy, rcx), (lung_ry, lung_rx_r))
        paint(image, r_lung, HU['lung'], noise_std=35)

        for cx_lung, mask_lung in ((lcx, l_lung), (rcx, r_lung)):
            for _ in range(18):
                vy = lcy + random.randint(-int(rows * 0.13), int(rows * 0.13))
                vx = cx_lung + random.randint(-int(cols * 0.08), int(cols * 0.08))
                v_mask = ellipse(sh, (vy, vx), (3, 3)) & mask_lung
                paint(image, v_mask, HU['lung_vessel'], noise_std=25)

        heart_t = max(0.0, (t - 0.12) / 0.88)
        if t > 0.12:
            h_ry = int(rows * 0.115 * min(heart_t * 2, 1.0))
            h_rx = int(cols * 0.095 * min(heart_t * 2, 1.0))
            h_cy = cy + int(rows * 0.025)
            h_cx = cx - int(cols * 0.025)
            if h_ry > 5 and h_rx > 5:
                heart_mask = ellipse(sh, (h_cy, h_cx), (h_ry, h_rx))
                paint(image, heart_mask, HU['muscle'], noise_std=8)

                lv_cy = h_cy + int(rows * 0.015)
                lv_cx = h_cx - int(cols * 0.018)
                lv_ry, lv_rx = int(h_ry * 0.50), int(h_rx * 0.42)
                lv_wall = ring(sh, (lv_cy, lv_cx), (lv_ry, lv_rx),
                               (max(lv_ry - int(rows * 0.018), 3), max(lv_rx - int(cols * 0.014), 3)))
                lv_lumen = ellipse(sh, (lv_cy, lv_cx),
                                   (max(lv_ry - int(rows * 0.018), 3), max(lv_rx - int(cols * 0.014), 3)))
                paint(image, lv_wall,  HU['muscle'] + 10, noise_std=8)
                paint(image, lv_lumen, HU['blood'], noise_std=6)

                rv_cx = h_cx + int(cols * 0.055)
                rv_ry, rv_rx = int(h_ry * 0.46), int(h_rx * 0.38)
                rv_mask = ellipse(sh, (h_cy - int(rows * 0.005), rv_cx), (rv_ry, rv_rx))
                rv_wall  = rv_mask & ~ellipse(sh, (h_cy - int(rows * 0.005), rv_cx),
                                              (max(rv_ry - 6, 3), max(rv_rx - 5, 3)))
                rv_lumen = rv_mask & ~rv_wall
                paint(image, rv_wall,  HU['muscle'] + 5, noise_std=8)
                paint(image, rv_lumen, HU['blood'] - 5, noise_std=6)

                la_cy = h_cy - int(rows * 0.04)
                la_mask = ellipse(sh, (la_cy, h_cx - int(cols * 0.01)),
                                  (int(h_ry * 0.35), int(h_rx * 0.38)))
                paint(image, la_mask, HU['blood'] - 5, noise_std=6)

        st_cy = cy - int(rows * 0.01)
        paint(image, ellipse(sh, (st_cy, cx), (int(rows * 0.025), int(cols * 0.013))),
              HU['bone_cancel'] + 80, noise_std=40)

        rib_angles = [0.20, 0.35, 0.50, 0.65, 0.80]
        for ang in rib_angles:
            theta = np.pi * ang
            for side in (-1, 1):
                rib_cy = int(cy + bry * 0.62 * np.sin(theta - 0.35))
                rib_cx = int(cx + side * brx * 0.82 * np.cos(theta * 0.6))
                paint(image, ellipse(sh, (rib_cy, rib_cx), (7, 10)), HU['bone_cortex'] - 100, noise_std=50)
                paint(image, ellipse(sh, (rib_cy, rib_cx), (4, 6)), HU['bone_cancel'] - 50, noise_std=30)

        if pos < 0.22:
            t_mask = ellipse(sh, (cy - int(rows * 0.04), cx), (int(rows * 0.028), int(cols * 0.018)))
            paint(image, t_mask, HU['air'] + 50, noise_std=20)
            t_wall = ring(sh, (cy - int(rows * 0.04), cx),
                          (int(rows * 0.028), int(cols * 0.018)),
                          (int(rows * 0.020), int(cols * 0.012)))
            paint(image, t_wall, HU['soft_tissue'], noise_std=10)
        elif pos < 0.32:
            for side in (-1, 1):
                br_cx = cx + side * int(cols * 0.04)
                br_mask = ellipse(sh, (cy - int(rows * 0.04), br_cx),
                                  (int(rows * 0.018), int(cols * 0.013)))
                paint(image, br_mask, HU['air'] + 50, noise_std=20)

        eso_mask = ellipse(sh, (sp_cy - sp_r - 10, cx + 8), (6, 5))
        paint(image, eso_mask, HU['air'] + 200, noise_std=80)

    elif pos < 0.63:
        t = (pos - 0.48) / 0.15

        if t < 0.5:
            base_size = int(rows * 0.07 * (1 - t * 2))
            for side in (-1, 1):
                lcx = cx + side * int(cols * 0.14)
                paint(image, ellipse(sh, (cy - int(rows * 0.11), lcx), (base_size, int(cols * 0.08))),
                      HU['lung'], noise_std=40)

        l_ry = int(rows * (0.18 - 0.02 * t))
        l_rx = int(cols * (0.16 - 0.02 * t))
        liv_cy = cy + int(rows * 0.015)
        liv_cx = cx + int(cols * 0.10)
        liver_mask = ellipse(sh, (liv_cy, liv_cx), (l_ry, l_rx))
        paint(image, liver_mask, HU['liver'], noise_std=8)
        for _ in range(8):
            hvy = liv_cy + random.randint(-int(rows * 0.10), int(rows * 0.10))
            hvx = liv_cx + random.randint(-int(cols * 0.09), int(cols * 0.09))
            hv_m = ellipse(sh, (hvy, hvx), (4, 6)) & liver_mask
            paint(image, hv_m, HU['blood'] - 10, noise_std=5)

        gb_cy = liv_cy + int(rows * 0.09)
        gb_cx = liv_cx - int(cols * 0.06)
        gb_mask = ellipse(sh, (gb_cy, gb_cx), (int(rows * 0.038), int(cols * 0.028)))
        paint(image, gb_mask, HU['gallbladder'], noise_std=5)
        paint(image, ring(sh, (gb_cy, gb_cx),
                          (int(rows * 0.038), int(cols * 0.028)),
                          (int(rows * 0.026), int(cols * 0.018))),
              HU['soft_tissue'] + 5, noise_std=6)

        sp_ry = int(rows * (0.095 - 0.01 * t))
        sp_rx = int(cols * (0.075 - 0.01 * t))
        spl_cy = cy - int(rows * 0.025)
        spl_cx = cx - int(cols * 0.135)
        paint(image, ellipse(sh, (spl_cy, spl_cx), (sp_ry, sp_rx)), HU['spleen'], noise_std=8)

        st_cy = cy - int(rows * 0.045)
        st_cx = cx - int(cols * 0.06)
        st_mask = ellipse(sh, (st_cy, st_cx), (int(rows * 0.075), int(cols * 0.065)))
        paint(image, st_mask, HU['bowel_air'] + 50, noise_std=80)
        paint(image, ring(sh, (st_cy, st_cx),
                          (int(rows * 0.075), int(cols * 0.065)),
                          (int(rows * 0.058), int(cols * 0.048))),
              HU['soft_tissue'] + 5, noise_std=8)

        pan_mask = ellipse(sh, (cy + int(rows * 0.025), cx - int(cols * 0.015)),
                           (int(rows * 0.030), int(cols * 0.090)))
        paint(image, pan_mask, HU['pancreas'], noise_std=8)

        for side in (-1, 1):
            for i in range(3):
                rib_cy = cy + int(rows * (0.08 + i * 0.07))
                rib_cx = cx + side * int(cols * (0.22 - i * 0.01))
                paint(image, ellipse(sh, (rib_cy, rib_cx), (7, 9)), HU['bone_cortex'] - 80, noise_std=50)
                paint(image, ellipse(sh, (rib_cy, rib_cx), (4, 5)), HU['bone_cancel'] - 30, noise_std=30)

    elif pos < 0.78:
        t = (pos - 0.63) / 0.15

        for side in (-1, 1):
            k_cy = cy + int(rows * (0.045 + 0.01 * t)) - side * int(rows * 0.015)
            k_cx = cx + side * int(cols * 0.125)
            k_ry = int(rows * (0.075 - 0.005 * t))
            k_rx = int(cols * (0.048 - 0.003 * t))
            paint(image, ellipse(sh, (k_cy, k_cx), (k_ry, k_rx)), HU['kidney_cortex'], noise_std=8)
            paint(image, ellipse(sh, (k_cy, k_cx), (int(k_ry * 0.68), int(k_rx * 0.68))),
                  HU['kidney_medull'], noise_std=8)
            paint(image, ellipse(sh, (k_cy, k_cx), (int(k_ry * 0.32), int(k_rx * 0.32))),
                  HU['fat'] + 20, noise_std=10)

        for side in (-1, 1):
            ps_cx = cx + side * int(cols * 0.082)
            ps_cy = cy + int(rows * 0.075)
            paint(image, ellipse(sh, (ps_cy, ps_cx), (int(rows * 0.065), int(cols * 0.040))),
                  HU['muscle'], noise_std=10)

        for side in (-1, 1):
            col_cx = cx + side * int(cols * 0.155)
            col_cy = cy + int(rows * 0.02)
            col_mask = ellipse(sh, (col_cy, col_cx), (int(rows * 0.055), int(cols * 0.042)))
            paint(image, col_mask, HU['bowel_air'] + random.randint(50, 200), noise_std=60)
            paint(image, ring(sh, (col_cy, col_cx),
                              (int(rows * 0.055), int(cols * 0.042)),
                              (int(rows * 0.038), int(cols * 0.028))),
                  HU['soft_tissue'], noise_std=8)

        for _ in range(10):
            by = cy + random.randint(-int(rows * 0.07), int(rows * 0.06))
            bx = cx + random.randint(-int(cols * 0.09), int(cols * 0.09))
            b_ry = random.randint(12, 22)
            b_rx = random.randint(10, 18)
            content_hu = random.choice([HU['bowel_air'], HU['bowel_air'] + 200,
                                         HU['bowel_fluid'], HU['bowel_fluid'] + 10])
            b_mask = ellipse(sh, (by, bx), (b_ry, b_rx))
            paint(image, b_mask, content_hu, noise_std=25)
            paint(image, ring(sh, (by, bx), (b_ry, b_rx), (max(b_ry - 5, 3), max(b_rx - 4, 3))),
                  HU['soft_tissue'], noise_std=8)

        for side in (-1, 1):
            rm_cx = cx + side * int(cols * 0.055)
            rm_cy = cy - int(rows * 0.04)
            paint(image, ellipse(sh, (rm_cy, rm_cx), (int(rows * 0.065), int(cols * 0.038))),
                  HU['muscle'], noise_std=10)

    else:
        t = (pos - 0.78) / 0.22

        for side in (-1, 1):
            il_cx = cx + side * int(cols * (0.155 + 0.02 * t))
            il_cy = cy + int(rows * 0.04)
            il_ry = int(rows * (0.11 + 0.02 * t))
            il_rx = int(cols * (0.10 + 0.02 * t))
            paint(image, ellipse(sh, (il_cy, il_cx), (il_ry, il_rx)), HU['bone_cancel'] + 50, noise_std=40)
            paint(image, ring(sh, (il_cy, il_cx), (il_ry, il_rx),
                              (il_ry - int(rows * 0.015), il_rx - int(cols * 0.013))),
                  HU['bone_cortex'], noise_std=40)

        bl_ry = int(rows * (0.10 - 0.05 * t))
        bl_rx = int(cols * (0.09 - 0.04 * t))
        if bl_ry > 10:
            bl_cy = cy - int(rows * (0.02 + 0.03 * t))
            bl_mask = ellipse(sh, (bl_cy, cx), (bl_ry, bl_rx))
            paint(image, bl_mask, HU['bladder'], noise_std=4)
            paint(image, ring(sh, (bl_cy, cx), (bl_ry, bl_rx),
                              (bl_ry - int(rows * 0.012), bl_rx - int(cols * 0.010))),
                  HU['soft_tissue'] + 10, noise_std=6)

        rect_cy = cy + int(rows * 0.055)
        rect_mask = ellipse(sh, (rect_cy, cx), (int(rows * 0.042), int(cols * 0.042)))
        paint(image, rect_mask, HU['bowel_air'] + random.randint(100, 300), noise_std=60)
        paint(image, ring(sh, (rect_cy, cx),
                          (int(rows * 0.042), int(cols * 0.042)),
                          (int(rows * 0.026), int(cols * 0.026))),
              HU['soft_tissue'], noise_std=8)

        for side in (-1, 1):
            il_m_cx = cx + side * int(cols * 0.090)
            il_m_cy = cy + int(rows * 0.025)
            paint(image, ellipse(sh, (il_m_cy, il_m_cx), (int(rows * 0.058), int(cols * 0.042))),
                  HU['muscle'], noise_std=10)

        if t > 0.55:
            fh_t = (t - 0.55) / 0.45
            fh_r = int(rows * 0.065 * fh_t)
            if fh_r > 8:
                for side in (-1, 1):
                    fh_cx = cx + side * int(cols * (0.185 + 0.01 * fh_t))
                    fh_cy = cy + int(rows * 0.06)
                    paint(image, ellipse(sh, (fh_cy, fh_cx), (fh_r, fh_r)), HU['bone_cancel'] + 80, noise_std=40)
                    paint(image, ring(sh, (fh_cy, fh_cx), (fh_r, fh_r),
                                      (fh_r - int(rows * 0.010), fh_r - int(rows * 0.010))),
                          HU['bone_cortex'], noise_std=40)

        if t > 0.30:
            paint(image, ellipse(sh, (cy - int(rows * 0.005), cx), (int(rows * 0.028), int(cols * 0.048))),
                  HU['bone_cancel'] + 100, noise_std=40)

    # Final CT range clip + blur
    image[~body_mask] = HU['air']
    image = np.clip(image, -1024, 3071)
    image = gaussian_filter(image, sigma=1.0)
    image[~body_mask] = HU['air']
    return image.astype(np.int16)


# ──────────────────────────────────────────────────────────────────────────────
# Anatomical slice generator – MR (T1-weighted)
# ──────────────────────────────────────────────────────────────────────────────

def generate_mr_slice(slice_idx: int, num_slices: int, rows: int = 512, cols: int = 512) -> np.ndarray:
    """
    Return MR T1-weighted equivalent of the CT slice (0–1023 unsigned range).
    Derives anatomy from generate_ct_slice() then applies HU → MR T1 mapping.
    """
    ct = generate_ct_slice(slice_idx, num_slices, rows, cols).astype(np.float32)
    mr = np.zeros_like(ct)

    mr[ct < -500]                      = np.interp(ct[ct < -500],           [-1024, -500], [0,   40])
    m = (ct >= -500) & (ct < -100);    mr[m] = np.interp(ct[m],             [-500,  -100], [40,  200])
    m = (ct >= -100) & (ct < -50);     mr[m] = np.interp(ct[m],             [-100,   -50], [820, 950])
    m = (ct >= -50)  & (ct <   0);     mr[m] = np.interp(ct[m],             [-50,      0], [950, 200])
    m = (ct >=   0)  & (ct <  80);     mr[m] = np.interp(ct[m],             [0,       80], [80,  520])
    m = (ct >=  80)  & (ct < 200);     mr[m] = np.interp(ct[m],             [80,     200], [520, 560])
    m = (ct >= 200)  & (ct < 400);     mr[m] = np.interp(ct[m],             [200,    400], [700, 750])
    mr[ct >= 400]                      = np.interp(ct[ct >= 400],            [400,   3071], [120,  50])

    mr += np.random.normal(0, 12, mr.shape)
    mr[ct <= -800] = 0
    mr = np.clip(mr, 0, 1023)
    return mr.astype(np.int16)


# ──────────────────────────────────────────────────────────────────────────────
# Pixel-data dispatcher: single entry point keyed by modality string
# ──────────────────────────────────────────────────────────────────────────────

PIXEL_GENERATORS = {
    "CT": generate_ct_slice,
    "MR": generate_mr_slice,
}


def generate_slice(modality: str, slice_idx: int, num_slices: int,
                   rows: int = 512, cols: int = 512) -> np.ndarray:
    """
    Dispatch to the correct pixel-data generator based on *modality*.

    Parameters
    ----------
    modality   : 'CT' or 'MR'
    slice_idx  : index of this slice within the series
    num_slices : total slices in the series
    rows, cols : pixel matrix dimensions

    Returns
    -------
    np.ndarray  dtype=int16, shape=(rows, cols)
    """
    generator = PIXEL_GENERATORS.get(modality)
    if generator is None:
        raise ValueError(f"Unknown modality '{modality}'. Valid choices: {list(PIXEL_GENERATORS)}")
    return generator(slice_idx, num_slices, rows, cols)


# ──────────────────────────────────────────────────────────────────────────────
# DICOM file writer
# ──────────────────────────────────────────────────────────────────────────────

def write_dicom(filepath: str, pixel_data: np.ndarray, modality: str,
                patient_id: str, patient_name: str,
                study_uid: str, series_uid: str,
                series_number: int, instance_number: int,
                slice_location: float, slice_thickness: float,
                study_date: str, study_time: str):

    sop_uid = generate_uid()
    sop_class = {
        "CT": "1.2.840.10008.5.1.4.1.1.2",
        "MR": "1.2.840.10008.5.1.4.1.1.4",
    }[modality]

    rows, cols = pixel_data.shape
    pixel_rep  = 1 if modality == "CT" else 0

    meta = Dataset()
    meta.MediaStorageSOPClassUID    = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID          = ExplicitVRLittleEndian
    meta.ImplementationClassUID     = generate_uid()
    meta.ImplementationVersionName  = "BodyPhantom_2.0"

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
    ds.AccessionNumber        = f"ACC{random.randint(1000, 9999)}"
    ds.StudyDescription       = f"{modality} Body Phantom Study"
    ds.ReferringPhysicianName = "Phantom^Generator"

    ds.SeriesInstanceUID  = series_uid
    ds.SeriesNumber       = series_number
    ds.SeriesDate         = study_date
    ds.SeriesTime         = study_time
    ds.SeriesDescription  = f"{modality} Body Series {series_number}"
    ds.Modality           = modality
    ds.BodyPartExamined   = "CHEST"

    ds.FrameOfReferenceUID        = generate_uid()
    ds.PositionReferenceIndicator = ""

    ds.Manufacturer          = "BodyPhantom Inc."
    ds.ManufacturerModelName = "AnatomoSim 5000"
    ds.SoftwareVersions      = "2.0.0"

    ds.SOPClassUID    = sop_class
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = instance_number
    ds.ImageType      = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.ContentDate    = study_date
    ds.ContentTime    = study_time

    ds.PixelSpacing            = [0.7422, 0.7422]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient    = [-(cols * 0.7422 / 2), -(rows * 0.7422 / 2), slice_location]
    ds.SliceThickness          = slice_thickness
    ds.SliceLocation           = slice_location

    ds.SamplesPerPixel          = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows                     = rows
    ds.Columns                  = cols
    ds.BitsAllocated            = 16
    ds.BitsStored               = 16
    ds.HighBit                  = 15
    ds.PixelRepresentation      = pixel_rep
    ds.PixelData                = pixel_data.tobytes()

    if modality == "CT":
        ds.RescaleIntercept  = -1024
        ds.RescaleSlope      = 1
        ds.RescaleType       = "HU"
        ds.KVP               = 120
        ds.ExposureTime      = 750
        ds.XRayTubeCurrent   = 250
        ds.ConvolutionKernel = "B30f"
        ds.WindowCenter      = 40
        ds.WindowWidth       = 400

    if modality == "MR":
        ds.ScanningSequence  = "SE"
        ds.SequenceVariant   = "NONE"
        ds.ScanOptions       = ""
        ds.MRAcquisitionType = "2D"
        ds.RepetitionTime    = 550.0
        ds.EchoTime          = 14.0
        ds.FlipAngle         = 90
        ds.NumberOfAverages  = 2
        ds.WindowCenter      = 512
        ds.WindowWidth       = 800

    pydicom.dcmwrite(filepath, ds)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def create_body_dicom(
    modality: str,
    output_dir: str = "body_dicoms",
    num_series: int = None,
    slices_per_series: int = None,
    image_size: int = 512,
):
    """
    Generate anatomically realistic DICOM series for one body phantom study,
    all slices in the chosen *modality*.

    Parameters
    ----------
    modality         : 'CT' or 'MR'  (from user selection)
    output_dir       : root output folder
    num_series       : number of series (default random 3–4)
    slices_per_series: slices per series (default random 10–15)
    image_size       : pixel matrix size (256 or 512 recommended)
    """
    if modality not in VALID_MODALITIES:
        raise ValueError(f"modality must be one of {VALID_MODALITIES}, got '{modality}'")

    os.makedirs(output_dir, exist_ok=True)

    num_series   = num_series or random.randint(3, 4)
    patient_id   = f"PAT-{random.randint(100000, 999999)}"
    patient_name = "Body^Phantom^01"
    study_uid    = generate_uid()
    now          = datetime.datetime.now()
    study_date   = now.strftime("%Y%m%d")
    study_time   = now.strftime("%H%M%S")
    slice_thickness = 5.0   # mm

    print(f"  Patient ID   : {patient_id}")
    print(f"  Modality     : {modality}  (all series)")
    print(f"  Study UID    : {study_uid}")
    print(f"  Series count : {num_series}")
    print(f"  Matrix size  : {image_size} × {image_size}")
    print("─" * 60)

    for series_idx in range(num_series):
        series_uid    = generate_uid()
        series_number = random.randint(100, 999)
        num_slices    = slices_per_series or random.randint(10, 15)

        series_dir = os.path.join(
            output_dir,
            f"Series_{series_idx + 1:02d}_{modality}_S{series_number}"
        )
        os.makedirs(series_dir, exist_ok=True)

        print(f"\n  [{series_idx + 1}/{num_series}]  {modality}  |  "
              f"SeriesNumber={series_number}  |  {num_slices} slices")

        for sl in range(num_slices):
            filepath  = os.path.join(series_dir, f"slice_{sl + 1:04d}.dcm")
            slice_loc = round(sl * slice_thickness, 3)

            # ── Pixel data created for the user-selected modality ─────────────
            pixels = generate_slice(modality, sl, num_slices, image_size, image_size)

            write_dicom(
                filepath        = filepath,
                pixel_data      = pixels,
                modality        = modality,
                patient_id      = patient_id,
                patient_name    = patient_name,
                study_uid       = study_uid,
                series_uid      = series_uid,
                series_number   = series_number,
                instance_number = sl + 1,
                slice_location  = slice_loc,
                slice_thickness = slice_thickness,
                study_date      = study_date,
                study_time      = study_time,
            )
            print(f"    slice {sl + 1:>2}/{num_slices}  →  {filepath}", end="\r")

        print(f"    ✓  {num_slices} slices saved  →  {series_dir}          ")

    print(f"\n{'═' * 60}")
    print(f"  ✓  Done!  All {modality} series saved under '{output_dir}/'")
    print(f"{'═' * 60}")
    print("\n  Open with:  3D Slicer · Horos · OsiriX · MicroDicom · RadiAnt")
    print("  Load the root folder — viewer will auto-group series.\n")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_banner()

    # ── Step 1: ask user to select modality ───────────────────────────────────
    chosen_modality = select_modality()

    # ── Step 2: generate all series in that modality ──────────────────────────
    create_body_dicom(
        modality          = chosen_modality,
        output_dir        = "body_dicoms_sr",
        # num_series      = 4,      # uncomment to fix
        # slices_per_series = 12,   # uncomment to fix
        image_size        = 512,
    )