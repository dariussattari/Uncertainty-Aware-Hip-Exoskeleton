# E13: UMAP exploration of raw windows against the AE latent: why stair ascent confounds with ramp ascent

| Field | Value |
|---|---|
| ID | E13 |
| Date | 2026-09-22 to 2026-09-23 |
| Dataset | Science Robotics (Molinaro 2024), AB15–AB34, all 7 tasks |
| Status | Complete (qualitative) |
| Sources | Live_Update.docx, "SEP 18 – 25" §3 ("UMAP Exploration") |
| Code / notebooks | `notebooks/sci_rob/model_exploration/09_umap_all_tasks.ipynb`, `_umap_exploration.py`, `interactive/` |

## Hypothesis

PCA hid the task structure (E02). A non-linear neighbourhood embedding (UMAP) of raw windows, and of the vanilla AE's
latent, should show why stair ascent is the universal blind spot. The specific hypothesis was that SA confounds with RA
far more than an OOD task such as SD confounds with anything ID.

## Method

- Interactive 2-D and 3-D UMAP and PCA of the raw 16-channel windows of all seven tasks (no model, no ID/OOD labels),
  and of the vanilla AE's 32-D latent (E05).
- Tasks were inspected pairwise: RA against RD, SA against SD, and SA/RA against SD/RD.

## Result

- PCA shows almost no task differences. UMAP is far more informative, and 3-D more so than 2-D.
- In raw space, ramp ascent and ramp descent nearly **mirror** each other; the same holds for SA and SD. Level ground
  lies between the ascending and descending representations.
- The AE latent **untangles** the gait rings, pulling them apart and projecting them onto flat planes.
- SD sits beside RD, and SA sits beside **RA**, in both raw and AE space. This explains why RA/SA and RD/SD confound,
  and why SA is harder to flag than SD: its nearest neighbour (RA) is in training.

## Conclusion

The detectors fail on stair ascent because, in every representation tried, it lands next to an ID task (ramp ascent).
This is a property of the data, not of one model. UMAP became the standard lens for latent spaces in all later work, and
the finding prompted an overall re-think of the problem. The next steps were the pelvis-IMU check (E16), the VAE (E17)
and the regime-aware pipeline (E14).

## Caveats

- Qualitative. UMAP preserves neighbourhoods, not density or distance.
