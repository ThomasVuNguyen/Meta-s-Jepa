# Meta-s-Jepa
The goal of this repository is to develop a deeper understanding of the research on JEPA &amp; World Model in general

# Jepa Family
| Model | Year | Modality | Params | Input | Output |
|---|---|---|---|---|---|
| **I-JEPA** | 2023 | Images | 632M (ViT-H) | Image + spatial mask | Latent embeddings of masked patches |
| **V-JEPA** | 2024 | Video | 632M (ViT-H) | Video clip + spatiotemporal mask | Latent embeddings of masked regions |
| **MC-JEPA** | 2024 | Video | ~307M (ViT-L) | Video frames | Motion (optical flow) + content embeddings |
| **VL-JEPA** | 2024 | Vision + Language | ~1.7B | Image/video + text query | Semantic embedding of answer (not tokens) |
| **V-JEPA 2** | 2025 | Video + Actions | 1.2B | Video frames + action vector `[7]` | Latent embedding of next world state |
| **C-JEPA** | 2025 | Images | 632M (ViT-H) | Image + spatial mask | Masked patch embeddings (VICReg regularized) |

All models predict in **latent space** — no pixel/token reconstruction. Backbone: ViT. Target encoder updated via EMA.
