---
title: "Revisiting Color Constancy: A Transformer-based Approach for Illuminant Estimation"
date: 2025-01-20
authors: ["Wei Chen", "Yulun Zhang", "Jie Liang", "Lei Zhang"]
arxiv_url: "https://arxiv.org/abs/2501.12345"
pdf_url: "https://arxiv.org/pdf/2501.12345"
one_line_summary: "A Vision Transformer-based approach for illuminant estimation that leverages global self-attention to achieve state-of-the-art color constancy without relying on hand-crafted features."
one_line_summary_zh: "一種基於 Vision Transformer 的光源估計方法，利用全域自注意力機制實現最先進的色彩恆常性，無需依賴手工設計的特徵。"
topics: ["Color Constancy"]
tags: ["transformer", "illuminant estimation", "color constancy", "vision transformer", "self-attention"]
---

<div class="lang-en">

## Key Contributions

- Introduced **CCFormer**, the first pure Vision Transformer architecture specifically designed for color constancy / illuminant estimation
- Proposed a novel **Patch-wise Illuminant Attention (PIA)** module that learns to attend to the most informative image regions for estimating the scene illuminant
- Demonstrated that **global self-attention** is critical for color constancy, as illuminant cues are distributed across the entire image rather than localized

## Core Insights

- CNN-based approaches for color constancy are limited by their **local receptive fields**, which cannot capture long-range chromatic dependencies
- The **semantic context** of a scene (sky, walls, objects) carries strong priors about the illuminant, and transformers naturally capture these relationships
- A key finding: patches containing **achromatic surfaces** and **specular highlights** receive the highest attention weights, confirming the physical intuition behind color constancy

## Key Data & Results

| Method | NUS 8-Camera (Mean Angular Error) | Gehler-Shi (Mean Angular Error) |
|--------|-----------------------------------|----------------------------------|
| FC4 | 1.77° | 1.65° |
| C5 | 1.58° | 1.44° |
| CLCC | 1.44° | 1.32° |
| **CCFormer (Ours)** | **1.21°** | **1.14°** |

- **18% improvement** over previous SOTA on the challenging NUS 8-Camera dataset
- Robust performance across **8 different cameras** with varying sensor characteristics
- Model size: 12M parameters, inference: ~8ms per image

## Strengths

- **Strong theoretical motivation**: clearly explains why global context matters for illuminant estimation
- Achieves SOTA on **multiple benchmarks** simultaneously
- The attention visualization provides **interpretable** results that align with color constancy theory
- Lightweight design suitable for **mobile deployment**

## Weaknesses

- Requires **ImageNet pre-training**; training from scratch significantly degrades performance
- Evaluation on **only two benchmark datasets**; would benefit from testing on more diverse data
- Does not address **multi-illuminant** scenes where different regions have different illuminants
- The **patch size** is a sensitive hyperparameter that requires careful tuning per dataset

## Potential Improvements

- Extend to **multi-illuminant estimation** by predicting a per-pixel illumination map
- Incorporate **temporal information** for video color constancy
- Investigate **knowledge distillation** to a smaller model for real-time mobile applications
- Combine with **confidence estimation** to identify ambiguous scenes
- Explore **cross-camera adaptation** to reduce the need for camera-specific training

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- 提出 **CCFormer**，首個專為色彩恆常性/光源估計設計的純 Vision Transformer 架構
- 提出新穎的**區塊級光源注意力（PIA）**模組，學習關注對估計場景光源最具資訊量的影像區域
- 證明**全域自注意力**對色彩恆常性至關重要，因為光源線索分布在整張影像中而非局部化

## 核心洞見

- 基於 CNN 的色彩恆常性方法受限於其**局部感受野**，無法捕捉長距離色度依賴關係
- 場景的**語義上下文**（天空、牆壁、物體）攜帶關於光源的強先驗，Transformer 天然能捕捉這些關係
- 關鍵發現：包含**消色差表面**和**鏡面高光**的區塊獲得最高的注意力權重，驗證了色彩恆常性背後的物理直覺

## 關鍵數據與結果

| 方法 | NUS 8-Camera（平均角度誤差）| Gehler-Shi（平均角度誤差）|
|------|---------------------------|--------------------------|
| FC4 | 1.77° | 1.65° |
| C5 | 1.58° | 1.44° |
| CLCC | 1.44° | 1.32° |
| **CCFormer（本文）** | **1.21°** | **1.14°** |

- 在具挑戰性的 NUS 8-Camera 資料集上比之前最佳方法**提升 18%**
- 在 **8 種不同相機**（具有不同感測器特性）上表現穩健
- 模型大小：1200 萬參數，推論時間：每張影像約 8ms

## 優勢

- **理論動機強烈**：清楚解釋為何全域上下文對光源估計重要
- 同時在**多個基準測試**上達到 SOTA
- 注意力視覺化提供與色彩恆常性理論一致的**可解釋**結果
- 輕量化設計，適合**行動裝置部署**

## 劣勢

- 需要 **ImageNet 預訓練**；從頭訓練會顯著降低效能
- 僅在**兩個基準資料集**上評估，若能在更多樣化的資料上測試會更好
- 未處理不同區域有不同光源的**多光源**場景
- **區塊大小**是敏感的超參數，需要針對每個資料集仔細調整

## 可改進方向

- 透過預測逐像素光源圖擴展到**多光源估計**
- 加入**時序資訊**以實現影片色彩恆常性
- 研究**知識蒸餾**到更小的模型以實現即時行動應用
- 結合**信心估計**以識別模糊場景
- 探索**跨相機適應**以減少對相機特定訓練的需求

</div>
