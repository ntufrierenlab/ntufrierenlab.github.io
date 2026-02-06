---
title: "Learning Auto White Balance from RAW Images: A Physically-Informed Neural ISP Approach"
date: 2025-02-01
authors: ["Shuwei Li", "Robby T. Tan", "Jiangwei Lao", "Bo Li"]
arxiv_url: "https://arxiv.org/abs/2502.00123"
pdf_url: "https://arxiv.org/pdf/2502.00123"
one_line_summary: "A physically-informed neural ISP pipeline that learns auto white balance directly from RAW sensor data, combining differentiable color science transforms with neural networks for superior AWB accuracy."
one_line_summary_zh: "一個以物理為基礎的神經 ISP 管線，直接從 RAW 感測器資料學習自動白平衡，結合可微分色彩科學轉換與神經網路以達到優越的 AWB 精度。"
topics: ["Auto White Balance", "Color Constancy"]
tags: ["auto white balance", "RAW processing", "neural ISP", "color science", "differentiable rendering"]
---

<div class="lang-en">

## Key Contributions

- Proposed **PhysAWB**, a neural ISP module that operates directly on **RAW sensor data** for auto white balance, preserving maximum color information
- Introduced a **differentiable color science pipeline** that embeds physical constraints (von Kries adaptation, Bradford transform) into the learning framework
- Built the first **large-scale RAW AWB dataset** with 15,000 RAW images from 12 camera models, with ground-truth illuminants measured by a spectrophotometer

## Core Insights

- Most existing AWB methods operate on **sRGB images** that have already undergone nonlinear transformations (gamma, tone mapping), which **destroys the linear relationship** between scene illuminant and pixel values
- By working in the **linear RAW domain**, the von Kries diagonal model becomes physically accurate, enabling the network to learn the correction more effectively
- The Bradford chromatic adaptation transform provides a **better perceptual color space** for white balance correction than simple diagonal scaling in camera RGB

## Key Data & Results

| Method | Domain | Mean Angular Error | Worst 25% |
|--------|--------|--------------------|-----------|
| FC4 | sRGB | 1.77° | 4.12° |
| CLCC | sRGB | 1.44° | 3.56° |
| FFCC | RAW | 1.38° | 3.21° |
| **PhysAWB (Ours)** | **RAW** | **1.05°** | **2.34°** |

- **24% improvement** over FFCC in mean angular error
- Consistently outperforms sRGB-based methods across **all 12 camera models**
- Particularly strong improvement on **worst-case scenarios** (25% hardest images)
- End-to-end inference: ~12ms (including RAW demosaicing)

## Strengths

- **Physically grounded**: the architecture reflects actual color science, not just black-box learning
- The **differentiable pipeline** allows end-to-end training while maintaining physical constraints
- The RAW dataset is a **significant contribution** to the community
- Strong **cross-camera generalization** due to the physics-informed design

## Weaknesses

- Requires access to **RAW image data**, which is not always available (e.g., compressed JPEG from smartphones)
- The Bradford transform assumption may not hold for **extreme illuminant changes** (e.g., very low color temperature)
- **Computational overhead** of the differentiable color pipeline during training (3x slower than simple CNN training)
- Limited evaluation on **video sequences** where temporal consistency matters

## Potential Improvements

- Add a **fallback sRGB pathway** for cases where RAW data is unavailable
- Incorporate **metadata** (camera model, ISO, exposure) as auxiliary inputs for better generalization
- Extend to **spatially-varying white balance** for scenes with multiple illuminants
- Investigate **few-shot adaptation** for new camera models with minimal calibration data
- Combine with **learned ISP** modules (demosaicing, denoising) for a fully differentiable camera pipeline

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- 提出 **PhysAWB**，一個直接在 **RAW 感測器資料**上運作的神經 ISP 模組，保留最大色彩資訊進行自動白平衡
- 引入**可微分色彩科學管線**，將物理約束（von Kries 適應、Bradford 轉換）嵌入學習框架
- 建立首個**大規模 RAW AWB 資料集**，包含來自 12 種相機型號的 15,000 張 RAW 影像，地面真值光源由分光光度計測量

## 核心洞見

- 大多數現有 AWB 方法在 **sRGB 影像**上操作，這些影像已經過非線性轉換（gamma、色調映射），**破壞了**場景光源與像素值之間的線性關係
- 在**線性 RAW 域**中工作，von Kries 對角線模型變得物理準確，使網路能更有效地學習校正
- Bradford 色彩適應轉換提供了比相機 RGB 中簡單對角線縮放**更好的感知色彩空間**

## 關鍵數據與結果

| 方法 | 域 | 平均角度誤差 | 最差 25% |
|------|------|------------|----------|
| FC4 | sRGB | 1.77° | 4.12° |
| CLCC | sRGB | 1.44° | 3.56° |
| FFCC | RAW | 1.38° | 3.21° |
| **PhysAWB（本文）** | **RAW** | **1.05°** | **2.34°** |

- 平均角度誤差比 FFCC **提升 24%**
- 在**所有 12 種相機型號**上持續優於基於 sRGB 的方法
- 在**最困難的場景**（最差 25% 影像）上改進尤為顯著
- 端到端推論時間：約 12ms（包含 RAW 去馬賽克）

## 優勢

- **以物理為基礎**：架構反映實際色彩科學，而非黑盒學習
- **可微分管線**允許端到端訓練同時維持物理約束
- RAW 資料集是對社群的**重要貢獻**
- 得益於物理知識驅動的設計，具有強大的**跨相機泛化**能力

## 劣勢

- 需要存取 **RAW 影像資料**，但這並非總是可用（如智慧型手機的壓縮 JPEG）
- Bradford 轉換假設在**極端光源變化**下可能不成立（如極低色溫）
- 訓練時可微分色彩管線的**計算開銷**（比簡單 CNN 訓練慢 3 倍）
- 在**影片序列**上的評估有限，而時序一致性在影片中很重要

## 可改進方向

- 為 RAW 資料不可用的情況增加**備援 sRGB 通道**
- 將**後設資料**（相機型號、ISO、曝光）作為輔助輸入以提升泛化能力
- 擴展到**空間變化白平衡**以處理多光源場景
- 研究**少樣本適應**以最少校準資料支援新相機型號
- 與**可學習 ISP** 模組（去馬賽克、去噪）結合，建立完全可微分的相機管線

</div>
