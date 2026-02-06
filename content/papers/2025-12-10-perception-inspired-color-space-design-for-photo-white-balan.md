---
title: "Perception-Inspired Color Space Design for Photo White Balance Editing"
date: 2025-12-10
authors:
  - "Yang Cheng"
  - "Ziteng Cui"
  - "Shenghan Su"
  - "Lin Gu"
  - "Zenghui Zhang"
arxiv_url: "https://arxiv.org/abs/2512.09383"
pdf_url: "https://arxiv.org/pdf/2512.09383"
one_line_summary: " This paper proposes a Learnable HSI (LHSI) color space with optimizable luminance axis and adaptive mapping functions for sRGB white balance correction, achieving competitive performance with a compact 6.4MB model."
one_line_summary_zh: " 本文提出可學習的 HSI (LHSI) 色彩空間，透過優化亮度軸與自適應映射函數實現 sRGB 白平衡校正，以僅 6.4MB 的緊湊模型達到競爭性效能。"
topics: ["Color Constancy"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **First learnable color space for white balance correction**: Proposes LHSI (Learnable HSI) color space that treats the luminance axis as an optimizable parameter rather than a fixed transformation, addressing limitations of fixed color spaces like sRGB, CIELAB, and standard HSI
- **Perception-inspired adaptive color representation**: Introduces learnable nonlinear mapping functions (piecewise linear with monotonicity constraints) across all three channels (hue, saturation, intensity) to enhance representational flexibility while maintaining invertibility
- **Specialized Mamba-based architecture**: Develops Dual-Branch Chromatic-Luminance Attention Network (DCLAN) with Mamba-Attention Blocks (MABs) that exploits the disentangled chromatic and intensity components of LHSI space using State Space Models for efficient long-range dependency modeling
- **Compact and efficient model**: Achieves competitive performance with only 6.4MB model size (~21% of WBFlow's 30.2MB), demonstrating efficiency advantages for practical deployment

## Core Insights

- **Limitation of fixed color spaces**: Traditional color spaces (sRGB, CIELAB, HSV) use fixed analytical transformations that fail to account for complex, device-specific ISP nonlinearities, resulting in suboptimal feature separation for white balance tasks
- **Optimizable luminance axis**: The learned luminance axis converges to **→n = (0.48, 0.76, 0.45)** after training, significantly different from standard HSI's (0.33, 0.33, 0.33), revealing task-specific optimal color decomposition
- **Disentanglement enhances correction**: By separating luminance from chromaticity in a learnable manner and processing them with dual-branch architecture, the network can more effectively suppress color casts while preserving structural details
- **Trade-off between specialization and generalization**: The learned color space provides superior in-distribution performance but shows slightly reduced cross-dataset generalization compared to more generic approaches, highlighting the balance between task-specific priors and broader adaptability

## Key Data & Results

**Performance Comparison on Set1-Test (21,046 images)**

| Method | MSE ↓ Mean | MAE (°) ↓ Mean | ΔE2000 ↓ Mean | Size (MB) |
|--------|------------|----------------|----------------|-----------|
| KNN | 77.49 | 3.06 | 3.58 | 21.8 |
| Deep-WB | 82.55 | 3.12 | 3.77 | 16.7 |
| Mixed-WB | 142.25 | 4.07 | 4.55 | 5.1 |
| WBFlow | 78.89 | 2.67 | 3.13 | 30.2 |
| SWBNet | 111.62 | 4.11 | 4.54 | 258.8 |
| **LHSI (Ours)** | **54.79** | **2.53** | **2.79** | **6.4** |

- **Best in-distribution performance**: Achieves 29.3% lower MSE (54.79 vs 77.49) and 17.3% lower MAE compared to KNN baseline on Set1-Test
- **Significant model compression**: Model size is only 6.4MB, representing 78% reduction compared to WBFlow (30.2MB) and 97.5% reduction compared to SWBNet (258.8MB)
- **Competitive cross-dataset results**: On Set2, achieves MSE of 115.53 and MAE of 3.56°; on Cube+, achieves MSE of 78.19 and MAE of 3.70°, demonstrating reasonable generalization despite specialization
- **Color space ablation results**: Using standard U-Net baseline, LHSI achieves MSE of 71.54 vs 101.79 (HVI), 113.91 (HSV), and 118.80 (CIELAB) on Set1-Test, validating the superiority of the learned representation

## Strengths

- **Novel paradigm shift**: First work to treat color space as a learnable component in white balance correction, opening new research directions for computational photography
- **Theoretically sound design**: The cylindrical color model with learnable luminance axis and monotonic piecewise linear mapping functions ensures invertibility while providing adaptive flexibility
- **Efficient architecture**: Mamba-based SSM approach achieves O(N) complexity for modeling long-range dependencies, significantly more efficient than transformer-based alternatives
- **Comprehensive evaluation**: Extensive experiments on three benchmark datasets (Set1-Test, Set2, Cube+) with multiple metrics (MSE, MAE, ΔE2000) and detailed ablation studies comparing different color spaces
- **Practical deployment advantages**: Extremely compact model size (6.4MB) makes it suitable for mobile and edge deployment scenarios

## Weaknesses

- **Limited cross-domain generalization**: Performance on out-of-distribution datasets (Set2, Cube+) is competitive but not consistently superior, suggesting the learned priors may be too specialized to the training distribution
- **Lack of mixed illumination handling**: The paper does not explicitly address scenes with multiple light sources, which are common in real-world photography (e.g., indoor/outdoor mixed lighting)
- **Insufficient interpretability analysis**: While the learned luminance axis direction is reported, there is limited analysis of what the learned mapping functions represent or why they improve performance
- **Limited baseline comparisons**: Missing comparisons with recent transformer-based methods and diffusion-based approaches (e.g., GCC mentioned in related work)
- **Training data dependency**: Uses only 12,000 images for training, which may limit the model's ability to learn generalizable color space transformations across diverse camera sensors and ISP pipelines

## Potential Improvements

- **Multi-domain training strategy**: Incorporate images from diverse camera sensors and ISP pipelines during training to enhance cross-domain generalization while maintaining task-specific advantages
- **Mixed illumination modeling**: Extend LHSI to handle spatially-varying illumination by introducing location-aware luminance axes or multiple parallel color space branches for different illumination regions
- **Interpretability enhancement**: Add visualization tools to analyze learned mapping functions and their gradients, potentially incorporating physical constraints based on colorimetry theory to improve interpretability
- **Hierarchical color space learning**: Explore learning different color space parameters at different network depths, allowing coarse-to-fine color correction with stage-specific color representations
- **Self-supervised pretraining**: Leverage large-scale unlabeled image datasets with color augmentation strategies to pretrain the color space transformation, potentially improving generalization to unseen domains
- **Uncertainty quantification**: Incorporate prediction uncertainty estimates to identify when the learned color space may not generalize well, enabling fallback to more conservative correction strategies
- **Real-world deployment validation**: Conduct user studies and real-world A/B testing to validate perceptual quality improvements beyond standard metrics, especially for challenging lighting conditions

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **首個用於白平衡校正的可學習色彩空間**：提出 LHSI (Learnable HSI) 色彩空間，將亮度軸視為可優化參數而非固定轉換，解決了 sRGB、CIELAB 和標準 HSI 等固定色彩空間的局限性
- **受感知啟發的自適應色彩表示**：在所有三個通道（色相、飽和度、亮度）引入可學習的非線性映射函數（具有單調性約束的分段線性函數），在保持可逆性的同時增強表示靈活性
- **專門的 Mamba 架構**：開發雙分支色度-亮度注意力網絡 (DCLAN)，配備 Mamba 注意力模塊 (MABs)，利用狀態空間模型高效建模長程依賴性，充分利用 LHSI 空間解耦的色度和亮度分量
- **緊湊高效的模型**：僅需 6.4MB 模型大小（約為 WBFlow 30.2MB 的 21%）即可達到競爭性能，展現實際部署的效率優勢

## 核心洞見

- **固定色彩空間的局限性**：傳統色彩空間（sRGB、CIELAB、HSV）使用固定的解析轉換，無法應對複雜的、設備特定的 ISP 非線性特性，導致白平衡任務中的特徵分離效果不佳
- **可優化的亮度軸**：學習得到的亮度軸收斂至 **→n = (0.48, 0.76, 0.45)**，與標準 HSI 的 (0.33, 0.33, 0.33) 顯著不同，揭示了任務特定的最優色彩分解方式
- **解耦增強校正能力**：透過可學習方式將亮度與色度分離，並使用雙分支架構處理，網絡能更有效地抑制色偏同時保持結構細節
- **專業化與泛化的權衡**：學習的色彩空間在分布內數據上表現優異，但在跨數據集泛化方面略有下降，凸顯了任務特定先驗與廣泛適應性之間的平衡

## 關鍵數據與結果

**在 Set1-Test 上的性能比較（21,046 張圖像）**

| 方法 | MSE ↓ 平均值 | MAE (°) ↓ 平均值 | ΔE2000 ↓ 平均值 | 大小 (MB) |
|--------|------------|----------------|----------------|-----------|
| KNN | 77.49 | 3.06 | 3.58 | 21.8 |
| Deep-WB | 82.55 | 3.12 | 3.77 | 16.7 |
| Mixed-WB | 142.25 | 4.07 | 4.55 | 5.1 |
| WBFlow | 78.89 | 2.67 | 3.13 | 30.2 |
| SWBNet | 111.62 | 4.11 | 4.54 | 258.8 |
| **LHSI (本文)** | **54.79** | **2.53** | **2.79** | **6.4** |

- **最佳分布內性能**：在 Set1-Test 上相比 KNN 基準，MSE 降低 29.3%（54.79 vs 77.49），MAE 降低 17.3%
- **顯著的模型壓縮**：模型大小僅 6.4MB，相比 WBFlow（30.2MB）減少 78%，相比 SWBNet（258.8MB）減少 97.5%
- **具競爭力的跨數據集結果**：在 Set2 上達到 MSE 115.53 和 MAE 3.56°；在 Cube+ 上達到 MSE 78.19 和 MAE 3.70°，展現合理的泛化能力
- **色彩空間消融實驗結果**：使用標準 U-Net 基準，LHSI 在 Set1-Test 上達到 MSE 71.54，優於 HVI（101.79）、HSV（113.91）和 CIELAB（118.80），驗證了學習表示的優越性

## 優勢

- **新穎的範式轉變**：首次將色彩空間視為白平衡校正中的可學習組件，為計算攝影學開啟新的研究方向
- **理論上健全的設計**：具有可學習亮度軸和單調分段線性映射函數的柱面色彩模型，在提供自適應靈活性的同時確保可逆性
- **高效的架構**：基於 Mamba 的 SSM 方法以 O(N) 複雜度建模長程依賴性，顯著優於基於 transformer 的替代方案
- **全面的評估**：在三個基準數據集（Set1-Test、Set2、Cube+）上進行廣泛實驗，使用多個指標（MSE、MAE、ΔE2000），並詳細比較不同色彩空間的消融研究
- **實際部署優勢**：極其緊湊的模型大小（6.4MB）使其適合移動和邊緣部署場景

## 劣勢

- **跨域泛化能力有限**：在分布外數據集（Set2、Cube+）上的性能具有競爭力但不一致優於其他方法，表明學習的先驗可能過於專注於訓練分布
- **缺乏混合光照處理**：論文未明確處理多光源場景，而這在現實攝影中很常見（例如室內/室外混合光照）
- **可解釋性分析不足**：雖然報告了學習得到的亮度軸方向，但對學習映射函數的含義或其改善性能的原因分析有限
- **基準比較有限**：缺少與最新基於 transformer 的方法和基於擴散的方法（如相關工作中提到的 GCC）的比較
- **訓練數據依賴**：僅使用 12,000 張圖像進行訓練，可能限制模型跨不同相機傳感器和 ISP 管線學習可泛化色彩空間轉換的能力

## 可改進方向

- **多域訓練策略**：在訓練期間納入來自不同相機傳感器和 ISP 管線的圖像，在保持任務特定優勢的同時增強跨域泛化能力
- **混合光照建模**：擴展 LHSI 以處理空間變化的光照，引入位置感知的亮度軸或針對不同光照區域的多個並行色彩空間分支
- **可解釋性增強**：添加可視化工具來分析學習的映射函數及其梯度，可能結合基於色彩學理論的物理約束以提高可解釋性
- **層次化色彩空間學習**：探索在不同網絡深度學習不同的色彩空間參數，允許具有階段特定色彩表示的由粗到細的色彩校正
- **自監督預訓練**：利用大規模無標註圖像數據集和色彩增強策略來預訓練色彩空間轉換，可能改善對未見域的泛化能力
- **不確定性量化**：納入預測不確定性估計，以識別學習的色彩空間可能無法很好泛化的情況，啟用更保守的校正策略作為備選
- **真實世界部署驗證**：進行用戶研究和真實世界 A/B 測試，以驗證標準指標之外的感知質量改善，特別是在具挑戰性的光照條件下

</div>
