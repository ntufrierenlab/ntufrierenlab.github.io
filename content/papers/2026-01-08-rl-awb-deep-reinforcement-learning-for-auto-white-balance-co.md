---
title: "RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Night-time Scenes"
date: 2026-01-08
authors:
  - "Yuan-Kang Lee"
  - "Kuan-Lin Chen"
  - "Chia-Che Chang"
  - "Yu-Lun Liu"
arxiv_url: "https://arxiv.org/abs/2601.05249"
pdf_url: "https://arxiv.org/pdf/2601.05249"
one_line_summary: " RL-AWB combines a novel nighttime statistical color constancy algorithm (SGP-LRD) with deep reinforcement learning to achieve superior cross-sensor white balance correction through adaptive per-image parameter tuning."
one_line_summary_zh: " RL-AWB 結合新型夜間統計色彩恆定性演算法（SGP-LRD）與深度強化學習，透過自適應逐影像參數調整實現卓越的跨感測器白平衡校正。"
topics: ["Auto White Balance"]
tags: []
---

Now I'll generate the structured summary based on the full paper content provided:



<div class="lang-en">

## Key Contributions

- **SGP-LRD Algorithm**: A novel nighttime-specific color constancy algorithm combining **Salient Gray Pixels** with **Local Reflectance Differences**, achieving state-of-the-art illumination estimation on nighttime benchmarks through spatial coherence-based noise filtering and luminance-adaptive confidence weighting
- **RL-AWB Framework**: The first deep reinforcement learning approach for color constancy that adaptively optimizes algorithm parameters (gray-pixel percentage N and Minkowski order p) on a per-image basis using **Soft Actor-Critic (SAC)** training with two-stage curriculum learning
- **LEVI Dataset**: The first multi-sensor nighttime color constancy benchmark comprising 700 RAW images from two camera systems (iPhone 16 Pro and Sony ILCE-6400) with ISO ranging 500-16,000, enabling rigorous cross-sensor evaluation
- **Cross-Sensor Generalization**: Achieved superior generalization with only **5 training images per dataset**, outperforming fully-supervised methods that require extensive labeled data
- **Hybrid Statistical-Learning Paradigm**: Successfully combines interpretable statistical methods with adaptive learning, preserving sensor-agnostic robustness while gaining adaptive optimization capability

## Core Insights

- **Nighttime-specific challenges**: Low-light nighttime environments fundamentally violate daytime color constancy assumptions due to severe chromatic noise, mixed illumination, and unstable gray pixel detection where sensor noise dominates signal
- **Parameter sensitivity in nighttime scenes**: The optimal configuration of algorithm parameters (N%, p) is inherently scene-dependent in low-light conditions; fixed parameters fail to generalize across diverse nighttime scenarios
- **Spatial coherence exploitation**: Reliable gray regions in nighttime scenes exhibit spatial continuity—the overlapping window design naturally amplifies high-SNR signals while filtering spurious noise through repeated sampling across neighboring regions
- **RL formulation advantage**: Formulating AWB as sequential decision-making enables adaptive parameter selection policies that mimic professional tuning experts, achieving convergence in typically 3 steps without requiring ground-truth illuminants at deployment

## Key Data & Results

| Method | NCC Dataset ||| LEVI Dataset ||| Cross-Dataset (NCC→LEVI) |||
|--------|-------------|---|---|--------------|---|---|------------------------|---|---|
| | Median | Mean | W-25% | Median | Mean | W-25% | Median | Mean | W-25% |
| **Statistical Methods** |||||||||
| GE-2nd | 3.58° | 4.64° | 9.93° | 4.17° | 4.49° | 7.76° | - | - | - |
| GI | 3.13° | 4.52° | 10.60° | 3.10° | 3.42° | 5.91° | - | - | - |
| RGP | 2.22° | 3.33° | 7.81° | 3.21° | 3.56° | 6.12° | - | - | - |
| SGP-LRD (Ours) | **2.12°** | **3.11°** | **7.22°** | 3.08° | 3.25° | 5.46° | - | - | - |
| **Learning-Based (5-shot)** |||||||||
| C⁴ | 6.24° | 7.88° | 17.42° | 7.01° | 8.22° | 15.75° | 13.18° | 13.52° | 19.40° |
| C⁵ (5) | 5.56° | 7.11° | 14.66° | 2.46° | 3.50° | 7.80° | 9.40° | 10.93° | 20.61° |
| PCC | 4.65° | 5.77° | 12.03° | 4.01° | 5.18° | 10.91° | 20.69° | 19.37° | 27.44° |
| **RL-AWB (Ours)** | **1.98°** | **3.07°** | **7.22°** | **3.01°** | **3.22°** | **5.32°** | **3.03°** | **3.24°** | **5.36°** |

- **In-domain performance**: RL-AWB achieves **1.98° median error** on NCC (6.6% improvement over SGP-LRD baseline) and **3.01° median error** on LEVI with only 5 training images
- **Cross-sensor robustness**: When trained on NCC and tested on LEVI, RL-AWB achieves **3.03° median error**, compared to 9.40° for C⁵ and 13.18° for C⁴—representing **67.8% and 77.0% error reduction** respectively
- **Sample efficiency**: With only **5 training images**, RL-AWB outperforms C⁵ trained on full 3-fold protocol (4.47° vs 1.99° median on LEVI→NCC cross-dataset evaluation)
- **Daytime generalization**: On well-lit Gehler-Shi dataset, RL-AWB achieves **2.24° median error** (5.9% improvement over SGP-LRD), demonstrating generalization beyond nighttime scenes
- **Training efficiency**: SAC agent converges in **150,000 timesteps** using 16 parallel CPU environments on Intel Core i5-13600K

## Strengths

- **Strong theoretical foundation**: Combines spatial coherence priors with luminance-adaptive confidence weighting, providing interpretable and physically-grounded nighttime color constancy rather than black-box learning
- **Exceptional data efficiency**: Achieves state-of-the-art cross-sensor performance with only 5 training images through curriculum learning (Stage 1: single-image stabilization; Stage 2: cyclic multi-image tuning), making it practical for deployment
- **Superior cross-sensor generalization**: RL-AWB maintains stable performance across different sensors (iPhone 16 Pro, Sony ILCE-6400) and datasets, while pure learning methods suffer 2-4× performance degradation under domain shift
- **Comprehensive evaluation**: Introduces LEVI dataset (700 images, 2 sensors, ISO 500-16,000) enabling rigorous multi-sensor benchmarking; evaluates on both nighttime (NCC, LEVI) and daytime (Gehler-Shi) datasets
- **Practical deployment**: No ground-truth illuminants required at inference; agent converges in typically 3 steps with clear convergence criterion (three consecutive stable estimations)

## Weaknesses

- **Limited action space**: Current implementation controls only 2 parameters (N%, p) while SGP-LRD exposes multiple tunable parameters (VarTh, ColorTh, window size w); expanding to full parameter set would substantially increase training complexity
- **Occasional over-correction**: Despite overall error reduction, RL-AWB may still over-correct challenging nighttime scenes, potentially resulting in visually degraded outputs on edge cases; lacks explicit safety constraints
- **Computational overhead**: Requires iterative policy execution (3 steps on average) compared to single-pass statistical or deep learning methods; training uses CPU-based RL updates rather than fully GPU-resident pipeline
- **Dataset scale limitations**: LEVI contains 700 images from 2 sensors; broader cross-sensor evaluation would benefit from additional camera systems (smartphones, DSLRs, surveillance cameras with varying sensor characteristics)
- **Ablation depth**: Limited ablation on reward design components (relative error improvement weight α=0.6, action cost λ=0.1, bonus structure); sensitivity to these hyperparameters not fully characterized

## Potential Improvements

- **Hierarchical action space**: Implement structured policies or low-dimensional latent action representations to efficiently coordinate control over all SGP-LRD parameters (VarTh, ColorTh, window size, etc.) without quadratic training complexity growth
- **Safety-aware optimization**: Incorporate constrained RL formulations with explicit penalties for abrupt parameter changes or visual quality degradation; add preference-based regularization to prevent over-correction on challenging scenes
- **Multi-illuminant extension**: Extend framework to handle spatially-varying illumination by learning to detect and segment regions under different light sources, applying adaptive parameter tuning per region
- **End-to-end GPU acceleration**: Migrate to fully GPU-resident training pipeline with batched environment rollouts to reduce wall-clock time; explore joint optimization across nighttime and daytime data for unified all-time AWB agent
- **Expanded cross-sensor evaluation**: Extend LEVI dataset to include additional camera systems (e.g., Nikon, Canon DSLRs; Google Pixel, Samsung smartphones; automotive/surveillance cameras) to comprehensively validate sensor-agnostic claims
- **Uncertainty quantification**: Add Bayesian or ensemble extensions to policy network to output confidence estimates; use uncertainty to trigger human-in-the-loop verification for high-risk scenes
- **Real-time deployment optimization**: Investigate policy distillation into smaller networks or lookup tables for edge devices; explore early termination strategies when convergence is detected in fewer than 3 steps

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **SGP-LRD 演算法**：一種新型夜間專用色彩恆定性演算法，結合**顯著灰色像素**與**局部反射率差異**，透過基於空間一致性的噪聲過濾和亮度自適應置信度加權，在夜間基準測試中實現最先進的照明估計
- **RL-AWB 框架**：首個用於色彩恆定性的深度強化學習方法，使用**Soft Actor-Critic (SAC)** 訓練與兩階段課程學習，在逐影像基礎上自適應優化演算法參數（灰色像素百分比 N 和 Minkowski 階數 p）
- **LEVI 資料集**：首個多感測器夜間色彩恆定性基準，包含來自兩個相機系統（iPhone 16 Pro 和 Sony ILCE-6400）的 700 張 RAW 影像，ISO 範圍 500-16,000，實現嚴格的跨感測器評估
- **跨感測器泛化能力**：僅使用**每個資料集 5 張訓練影像**即實現卓越泛化，優於需要大量標註資料的全監督方法
- **混合統計學習範式**：成功結合可解釋的統計方法與自適應學習，在獲得自適應優化能力的同時保持感測器無關的穩健性

## 核心洞見

- **夜間特定挑戰**：低光夜間環境從根本上違反日間色彩恆定性假設，原因包括嚴重的色度噪聲、混合照明，以及感測器噪聲主導信號導致的不穩定灰色像素檢測
- **夜間場景的參數敏感性**：在低光條件下，演算法參數（N%, p）的最佳配置本質上依賴於場景；固定參數無法在不同夜間場景中泛化
- **空間一致性利用**：夜間場景中可靠的灰色區域表現出空間連續性——重疊窗口設計通過在相鄰區域重複採樣，自然放大高信噪比信號同時過濾虛假噪聲
- **強化學習公式化優勢**：將 AWB 公式化為序列決策問題，實現模仿專業調校專家的自適應參數選擇策略，通常在 3 步內收斂，且部署時不需要真實照明標註

## 關鍵數據與結果

| 方法 | NCC 資料集 ||| LEVI 資料集 ||| 跨資料集 (NCC→LEVI) |||
|--------|-------------|---|---|--------------|---|---|------------------------|---|---|
| | 中位數 | 平均值 | 最差-25% | 中位數 | 平均值 | 最差-25% | 中位數 | 平均值 | 最差-25% |
| **統計方法** |||||||||
| GE-2nd | 3.58° | 4.64° | 9.93° | 4.17° | 4.49° | 7.76° | - | - | - |
| GI | 3.13° | 4.52° | 10.60° | 3.10° | 3.42° | 5.91° | - | - | - |
| RGP | 2.22° | 3.33° | 7.81° | 3.21° | 3.56° | 6.12° | - | - | - |
| SGP-LRD（本研究） | **2.12°** | **3.11°** | **7.22°** | 3.08° | 3.25° | 5.46° | - | - | - |
| **基於學習（5-shot）** |||||||||
| C⁴ | 6.24° | 7.88° | 17.42° | 7.01° | 8.22° | 15.75° | 13.18° | 13.52° | 19.40° |
| C⁵ (5) | 5.56° | 7.11° | 14.66° | 2.46° | 3.50° | 7.80° | 9.40° | 10.93° | 20.61° |
| PCC | 4.65° | 5.77° | 12.03° | 4.01° | 5.18° | 10.91° | 20.69° | 19.37° | 27.44° |
| **RL-AWB（本研究）** | **1.98°** | **3.07°** | **7.22°** | **3.01°** | **3.22°** | **5.32°** | **3.03°** | **3.24°** | **5.36°** |

- **域內性能**：RL-AWB 在 NCC 上達到 **1.98° 中位數誤差**（比 SGP-LRD 基線改善 6.6%），在 LEVI 上達到 **3.01° 中位數誤差**，僅使用 5 張訓練影像
- **跨感測器穩健性**：在 NCC 上訓練並在 LEVI 上測試時，RL-AWB 達到 **3.03° 中位數誤差**，相比 C⁵ 的 9.40° 和 C⁴ 的 13.18°，分別實現 **67.8% 和 77.0% 的誤差降低**
- **樣本效率**：僅使用 **5 張訓練影像**，RL-AWB 優於使用完整 3-fold 協議訓練的 C⁵（在 LEVI→NCC 跨資料集評估中為 4.47° vs 1.99° 中位數）
- **日間泛化**：在光照充足的 Gehler-Shi 資料集上，RL-AWB 達到 **2.24° 中位數誤差**（比 SGP-LRD 改善 5.9%），展現超越夜間場景的泛化能力
- **訓練效率**：SAC 代理在 Intel Core i5-13600K 上使用 16 個並行 CPU 環境，在 **150,000 時間步**內收斂

## 優勢

- **堅實的理論基礎**：結合空間一致性先驗與亮度自適應置信度加權，提供可解釋且基於物理的夜間色彩恆定性，而非黑盒學習
- **卓越的資料效率**：透過課程學習（階段 1：單影像穩定化；階段 2：循環多影像調整），僅用 5 張訓練影像即達到最先進的跨感測器性能，使其具實用性
- **優異的跨感測器泛化**：RL-AWB 在不同感測器（iPhone 16 Pro、Sony ILCE-6400）和資料集間保持穩定性能，而純學習方法在域轉移下性能下降 2-4 倍
- **全面的評估**：引入 LEVI 資料集（700 張影像、2 個感測器、ISO 500-16,000）實現嚴格的多感測器基準測試；在夜間（NCC、LEVI）和日間（Gehler-Shi）資料集上評估
- **實用部署**：推理時不需要真實照明標註；代理通常在 3 步內收斂，具有明確的收斂準則（三次連續穩定估計）

## 劣勢

- **有限的動作空間**：目前實作僅控制 2 個參數（N%, p），而 SGP-LRD 暴露多個可調參數（VarTh、ColorTh、窗口大小 w）；擴展到完整參數集將大幅增加訓練複雜度
- **偶爾過度校正**：儘管整體誤差降低，RL-AWB 仍可能過度校正具挑戰性的夜間場景，在邊緣案例上可能導致視覺品質下降；缺乏明確的安全約束
- **計算開銷**：相比單次通過的統計或深度學習方法，需要迭代策略執行（平均 3 步）；訓練使用基於 CPU 的強化學習更新，而非完全 GPU 常駐管線
- **資料集規模限制**：LEVI 包含來自 2 個感測器的 700 張影像；更廣泛的跨感測器評估將受益於額外的相機系統（具不同感測器特性的智慧手機、單眼相機、監控攝影機）
- **消融深度**：對獎勵設計組件的消融有限（相對誤差改善權重 α=0.6、動作成本 λ=0.1、獎勵結構）；這些超參數的敏感性未完全表徵

## 可改進方向

- **階層式動作空間**：實作結構化策略或低維潛在動作表示，以有效協調對所有 SGP-LRD 參數（VarTh、ColorTh、窗口大小等）的控制，而不會出現二次方訓練複雜度增長
- **安全感知優化**：納入受約束的強化學習公式，對參數突變或視覺品質下降施加明確懲罰；添加基於偏好的正則化以防止在具挑戰性場景上過度校正
- **多照明擴展**：將框架擴展以處理空間變化的照明，學習檢測和分割不同光源下的區域，對每個區域應用自適應參數調整
- **端到端 GPU 加速**：遷移到具批次環境展開的完全 GPU 常駐訓練管線以減少實際時間；探索跨夜間和日間資料的聯合優化，實現統一的全天候 AWB 代理
- **擴展跨感測器評估**：擴展 LEVI 資料集以包含額外的相機系統（例如 Nikon、Canon 單眼相機；Google Pixel、Samsung 智慧手機；汽車/監控攝影機）以全面驗證感測器無關聲明
- **不確定性量化**：在策略網路中添加貝葉斯或集成擴展以輸出置信度估計；使用不確定性觸發高風險場景的人機協作驗證
- **即時部署優化**：研究將策略蒸餾為較小網路或查找表以用於邊緣設備；探索當在少於 3 步內檢測到收斂時的早期終止策略

</div>
