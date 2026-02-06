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
one_line_summary: "RL-AWB combines statistical color constancy algorithms with deep reinforcement learning for adaptive white balance correction in nighttime photography, achieving superior cross-sensor generalization with only 5 training images."
one_line_summary_zh: "RL-AWB 結合統計色彩恆常性演算法與深度強化學習進行自適應白平衡修正，在夜間攝影中實現優越的跨感測器泛化，僅需 5 張訓練影像。"
topics: ["Auto White Balance"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **SGP-LRD Algorithm**: Developed a novel nighttime-specific color constancy algorithm integrating salient gray pixel detection with local reflectance differences, achieving state-of-the-art illumination estimation on public nighttime benchmarks (NCC and LEVI datasets).

- **RL-AWB Framework**: Designed the first deep reinforcement learning approach for color constancy using Soft Actor-Critic (SAC) training with two-stage curriculum learning, enabling adaptive per-image parameter optimization with exceptional data efficiency (only 5 training images required).

- **LEVI Dataset**: Introduced the first multi-sensor nighttime color constancy dataset comprising 700 linear RAW images from two cameras (iPhone 16 Pro and Sony ILCE-6400), enabling rigorous cross-sensor generalization evaluation with ISO ranges from 500 to 16,000.

- **Superior Cross-Sensor Generalization**: Demonstrated that the hybrid statistical-learning approach achieves remarkable robustness across different camera sensors and datasets, substantially outperforming purely learning-based methods under cross-dataset evaluation.

## Core Insights

- **Hybrid Design Advantages**: Combining statistical algorithms (interpretable and sensor-agnostic) with reinforcement learning (adaptive capability) preserves algorithmic transparency while gaining adaptive power, addressing limitations of purely data-driven approaches that suffer severe generalization degradation.

- **Parameter Sensitivity in Nighttime Scenes**: Nighttime imaging violates fundamental assumptions of traditional AWB algorithms due to low-light noise and complex illumination. Scene-dependent parameter selection (gray pixel percentage N and Minkowski norm p) is critical, making RL-based dynamic tuning superior to fixed parameters.

- **Data Efficiency Through Curriculum Learning**: A two-stage curriculum (single-image stabilization followed by multi-image cyclic training) enables the RL agent to learn robust policies with minimal training data while maintaining stability, achieving best performance with M=5 training images.

- **Statistical Assumptions Remain Valid**: Despite extreme low-light conditions, the underlying statistical framework for achromatic pixel detection and illuminant estimation remains valid when properly adapted, avoiding the data hunger and generalization issues of end-to-end deep learning approaches.

## Key Data & Results

| Method | Dataset | Median (°) | Mean (°) | Worst-25% (°) | Notes |
|--------|---------|------------|----------|---------------|-------|
| SGP-LRD (Statistical) | NCC | 2.12 | 3.11 | 7.22 | State-of-the-art statistical |
| RL-AWB (Ours) | NCC | **1.98** | **3.07** | **7.22** | 5-shot learning-based |
| C5 (5-shot) | NCC | 5.56 | 7.11 | 14.66 | Standard deep learning baseline |
| RL-AWB | LEVI | **3.01** | **3.22** | **5.32** | Maintains performance on new dataset |
| C5 (5-shot) | LEVI | 2.46 | 3.50 | 7.80 | Struggles on new sensor |
| **Cross-dataset: NCC→LEVI** | | | | | |
| RL-AWB | NCC→LEVI | **3.03** | **3.24** | **5.36** | Minimal degradation |
| C5 (5-shot) | NCC→LEVI | 9.40 | 10.93 | 20.61 | Severe domain shift impact |
| **Daytime Generalization** | | | | | |
| RL-AWB | Gehler-Shi | **2.24** | **3.50** | **8.67** | Generalizes to well-lit scenes |

- **In-dataset performance**: RL-AWB achieves median angular error of 1.98° on NCC and 3.01° on LEVI, outperforming 5-shot learning-based baselines (C5: 5.56°, PCC: 4.65°) while using equivalent training data.

- **Cross-dataset robustness**: When trained on NCC and tested on LEVI, RL-AWB maintains median error of 3.03° compared to C5's degradation to 9.40°—a 3× improvement. The reverse direction (LEVI→NCC) shows similar robustness with 1.99° error versus C5's 11.38°.

- **Daytime generalization**: Despite being tailored for nighttime scenes, RL-AWB trained on NCC achieves median error of 2.24° on the Gehler-Shi daytime dataset, competitive with state-of-the-art methods and demonstrating unexpected versatility across illumination ranges.

- **Data efficiency**: Ablation study shows M=5 curriculum pool size is optimal (Table 4), with M<5 lacking diversity and M>15 reducing per-sample visitation. RL-AWB achieves competitive performance with only 5 training images compared to C5(full) using hundreds of images.

## Strengths

- **Novel RL Formulation for ISP**: First application of deep reinforcement learning to color constancy, framing AWB as sequential decision-making rather than direct regression, enabling per-image parameter adaptation that mimics professional AWB tuning.

- **Exceptional Few-Shot Learning**: Achieves state-of-the-art performance with only 5 training images per dataset, dramatically reducing data requirements compared to learning-based baselines that require extensive labeled nighttime data—addressing practical deployment constraints.

- **Robust Cross-Sensor Generalization**: Demonstrates 3× improvement in cross-dataset median error compared to learning-based methods (RL-AWB: 3.03° vs C5: 9.40°), validating the hypothesis that sensor-agnostic statistical algorithms combined with curriculum-learned RL policies are more robust than direct illuminant regression.

- **Well-Motivated Algorithm Design**: SGP-LRD incorporates three principled design elements (reliability amplification, implicit noise filtering, spatial prior exploitation) with two-layer filtering (variance and color deviation) explicitly addressing nighttime challenges like sensor noise and spurious color casts.

- **Comprehensive Evaluation**: Multi-faceted experimental validation including in-dataset, cross-dataset, cross-sensor, and daytime generalization evaluation, plus detailed ablations on curriculum pool size (M∈{3,5,7,9,15}) and RL algorithm choice (PPO vs SAC), and both recovery and reproduction angular error metrics.

## Weaknesses

- **Limited Parameter Space**: The framework controls only two algorithm parameters (gray pixel selection percentage N and Minkowski norm p), leaving other tunable parameters in SGP-LRD unexploited. The authors acknowledge naive expansion would substantially increase training complexity without exploring efficient multi-parameter control strategies.

- **Inconsistent Performance on LEVI**: While RL-AWB excels on NCC and cross-dataset evaluation, in-dataset LEVI results show C5(5) achieves 2.46° median versus RL-AWB's 3.01°, suggesting the method may not capture all relevant statistics for this particular sensor/scene distribution despite overall robustness.

- **Over-Correction on Challenging Scenes**: Authors acknowledge that while overall angular error is reduced, RL-AWB may over-correct a small number of challenging nighttime scenes, resulting in visually degraded outputs—a failure mode not thoroughly characterized quantitatively in ablations.

- **Computational Overhead Not Fully Specified**: Implementation details show separate CPU-based RL training and GPU-accelerated environment simulation (RTX 3080), but inference time comparisons with baselines are absent. Wall-clock training time and latency implications for real-time ISP pipelines are not discussed.

- **Limited Dataset Diversity**: LEVI dataset, while multi-camera, contains only 700 images from two sensors (iPhone 16 Pro and Sony ILCE-6400) with scenes captured in evening/nighttime. Generalization to other low-light scenarios (e.g., indoor lighting, artificial street lighting varieties) remains unvalidated.

## Potential Improvements

- **Hierarchical/Structured Multi-Parameter Control**: Investigate hierarchical policies or low-dimensional latent action representations to efficiently coordinate multiple ISP parameters beyond N and p, addressing the acknowledged limitation of current 2D action space without exponential complexity growth.

- **Safety-Aware Reward Formulation**: Develop constrained optimization strategies incorporating preference-based regularization or penalizing abrupt parameter changes to explicitly mitigate over-correction failure cases on challenging scenes, moving toward safety-aware RL for ISP applications.

- **GPU-Resident End-to-End Training**: Implement fully GPU-resident training pipeline with batched rollouts to reduce wall-clock training time and enable joint optimization across both nighttime and daytime data, facilitating a unified all-time AWB agent rather than separate models.

- **Adaptive Termination Criteria**: Current implementation uses fixed three-step stabilization criterion (Sec. 3.2). Explore learned termination policies conditioned on scene characteristics to allow scenes requiring fewer/more correction steps, potentially reducing both error variance and computational cost.

- **Extended Cross-Domain Evaluation**: Expand LEVI dataset with additional camera sensors and diverse low-light scenarios (industrial lighting, medical imaging, surveillance under various street lights) to validate generalization claims beyond the current two-camera setup and enable future benchmark comparisons.

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **SGP-LRD 演算法**：開發了一個專門針對夜景的色彩恆常性演算法，整合顯著灰色像素檢測與本地反射率差異，在公開的夜景基準（NCC 和 LEVI 資料集）上達成最優照度估計性能。

- **RL-AWB 框架**：設計了第一個用於色彩恆常性的深度強化學習方法，採用 Soft Actor-Critic (SAC) 訓練和兩階段課程學習，實現自適應逐影像參數最佳化，僅需 5 張訓練影像即可達成出色效能。

- **LEVI 資料集**：引入首個多感測器夜景色彩恆常性資料集，包含來自兩個相機（iPhone 16 Pro 和 Sony ILCE-6400）的 700 張線性 RAW 影像，支援嚴格的跨感測器泛化評估，ISO 範圍從 500 到 16,000。

- **優越的跨感測器泛化能力**：展示混合統計學習方法在不同相機感測器和資料集間具備顯著的魯棒性，在跨資料集評估中遠超純粹基於學習的方法。

## 核心洞見

- **混合設計優勢**：將統計演算法（可解釋和感測器不可知的）與強化學習（自適應能力）結合，保留演算法透明度同時獲得自適應能力，解決純資料驅動方法的嚴重泛化降解問題。

- **夜景中的參數敏感性**：夜間成像因低光噪聲和複雜光照條件違反傳統自動白平衡演算法的基本假設。場景相依的參數選擇（灰色像素百分比 N 和 Minkowski 範數 p）至關重要，使基於強化學習的動態調優優於固定參數。

- **課程學習帶來的資料效率**：兩階段課程（單影像穩定化後跟著多影像循環訓練）使強化學習代理能以最少訓練資料學習魯棒策略，同時保持穩定性，在 M=5 訓練影像時達到最優性能。

- **統計假設仍然有效**：儘管光線極低，但經過適當適應後，用於無色像素檢測和照度估計的底層統計框架仍然有效，避免了端到端深度學習的資料需求和泛化問題。

## 關鍵數據與結果

| 方法 | 資料集 | 中位數 (°) | 平均值 (°) | 最差 25% (°) | 備註 |
|------|--------|-----------|-----------|-------------|------|
| SGP-LRD (統計方法) | NCC | 2.12 | 3.11 | 7.22 | 最優統計方法 |
| RL-AWB (我們的方法) | NCC | **1.98** | **3.07** | **7.22** | 五樣本學習方法 |
| C5 (5-shot) | NCC | 5.56 | 7.11 | 14.66 | 標準深度學習基線 |
| RL-AWB | LEVI | **3.01** | **3.22** | **5.32** | 在新資料集上維持效能 |
| C5 (5-shot) | LEVI | 2.46 | 3.50 | 7.80 | 在新感測器上表現不佳 |
| **跨資料集：NCC→LEVI** | | | | | |
| RL-AWB | NCC→LEVI | **3.03** | **3.24** | **5.36** | 最小化的性能下降 |
| C5 (5-shot) | NCC→LEVI | 9.40 | 10.93 | 20.61 | 嚴重的域偏移影響 |
| **日間泛化** | | | | | |
| RL-AWB | Gehler-Shi | **2.24** | **3.50** | **8.67** | 泛化到光線充足的場景 |

- **資料集內性能**：RL-AWB 在 NCC 上達成 1.98° 的中位角度誤差，在 LEVI 上為 3.01°，優於 5 樣本學習基線（C5：5.56°，PCC：4.65°），同時使用等量的訓練資料。

- **跨資料集魯棒性**：當在 NCC 上訓練並在 LEVI 上測試時，RL-AWB 維持 3.03° 的中位誤差，相比 C5 下降到 9.40°——提升 3 倍。反向方向（LEVI→NCC）顯示相似的魯棒性，誤差為 1.99° 對比 C5 的 11.38°。

- **日間泛化**：儘管針對夜景調整，RL-AWB 在 NCC 上訓練後在 Gehler-Shi 日間資料集上達成 2.24° 的中位誤差，與最優方法相當，展示出在光照範圍間的意外多功能性。

- **資料效率**：消融研究顯示 M=5 課程池大小最優（表 4），M<5 缺乏多樣性，M>15 降低每樣本訪問次數。RL-AWB 僅用 5 張訓練影像達到競爭效能，相比使用數百張影像的 C5(full)。

## 優勢

- **新穎的強化學習公式化**：首次將深度強化學習應用於色彩恆常性，將自動白平衡框架化為序列決策問題而非直接迴歸，實現逐影像參數自適應以模仿專業自動白平衡調整專家。

- **卓越的少樣本學習**：僅用 5 張訓練影像達到最優效能，與需要大量標註夜景資料的學習方法相比，大幅減少資料需求——解決了實際部署的約束。

- **穩健的跨感測器泛化**：相比學習方法的跨資料集中位誤差提升 3 倍（RL-AWB：3.03° vs C5：9.40°），驗證了感測器不可知統計演算法與課程學習強化學習策略組合的假設優於直接照度迴歸。

- **設計充分論證的演算法**：SGP-LRD 包含三個原理性設計要素（可靠性放大、隱式噪聲濾除、空間先驗利用）與兩層濾除器（方差和顏色偏差）明確處理夜景挑戰如感測器噪聲和虛假色偏。

- **全面的評估**：多面向的實驗驗證包括資料集內、跨資料集、跨感測器與日間泛化評估，加上課程池大小（M∈{3,5,7,9,15}）和強化學習演算法選擇（PPO vs SAC）的詳細消融，以及恢復和再現角度誤差指標。

## 劣勢

- **有限的參數空間**：框架僅控制兩個演算法參數（灰色像素選擇百分比 N 和 Minkowski 範數 p），遺漏 SGP-LRD 中其他可調參數。作者承認天真擴展會大幅增加訓練複雜性，未探索有效的多參數控制策略。

- **LEVI 上的不一致性能**：雖然 RL-AWB 在 NCC 和跨資料集評估中表現優異，但資料集內 LEVI 結果顯示 C5(5) 達成 2.46° 中位數對比 RL-AWB 的 3.01°，表明儘管整體魯棒性，方法可能未捕捉該特定感測器/場景分佈的所有相關統計。

- **在困難場景上的過度修正**：作者承認雖然整體角度誤差降低，RL-AWB 可能在少數困難夜景上過度修正，導致視覺上降解的輸出——此失敗模式未在消融中充分量化刻畫。

- **計算開銷未完全指定**：實現細節顯示分離的 CPU 強化學習訓練和 GPU 加速環境模擬（RTX 3080），但與基線的推理時間比較缺失。牆鐘訓練時間和實時 ISP 管道的延遲含義未討論。

- **有限的資料集多樣性**：LEVI 資料集雖為多相機，但只含來自兩個感測器（iPhone 16 Pro 和 Sony ILCE-6400）700 張影像，場景限於晚間/夜間拍攝。泛化到其他低光場景（如室內光線、人工街道光源多樣性）仍未驗證。

## 可改進方向

- **階層式/結構化多參數控制**：研究階層式策略或低維潛在動作表示以有效協調 N 和 p 以外的多個 ISP 參數，不產生指數級複雜性增長而解決當前 2D 動作空間的限制。

- **安全感知獎勵公式**：開發納入偏好基礎正則化或懲罰急劇參數變化的約束最佳化策略以明確緩解過度修正失敗，朝向 ISP 應用的安全感知強化學習邁進。

- **完全 GPU 駐留端到端訓練**：實現帶批量展開的完全 GPU 駐留訓練管道以縮短牆鐘訓練時間並支援夜間與日間資料聯合最佳化，促進統一全時段自動白平衡代理而非分離模型。

- **自適應終止準則**：目前實現使用固定三步穩定化準則（3.2 節）。探索以場景特徵為條件的已學終止策略允許需要更少/更多修正步驟的場景，潛在降低誤差方差和計算成本。

- **擴展的跨域評估**：以額外相機感測器和多樣低光場景（工業光線、醫學成像、各種街道光線下監控）擴展 LEVI 資料集以驗證超越當前雙相機設置的泛化聲明，並支援未來基準比較。

</div>

---
