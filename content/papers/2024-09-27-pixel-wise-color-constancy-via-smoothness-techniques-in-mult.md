---
title: "Pixel-wise Color Constancy via Smoothness Techniques in Multi-Illuminant Scenes"
date: 2024-09-27
authors:
  - "Umut Cem Entok"
  - "Firas Laakom"
  - "Farhad Pakdaman"
  - "Moncef Gabbouj"
source: "Upload"
arxiv_url: ""
pdf_url: ""
one_line_summary: "This paper proposes a pixel-wise multi-illuminant color constancy method using U-Net with total variation regularization and bilateral filtering, achieving 13% improvement over state-of-the-art on the LSMI dataset."
one_line_summary_zh: "本文提出一種使用 U-Net 和全變差正則化的逐像素多光源顏色恆常性方法，在 LSMI 資料集上比最先進方法提升 13%。"
date_added: 2026-02-14
topics: ["Auto White Balance"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **Pixel-wise Illumination Estimation via U-Net**: The paper proposes a deep learning approach to estimate pixel-wise illumination maps in multi-illuminant scenes using a U-Net architecture. Unlike earlier patch-wise methods that fail to capture spatial dependencies between patches, this approach directly predicts per-pixel illumination variations caused by multiple light sources, enabling more granular white-balance correction across the image.

- **Total Variation Loss for Smoothness Regularization**: A key novelty is the integration of Total Variation (TV) loss during training to enforce smooth illumination changes between neighboring pixels. This addresses a critical limitation in prior work—most existing multi-illuminant methods neglect the fact that illuminants change gradually across natural images. The TV loss regularizes the estimated illumination map such that spatial discontinuities are penalized, resulting in physically plausible and visually coherent estimations.

- **Bilateral Filtering as Post-processing**: The method applies bilateral filtering as a post-processing step to enhance visual quality while preserving edges and spatial consistency. With parameters σ_s and σ_r both set to 75 using a 9-pixel diameter kernel, this denoising technique maintains photorealism without introducing the spatial artifacts observed in GAN-based competitors, particularly addressing the problem that recent GAN approaches (e.g., AngularGAN) produce visually unpleasant results.

- **Label-Smoothing for Noisy Ground Truth**: To handle inaccuracies in the LSMI dataset's ground truth (which omits surface reflectivity information), the paper introduces a label-smoothing technique that injects Gaussian noise into the alpha parameter α with standard deviation σ_n = α_raw/10. This prevents overfitting to noisy labels and improves generalization without requiring re-annotation of training data.

- **Comprehensive Evaluation on LSMI Dataset**: The work conducts thorough experimental validation on the Large Scale Multi-Illuminant (LSMI) dataset, using 2,360 images captured under two illuminants. Two variants (PWCC v1 and PWCC v2) are proposed with different hyperparameters (λ_TV and learning rates), demonstrating systematic ablation and providing reproducible baselines for future research.

## Core Insights

- **Spatial Smoothness is Crucial for Multi-Illuminant Scenes**: The fundamental insight driving this work is that illumination in natural scenes exhibits spatial continuity—illuminants change smoothly across pixels rather than abruptly. This observation, grounded in physical principles, was largely overlooked by prior deep learning approaches that focused purely on pixel-level prediction accuracy without enforcing smoothness priors. By incorporating TV regularization, the method achieves significantly better worst-case performance (25% worst improvement: 4.24° → 3.8°), indicating that smoothness constraints particularly help on difficult cases.

- **Ground Truth Noise is a Practical Challenge**: The observation that the LSMI dataset lacks surface reflectivity information reveals a critical practical issue in multi-illuminant research. Rather than dismissing the dataset or pursuing perfect ground truth (which may be infeasible), the paper pragmatically applies label smoothing to acknowledge and mitigate this noise. This approach improves mean error from 2.31° (baseline) to 2.0° (PWCC v2), showing that explicit handling of label uncertainty provides measurable benefits even without addressing the root cause.

- **Bilateral Filtering Preserves Details While Reducing Artifacts**: The post-processing bilateral filter tackles a specific weakness of GAN-based approaches, which often introduce unnatural spatial artifacts. By independently filtering the estimated image using both spatial proximity (σ_s) and intensity similarity (σ_r), the method achieves edge-preserving smoothing. Qualitative comparisons demonstrate this advantage, particularly in cases where the baseline U-Net over-compensates for color casts (e.g., removing natural colors from walls), while the proposed method maintains visual fidelity.

- **Hyperparameter Sensitivity Reveals TV Loss Trade-offs**: The two proposed variants (PWCC v1 with λ_TV=2e-4 and PWCC v2 with λ_TV=2e-3, using different learning rates) show that increasing TV loss weight moderately improves performance (2.08° → 2.0° mean error). However, the paper does not provide detailed ablation studies on the exact impact of λ_TV alone, suggesting potential for further tuning. This indicates that smooth regularization is beneficial but must be balanced against other loss components.

- **Recovery Angular Error is Intensity-Independent**: The choice to evaluate using recovery angular error rather than raw RGB error is justified because it measures the directionality of illumination estimation, independent of intensity magnitude. This metric appropriately reflects the physical properties of color constancy, where the relationship between light source colors matters more than absolute magnitude. The 13% improvement (2.31° → 2.0°) is substantial in this well-established evaluation framework.

## Key Data & Results

| Method | Mean (°) | Median (°) | Worst 25% (°) | Best 25% (°) |
|--------|----------|-----------|---------------|-------------|
| Gray World [26] | 11.3 | 8.8 | 20.74 | 4.93 |
| White Patch [27] | 12.8 | 14.3 | 23.49 | 5.6 |
| LSMI U-Net [19] | 2.31 | 1.91 | 4.24 | 1.01 |
| PWCC v1 | 2.08 | 1.74 | 3.8 | 0.9 |
| PWCC v2 | 2.0 | 1.7 | 3.8 | 0.86 |

- **13% Improvement Over State-of-the-Art**: PWCC v2 achieves a mean recovery angular error of 2.0°, compared to 2.31° for the baseline LSMI U-Net, representing a 13% improvement. This improvement is consistent across metrics: median error improves from 1.91° to 1.7°, and worst 25% error remains at 3.8° (matching v1), indicating robustness on difficult cases.

- **Significant Worst-Case Performance Gain**: The worst 25% improvement from 4.24° (baseline) to 3.8° (proposed) is particularly noteworthy, as corner cases often dominate user-perceived quality. This suggests the smoothness constraint via TV loss is especially effective at preventing estimation failures on atypical images, addressing a practical pain point in deployment scenarios.

- **Label Smoothing Provides Measurable Benefit**: Comparing PWCC v1 (no label smoothing, λ_TV=2e-4) at 2.08° mean error to PWCC v2 (with label smoothing, λ_TV=2e-3) at 2.0° shows that the label-smoothing technique contributes approximately 0.08° improvement when combined with higher TV regularization weight. While modest in absolute terms, this validates the theoretical motivation for handling label noise.

- **Classical Methods Show Significantly Worse Performance**: The stark gap between classical methods (Gray World: 11.3°, White Patch: 12.8°) and learning-based approaches (LSMI U-Net: 2.31°, PWCC v2: 2.0°) underscores the transformative impact of deep learning for color constancy. The 5-6x error reduction justifies the computational overhead of neural network inference.

- **Qualitative Evaluation Shows Competitive or Superior Results**: Visual comparisons on three example images demonstrate that PWCC consistently outperforms the baseline, particularly in preventing over-compensation artifacts. The baseline exhibits greenish color cast in one example and removes natural wall colors in another, while the proposed method achieves more natural-looking results with better color preservation.

## Strengths

- **Well-Motivated Technical Approach**: The paper clearly identifies a limitation in existing multi-illuminant methods—the failure to preserve smooth illumination changes—and proposes a principled solution via total variation regularization. This motivation is grounded in physics (illuminants change gradually in natural scenes) and prior work in inverse problems, making the approach theoretically sound. The connection between spatial smoothness and improved performance is compelling.

- **Addresses a Practical Limitation of Ground Truth Data**: Rather than ignoring known inaccuracies in the LSMI dataset (missing surface reflectivity), the authors propose a pragmatic label-smoothing solution. This demonstrates awareness of real-world challenges in dataset construction and shows good research practice by acknowledging and mitigating limitations rather than dismissing them. The technique is simple yet effective.

- **Comprehensive Technical Contributions**: The paper combines three complementary techniques (TV loss for training, bilateral filtering for post-processing, label smoothing for noise robustness) into a cohesive framework. The ablation through two variants (PWCC v1 and v2) demonstrates the combined effect, though more granular ablations would strengthen this further.

- **Clear Presentation and Mathematical Formulation**: The paper is well-written with clear mathematical notation, explicit loss function definitions (equations 11-13), and a helpful pipeline diagram (Figure 1). The method is presented clearly enough that reproduction appears feasible, with specific hyperparameter values provided in Table 1.

- **Consistent Improvements Across Metrics**: The proposed method improves not just the mean error but also median, worst-25%, and best-25% error metrics, indicating robust and generalized improvements rather than cherry-picked results. The improvement in worst-case performance is particularly valuable for practical applications.

- **Suitable Evaluation Framework**: Using recovery angular error as the primary metric is appropriate, as this intensity-independent measure better reflects the true nature of color constancy problems. Comparisons include relevant classical and recent learning-based baselines on the standard LSMI benchmark.

## Weaknesses

- **Limited Ablation Study**: While the paper compares PWCC v1 and v2 (differing in label smoothing and λ_TV), it lacks detailed ablation studies isolating the individual contributions of TV loss, bilateral filtering, and label smoothing. The authors should present results showing (a) baseline U-Net only, (b) U-Net + TV loss, (c) U-Net + TV loss + bilateral filtering, and (d) full method with label smoothing. Without this breakdown, it is unclear which components drive the 13% improvement.

- **Narrow Dataset Evaluation**: The experimental evaluation focuses exclusively on 2,360 two-illuminant images from a single camera (Samsung Galaxy Note 20 Ultra) in the LSMI dataset. The paper does not evaluate on: (1) images with 3+ illuminants despite LSMI containing such data, (2) other public datasets or real-world collections, or (3) cross-dataset generalization. This limits confidence in the generalizability of the method to diverse capture conditions.

- **Insufficient Analysis of Label Smoothing Design**: The label-smoothing implementation injects Gaussian noise with σ_n = α_raw/10 (equation 8), but no justification or sensitivity analysis is provided for this specific constant (wn = 10). Why 10 and not 5 or 20? How does performance vary with different smoothing strengths? Figure 2 visualizes the effect but provides no quantitative validation that this particular noise level is optimal.

- **Incomplete Comparison with State-of-the-Art**: The paper compares primarily against LSMI U-Net (2019) and classical methods but omits comparisons with other recent multi-illuminant approaches mentioned in related work. Reference [9] (Mimt, 2023) and [38] (N-white balancing, 2022) are cited but not compared quantitatively. Including comparisons with these more recent methods would better establish the contribution's significance.

- **Limited Discussion of Failure Cases and Limitations**: While Figure 3 shows that PWCC and baseline have "competing performances" in the second row, no detailed analysis explains why the method struggles in specific scenarios. Are there illumination configurations (e.g., very sharp transitions, grazing light) where TV regularization becomes problematic? What are the failure modes?

- **Hyperparameter Sensitivity Not Thoroughly Explored**: The bilateral filter parameters (σ_s=75, σ_r=75, 9-pixel diameter) are stated as "optimized" but no sensitivity analysis or ablation is provided. Similarly, the learning rate decay (factor 800, starting at epoch 800) appears arbitrary. The paper would benefit from ablation on these design choices to justify the specific parameter selections.

- **Computational Cost and Inference Speed Not Discussed**: The paper mentions training on 4 NVIDIA GeForce RTX 2080 Ti GPUs for 2000 epochs but provides no information about: (1) total training time, (2) inference speed per image, or (3) memory requirements. For practical deployment, these metrics are crucial and should be reported.

- **Ground Truth Noise Acknowledgment But Not Fully Addressed**: While label smoothing mitigates noisy ground truth, the paper does not propose a long-term solution (e.g., acquiring better ground truth with surface reflectivity) or explore alternative evaluation approaches (e.g., perceptual metrics independent of ground truth). The reliance on noisy labels remains a fundamental limitation acknowledged but not resolved.

## Research Directions

- **Multi-Illuminant Extension to 3+ Light Sources**: Extend the method to images with three or more illuminants, which LSMI contains but the current evaluation ignores. Investigate whether TV regularization remains effective with higher illuminant counts, and whether the per-pixel formulation scales to more complex lighting. This would require: (1) filtering LSMI for 3-illuminant images, (2) modifying the ground truth model (equation 4) to sum over more illuminants, and (3) ablating whether TV loss remains appropriate with increased complexity. Success here would yield a more generalizable method and unlock evaluation on a larger subset of LSMI.

- **Learnable Smoothness Regularization**: Instead of hand-tuning λ_TV and bilateral filter parameters, develop an adaptive or learned smoothness prior that adjusts regularization strength based on image content. For example, use meta-learning to optimize λ_TV per-image or learn a task-specific edge-preserving kernel for bilateral filtering. This would address the observation that optimal smoothness varies—some images may have sharper illumination transitions—and could improve robustness without manual hyperparameter search.

- **Physics-Informed Network Design**: Incorporate physical constraints more explicitly into the architecture, such as enforcing the von Kries model directly in the network or using physically-motivated loss functions that account for spectral properties of light sources. Explore whether incorporating priors on typical illuminant spectral distributions (e.g., daylight, tungsten) improves generalization, particularly when training data is limited or imbalanced across illuminant types.

- **Cross-Dataset Generalization and Domain Adaptation**: Develop and evaluate the method on datasets beyond LSMI (e.g., synthetic datasets like Rendered Multi-Illuminant Images, or different real-world captured data). Investigate domain adaptation strategies to enable transfer learning from large synthetic datasets to real camera captures. This research direction is critical for practical deployment and would demonstrate whether the method's improvements generalize or are dataset-specific.

- **Perceptual and User-Centric Evaluation**: Complement quantitative metrics (recovery angular error) with perceptual studies and user preferences. Conduct human evaluation comparing the proposed method's outputs with baselines to assess visual quality, naturalness, and acceptability. Explore perceptual metrics (LPIPS, SSIM, color difference ΔE) as alternatives or supplements to angular error, which may not fully capture user satisfaction.

- **Theoretical Analysis of TV Regularization for Multi-Illuminant Estimation**: Provide theoretical analysis or proofs regarding when and why TV regularization improves multi-illuminant estimation. Characterize the conditions under which smoothness assumptions hold (e.g., illuminants vary with low spatial frequency) and cases where they may fail. This mathematical rigor would strengthen the contribution's foundation and guide future improvements.

- **Joint Estimation of Illumination and Surface Reflectivity**: Extend the approach to jointly estimate both per-pixel illumination maps and surface reflectivity, addressing the acknowledged limitation that LSMI ground truth lacks reflectivity information. This could involve: (1) augmenting the network to output both illumination and reflectivity, (2) developing appropriate loss functions and constraints, and (3) creating or using datasets with complete ground truth. Success would eliminate the need for label smoothing as a workaround and provide a principled solution to ground truth noise.

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **基於 U-Net 的逐像素光照估計**：本文提出了一種深度學習方法，使用 U-Net 架構估計多光源場景中的逐像素光照映射圖。與早期的分塊方法不同，這種方法無法捕捉分塊之間的空間依賴性，此方法直接預測由多個光源造成的逐像素光照變化，從而能夠在整個影像中進行更精細的白平衡校正。

- **利用全變差損失函數進行平滑性正則化**：這項工作的核心創新在於在訓練期間整合全變差 (TV) 損失函數以強制相鄰像素之間的光照變化保持平滑。此方法解決了先前工作中的一個關鍵限制—大多數現有的多光源方法忽視了自然影像中光照逐漸變化的事實。TV 損失對估計的光照映射進行正則化，使得空間不連續性受到懲罰，從而產生物理上合理且視覺上一致的估計。

- **雙邊濾波作為後處理步驟**：該方法在後處理中應用雙邊濾波以增強視覺品質，同時保留邊緣和空間一致性。參數 σ_s 和 σ_r 均設置為 75，使用 9 像素直徑的核，該去噪技術保持了相片級真實感，而不會引入基於 GAN 的競爭對手所觀察到的空間偽影，特別是解決了最近 GAN 方法（例如 AngularGAN）產生視覺上不愉快結果的問題。

- **用於處理雜訊地面真值的標籤平滑**：為了處理 LSMI 資料集中地面真值的不準確性（未考慮表面反射率），本文引入了一種標籤平滑技術，向 alpha 參數 α 注入高斯噪聲，標準差 σ_n = α_raw/10。此方法防止對雜訊標籤的過擬合，並改善泛化性能，而無需重新標註訓練資料。

- **在 LSMI 資料集上進行全面評估**：本文在大規模多光源 (LSMI) 資料集上進行了徹底的實驗驗證，使用了 2,360 張在兩個光源下捕獲的影像。提出了兩種變體（PWCC v1 和 PWCC v2），具有不同的超參數（λ_TV 和學習率），展示了系統的消融研究，並為未來的研究提供了可復現的基線。

## 核心洞見

- **空間平滑性對多光源場景至關重要**：驅動此項工作的基本洞見是，自然場景中的光照表現出空間連續性—光照變化是平滑的，而不是突然的。這一觀察基於物理原理，在很大程度上被先前的深度學習方法所忽視，這些方法專注於逐像素預測精度，而未施加平滑性先驗。通過整合 TV 正則化，該方法在最壞情況下的性能上實現了顯著改善（25% 最差改善：4.24° → 3.8°），表明平滑性約束特別有助於解決困難案例。

- **地面真值噪聲是一個實踐挑戰**：LSMI 資料集缺乏表面反射率信息的觀察揭示了多光源研究中的關鍵實踐問題。與其忽視資料集或追求完美的地面真值（這可能不可行）相反，本文實際地應用標籤平滑來承認和緩解這種噪聲。此方法將平均誤差從 2.31°（基線）改善到 2.0°（PWCC v2），表明即使未解決根本原因，顯式處理標籤不確定性也能提供可測量的益處。

- **雙邊濾波保留細節同時減少偽影**：後處理雙邊濾波解決了基於 GAN 方法的一個特定弱點，這些方法經常引入不自然的空間偽影。通過使用空間接近度 (σ_s) 和強度相似度 (σ_r) 獨立地過濾估計影像，該方法實現了邊緣保留平滑。定性比較展示了這一優勢，特別是在基線 U-Net 過度補償顏色偏移的情況下（例如移除牆壁上的自然顏色），而提議的方法保持了視覺保真度。

- **超參數敏感性揭示 TV 損失的權衡**：兩個提議的變體（PWCC v1 帶 λ_TV=2e-4 和 PWCC v2 帶 λ_TV=2e-3，使用不同的學習率）顯示增加 TV 損失權重可適度改善性能（2.08° → 2.0° 平均誤差）。然而，本文未提供關於 λ_TV 單獨影響的詳細消融研究，表明進一步調整的潛力。這表明平滑正則化是有益的，但必須與其他損失分量保持平衡。

- **恢復角誤差是強度無關的**：選擇使用恢復角誤差而非原始 RGB 誤差進行評估是有合理的，因為它測量光照估計的方向性，與強度大小無關。此度量適當地反映了顏色恆常性的物理特性，其中光源顏色之間的關係比絕對大小更重要。13% 的改善（2.31° → 2.0°）在此既定的評估框架中是實質性的。

## 關鍵數據與結果

| 方法 | 平均 (°) | 中位數 (°) | 最差 25% (°) | 最佳 25% (°) |
|--------|----------|-----------|---------------|-------------|
| Gray World [26] | 11.3 | 8.8 | 20.74 | 4.93 |
| White Patch [27] | 12.8 | 14.3 | 23.49 | 5.6 |
| LSMI U-Net [19] | 2.31 | 1.91 | 4.24 | 1.01 |
| PWCC v1 | 2.08 | 1.74 | 3.8 | 0.9 |
| PWCC v2 | 2.0 | 1.7 | 3.8 | 0.86 |

- **相比最先進方法提升 13%**：PWCC v2 達到 2.0° 的平均恢復角誤差，而基線 LSMI U-Net 為 2.31°，代表 13% 的改善。此改善在各項指標上保持一致：中位數誤差從 1.91° 改善到 1.7°，最差 25% 誤差保持在 3.8°（與 v1 匹配），表明在困難案例上的穩健性。

- **最壞情況下性能顯著提升**：最差 25% 的改善（從基線 4.24° 到提議的 3.8°）特別值得關注，因為邊角案例通常主導使用者感知的品質。這表明通過 TV 損失的平滑性約束特別有效於防止在非典型影像上的估計失敗，解決了部署場景中的實踐痛點。

- **標籤平滑提供可測量的益處**：比較 PWCC v1（無標籤平滑，λ_TV=2e-4）在 2.08° 的平均誤差與 PWCC v2（帶標籤平滑，λ_TV=2e-3）在 2.0° 時，顯示當與更高的 TV 正則化權重結合時，標籤平滑技術貢獻約 0.08° 的改善。雖然絕對值較小，但這驗證了處理標籤噪聲的理論動機。

- **傳統方法表現明顯更差**：傳統方法（Gray World：11.3°，White Patch：12.8°）與基於學習的方法（LSMI U-Net：2.31°，PWCC v2：2.0°）之間的巨大差距凸顯了深度學習對顏色恆常性的變革性影響。5-6 倍的誤差減少證實了神經網路推理計算開銷的合理性。

- **定性評估顯示競爭性或優越的結果**：在三個示例影像上的視覺比較表明 PWCC 始終優於基線，特別是在防止過度補償偽影方面。基線在一個示例中呈現綠色偏色，在另一個示例中移除了自然的牆色，而提議的方法實現了更自然的外觀，具有更好的顏色保留。

## 優勢

- **動機充分的技術方法**：本文清楚地識別了現有多光源方法中的一個限制—未能保持平滑光照變化—並通過全變差正則化提出了一個有原則的解決方案。此動機基於物理原理（自然場景中光照逐漸變化）和反演問題中的先前工作，使該方法在理論上是健全的。空間平滑性與改善性能之間的聯繫很有說服力。

- **解決地面真值資料的實踐限制**：與其忽視 LSMI 資料集中已知的不準確性（缺少表面反射率），作者提出了一個實用的標籤平滑解決方案。這展示了對資料集構建中實際挑戰的認識，並通過承認和緩解限制而非駁回它們展示了良好的研究實踐。此技術簡單但有效。

- **全面的技術貢獻**：本文將三種互補的技術（用於訓練的 TV 損失、用於後處理的雙邊濾波、用於噪聲穩健性的標籤平滑）結合成一個連貫的框架。透過兩個變體（PWCC v1 和 v2）的消融展示了組合效果，儘管更細化的消融會進一步加強這一點。

- **清晰的演示和數學公式化**：本文撰寫清晰，具有明確的數學符號、明確的損失函數定義（方程 11-13）和有用的管道圖（圖 1）。方法呈現足夠清晰以便能夠復現，具有表 1 中提供的特定超參數值。

- **在各項指標上的一致改善**：提議的方法不僅改善了平均誤差，而且改善了中位數、最差 25% 和最佳 25% 誤差指標，表明穩健和普遍化的改善而非精選結果。在最壞情況下性能的改善對實踐應用特別寶貴。

- **適當的評估框架**：使用恢復角誤差作為主要度量是合適的，因為這種強度無關的度量更好地反映了顏色恆常性問題的真實性質。比較包括在標準 LSMI 基準上的相關古典和最近的基於學習的基線。

## 劣勢

- **消融研究受限**：雖然本文比較了 PWCC v1 和 v2（在標籤平滑和 λ_TV 上不同），但缺乏詳細的消融研究來孤立 TV 損失、雙邊濾波和標籤平滑的個別貢獻。作者應呈現結果顯示 (a) 僅基線 U-Net，(b) U-Net + TV 損失，(c) U-Net + TV 損失 + 雙邊濾波，和 (d) 帶標籤平滑的完整方法。沒有此分解，不清楚哪些組件驅動了 13% 的改善。

- **狹窄的資料集評估**：實驗評估專門集中在來自單一相機（Samsung Galaxy Note 20 Ultra）的 LSMI 資料集中的 2,360 張雙光源影像。本文未評估：(1) 儘管 LSMI 包含此類資料，具有 3 個或更多光源的影像，(2) 其他公共資料集或實際收集，或 (3) 跨資料集泛化。這限制了對該方法在不同捕獲條件下的泛化能力的信心。

- **標籤平滑設計分析不足**：標籤平滑實現注入高斯噪聲，標準差 σ_n = α_raw/10（方程 8），但未為此特定常數（wn = 10）提供合理化或敏感性分析。為什麼是 10 而不是 5 或 20？性能如何隨著不同平滑強度而變化？圖 2 將效果視覺化，但未提供定量驗證表明此特定噪聲級別是最優的。

- **與最先進技術的比較不完整**：本文主要與 LSMI U-Net（2019）和古典方法進行比較，但省略了與相關工作中提及的其他最近多光源方法的比較。參考文獻 [9]（Mimt，2023）和 [38]（N-white balancing，2022）被引用但未進行定量比較。包括與這些更最近方法的比較將更好地確立貢獻的重要性。

- **失敗案例和限制的討論受限**：儘管圖 3 顯示 PWCC 和基線在第二行具有「競爭性能」，但未詳細分析該方法在特定場景中的掙扎原因。是否存在光照配置（例如非常尖銳的過渡、掠過光）使 TV 正則化變得有問題的情況？失敗模式是什麼？

- **超參數敏感性未充分探索**：雙邊濾波參數（σ_s=75，σ_r=75，9 像素直徑）被陳述為「最優化」，但未提供敏感性分析或消融。類似地，學習率衰減（因子 800，從第 800 個 epoch 開始）似乎是任意的。本文將從這些設計選擇的消融中受益，以證實特定參數選擇的正當性。

- **計算成本和推理速度未討論**：本文提到在 4 個 NVIDIA GeForce RTX 2080 Ti GPU 上訓練 2000 個 epoch，但未提供以下信息：(1) 總訓練時間，(2) 每張影像的推理速度，或 (3) 記憶體需求。對於實踐部署，這些指標至關重要，應當報告。

- **地面真值噪聲認知但未完全解決**：雖然標籤平滑減輕了雜訊地面真值，本文未提出長期解決方案（例如獲取包含表面反射率的更好地面真值）或探索替代評估方法（例如獨立於地面真值的感知指標）。對雜訊標籤的依賴仍然是一個根本限制，已被認可但未被解決。

## 研究方向

- **多光源擴展到 3 個或更多光源**：將該方法擴展到具有三個或更多光源的影像，LSMI 包含但當前評估忽視此類影像。調查 TV 正則化是否隨著光源數量增加而保持有效，以及逐像素公式是否擴展到更複雜的光照。這將需要：(1) 過濾 LSMI 以尋找 3 光源影像，(2) 修改地面真值模型（方程 4）以對更多光源求和，和 (3) 消融是否 TV 損失隨著複雜性增加而保持適當。在這裡的成功將產生一個更通用的方法，並解鎖對 LSMI 更大子集的評估。

- **可學習的平滑性正則化**：與其手動調整 λ_TV 和雙邊濾波參數，開發一個自適應或可學習的平滑性先驗，根據影像內容調整正則化強度。例如，使用元學習為每張影像優化 λ_TV，或為雙邊濾波學習一個任務特定的邊緣保留核。這將解決最優平滑性隨影像變化的觀察—某些影像可能具有更尖銳的光照過渡—並可在沒有手動超參數搜索的情況下改善穩健性。

- **物理信息化網路設計**：將物理約束更明確地整合到架構中，例如直接在網路中強制 von Kries 模型或使用考慮光源光譜特性的物理啟發損失函數。探索是否整合典型光照光譜分佈的先驗（例如日光、鎢光）改善泛化，特別是當訓練資料有限或在光照類型上不平衡時。

- **跨資料集泛化和領域適應**：在 LSMI 以外的資料集上開發和評估該方法（例如，合成資料集如渲染多光源影像，或不同的實際捕獲資料）。調查領域適應策略以實現從大型合成資料集到實際相機捕獲的遷移學習。此研究方向對實踐部署至關重要，並將展示該方法的改善是否泛化或是資料集特定的。

- **感知和使用者中心評估**：用感知研究和使用者偏好補充定量指標（恢復角誤差）。進行人類評估比較提議方法的輸出與基線，以評估視覺品質、自然度和可接受性。探索感知指標（LPIPS、SSIM、顏色差異 ΔE）作為角誤差的替代或補充，其可能未充分捕捉使用者滿意度。

- **多光源估計的 TV 正則化理論分析**：提供關於何時以及為什麼 TV 正則化改善多光源估計的理論分析或證明。刻劃平滑性假設成立的條件（例如光照以低空間頻率變化）以及假設可能失敗的案例。此數學嚴謹性將加強貢獻的基礎並指導未來的改善。

- **光照和表面反射率的聯合估計**：擴展該方法以聯合估計逐像素光照映射和表面反射率，解決已承認的限制，即 LSMI 地面真值缺乏反射率信息。這可能涉及：(1) 擴展網路以輸出光照和反射率，(2) 開發適當的損失函數和約束，和 (3) 創建或使用具有完整地面真值的資料集。成功將消除需要標籤平滑作為解決方案，並為地面真值噪聲提供原則性的解決方案。

</div>

