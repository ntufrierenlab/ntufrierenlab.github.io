---
title: "Simple Baselines for Image Restoration"
date: 2022-04-10
authors:
  - "Liangyu Chen"
  - "Xiaojie Chu"
  - "Xiangyu Zhang"
  - "Jian Sun"
source: "arXiv"
arxiv_url: "https://arxiv.org/abs/2204.04676"
pdf_url: "https://arxiv.org/pdf/2204.04676"
one_line_summary: "NAFNet demonstrates that nonlinear activation functions are unnecessary for state-of-the-art image restoration, achieving superior performance to existing methods with dramatically lower computational cost through simplified architecture design."
one_line_summary_zh: "NAFNet 通過簡化的架構設計證明非線性激活函數對於最先進圖像復原並非必要，以遠低於現有方法的計算成本實現更優性能。"
date_added: 2026-03-09
topics: ["Image Deblurring"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **Simple Yet Powerful Baseline Architecture**: The paper proposes a straightforward image restoration baseline that surpasses state-of-the-art methods while being computationally efficient. Built on a single-stage UNet architecture, the baseline combines only essential components—Layer Normalization, GELU activation, Channel Attention, and depthwise convolution—and achieves 40.30 dB PSNR on SIDD and 33.40 dB on GoPro, exceeding previous SOTA results with a fraction of the computational cost. This demonstrates that escalating system complexity (both inter-block and intra-block) is unnecessary for achieving competitive performance, challenging the prevailing design philosophy in image restoration.

- **Nonlinear Activation Free Network (NAFNet)**: The paper's primary contribution is revealing that nonlinear activation functions (Sigmoid, ReLU, GELU, Softmax) can be completely removed from the network without performance degradation. By identifying that GELU can be viewed as a special case of Gated Linear Units (GLU) and that element-wise multiplication of feature maps (SimpleGate) provides sufficient nonlinearity, the authors derive NAFNet, which achieves equal or better performance than the baseline despite being further simplified. This is framed as potentially the first work demonstrating that nonlinear activations may not be necessary for SOTA computer vision methods.

- **Connection Between GELU, Channel Attention, and GLU**: The paper provides theoretical insights by establishing formal connections between existing components and GLU. It shows that GELU(x) = x·Φ(x) is a special case of GLU (where f and g are identity functions), and that Channel Attention can be reformulated similarly to GLU. Based on these observations, the authors propose Simplified Channel Attention (SCA), replacing the original CA's nonlinear activations with a simple linear transformation, reducing computational complexity while maintaining performance.

- **Comprehensive Empirical Validation Across Multiple Tasks**: The paper demonstrates the effectiveness of the proposed baselines across diverse image restoration tasks: RGB image denoising (SIDD), image deblurring (GoPro), raw image denoising (4Scenes), and image deblurring with JPEG artifacts (REDS). On GoPro, NAFNet achieves 33.69 dB PSNR with only 8.4% of the computational cost of previous SOTA (MPRNet-local), demonstrating both superior performance and dramatic efficiency gains across multiple benchmarks.

- **Systematic Design Space Exploration with Extensive Ablation Studies**: The paper rigorously validates each design choice through detailed ablation experiments on SIDD and GoPro datasets. Table 1 shows Layer Normalization contributes +0.46 dB (SIDD) and +3.39 dB (GoPro), Channel Attention adds +0.14-0.24 dB, and Table 2 demonstrates SimpleGate and SCA replacements yield consistent improvements or neutral effects. This methodical approach makes the paper valuable as a benchmark for future research, facilitating easier verification of new ideas.

- **Released Code and Pre-trained Models**: The authors release code and pre-trained models at github.com/megvii-research/NAFNet, significantly enhancing reproducibility and enabling broader adoption. This contribution is important for establishing NAFNet as a practical baseline that researchers can build upon.

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **簡潔高效的基準模型架構**：論文提出了一個簡明有效的圖像復原基準模型，其性能超越最先進方法，同時計算效率更高。該基準模型基於單階段 UNet 架構，僅結合必要的組件——層歸一化（Layer Normalization）、GELU 激活、通道注意力（Channel Attention）和深度卷積——在 SIDD 上達到 40.30 dB PSNR，在 GoPro 上達到 33.40 dB，超越先前最優方法且計算成本僅為其零頭。這表明增加系統複雜度（包括塊間和塊內複雜度）並非實現競爭性性能的必要條件，挑戰了圖像復原領域普遍的設計哲學。

- **無非線性激活函數網絡（NAFNet）**：論文的核心貢獻是揭示非線性激活函數（Sigmoid、ReLU、GELU、Softmax）可以完全從網絡中移除而不影響性能。通過發現 GELU 可視為門控線性單元（GLU）的特殊情況，以及特徵圖的逐元素相乘（SimpleGate）提供了足夠的非線性，作者推導出 NAFNet，其性能與基準模型相當或更優，儘管進一步簡化。據論文所知，這是首項證明非線性激活可能對最先進計算機視覺方法不必要的工作。

- **GELU、通道注意力與 GLU 的連接**：論文通過建立現有組件與 GLU 之間的形式化連接提供了理論洞見。它表明 GELU(x) = x·Φ(x) 是 GLU 的特殊情況（其中 f 和 g 是恆等函數），並且通道注意力可以以類似 GLU 的方式重新表述。基於這些觀察，作者提出簡化通道注意力（SCA），將原始 CA 中的非線性激活替換為簡單線性變換，降低計算複雜度同時保持性能。

- **跨多個任務的全面實證驗證**：論文在多項圖像復原任務中展示了所提基準模型的有效性：RGB 圖像去噪（SIDD）、圖像去模糊（GoPro）、原始圖像去噪（4Scenes）和帶 JPEG 偽跡的圖像去模糊（REDS）。在 GoPro 上，NAFNet 達到 33.69 dB PSNR，計算成本僅為先前最優方法（MPRNet-local）的 8.4%，表現出跨多個基準的優越性能和顯著的效率提升。

- **系統化設計空間探索及全面的消融研究**：論文通過在 SIDD 和 GoPro 數據集上進行詳細的消融實驗嚴格驗證每個設計選擇。表 1 顯示層歸一化貢獻 +0.46 dB（SIDD）和 +3.39 dB（GoPro），通道注意力增加 +0.14-0.24 dB，表 2 展示 SimpleGate 和 SCA 替換產生一致的改進或中立效果。這種系統化方法使論文成為未來研究的寶貴基準，便於對新想法的驗證。

- **開放代碼和預訓練模型**：作者在 github.com/megvii-research/NAFNet 發佈代碼和預訓練模型，顯著增強了可復現性並促進更廣泛的應用。此貢獻對於建立 NAFNet 作為實用基準供研究人員構建其上而言至關重要。

</div>

<div class="lang-en">

## Core Insights

- **Nonlinearity Through Multiplication, Not Activation Functions**: The paper's fundamental insight is that nonlinearity can be achieved through the product of two linear transformations without requiring explicit nonlinear activation functions. This stems from the mathematical observation that SimpleGate(X, Y) = X ⊙ Y (element-wise multiplication) produces nonlinearity because the product operation itself is inherently nonlinear. This challenges the conventional wisdom that deep networks require nonlinear activations after every linear transformation. Table 2 shows that replacing GELU with SimpleGate yields +0.08 dB on SIDD and +0.41 dB on GoPro, confirming this substitution is not just viable but beneficial.

- **Layer Normalization is Critical for Training Stability**: The dramatic performance improvement from adding Layer Normalization (LN) is striking: +3.39 dB on GoPro when combined with a 10× learning rate increase. This suggests that LN not only stabilizes training but fundamentally changes the optimization landscape, enabling larger learning rates and better convergence. The paper notes that reducing learning rate by 10× is necessary without LN, indicating that normalization is essential for proper gradient flow in simple architectures without complex attention mechanisms.

- **Depthwise Convolution Captures Local Information Sufficiently**: Rather than adopting window-based self-attention mechanisms (common in recent SOTA methods like Swin Transformer), the paper demonstrates that depthwise convolution alone can capture necessary local information for image restoration. This is evidenced by the baseline's competitive performance using only local receptive fields, suggesting that global information through Channel Attention plus local feature extraction through depthwise convolution provides a better efficiency-performance trade-off than complex attention mechanisms.

- **Channel Attention Provides Global Information More Efficiently Than Spatial Attention**: The paper justifies choosing Channel Attention (CA) over vanilla spatial self-attention or window-based attention because CA maintains global information while avoiding quadratic computational complexity. Table 1 shows CA contributes +0.14 dB on SIDD and +0.24 dB on GoPro. Remarkably, Simplified Channel Attention (SCA)—which replaces CA's nonlinear activations with a simple linear layer—performs comparably (+0.03 dB on SIDD, +0.09 dB on GoPro), demonstrating that the global aggregation mechanism, not the complexity of the attention implementation, is what matters.

- **Simplicity in Architecture Design Enables Better Performance**: By maintaining a single-stage UNet and avoiding multi-stage refinement pipelines used in other SOTA methods, the paper achieves better results with dramatically lower computational cost. This challenges the assumption that breaking image restoration into multiple sequential stages (as in MPRNet and Restormer variants) is necessary. The architectural simplicity, combined with careful block design, proves sufficient for SOTA performance on both SIDD (40.30 dB) and GoPro (33.69 dB).

- **Computational Cost vs. Performance Trade-off is Not Well-Explored in Existing Methods**: Figure 1 vividly illustrates that many SOTA methods operate at highly inefficient points in the performance-computation trade-off space. NAFNet achieves better performance than previous SOTA with only 8.4% of MACs on GoPro and less than half on SIDD. This suggests the field has been optimizing primarily for performance without sufficient consideration of efficiency, and that careful, principled design can recover this lost efficiency.

</div>

<div class="lang-zh" style="display:none;">

## 核心洞見

- **通過乘法而非激活函數實現非線性**：論文的根本洞見是非線性可以通過兩個線性變換的乘積實現，而無需顯式的非線性激活函數。這源自於數學觀察，即 SimpleGate(X, Y) = X ⊙ Y（逐元素相乘）產生非線性，因為乘積運算本身是固有非線性的。這挑戰了深度網絡在每個線性變換後需要非線性激活的常規智慧。表 2 顯示用 SimpleGate 替換 GELU 在 SIDD 上產生 +0.08 dB，在 GoPro 上產生 +0.41 dB，確認此替換不僅可行而且有益。

- **層歸一化對訓練穩定性至關重要**：添加層歸一化（LN）的性能改進令人矚目：結合 10 倍學習率增加時在 GoPro 上產生 +3.39 dB。這表明 LN 不僅穩定訓練，而且從根本上改變優化景觀，使得更大學習率和更好的收斂成為可能。論文指出沒有 LN 需要將學習率降低 10 倍，表明歸一化對於簡單架構中適當梯度流是必不可少的。

- **深度卷積足以捕捉本地信息**：該論文並未採用最近最優方法（如 Swin Transformer）常見的基於窗口的自注意機制，而是證明深度卷積單獨足以捕捉圖像復原所需的本地信息。基準模型僅使用本地感受野實現競爭性性能，表明通過通道注意力進行全局信息提取加上通過深度卷積進行本地特徵提取比複雜注意機制提供更好的效率性能權衡。

- **通道注意力比空間注意力更高效地提供全局信息**：論文選擇通道注意力（CA）而非掩碼空間自注意或基於窗口注意力的理由是 CA 在避免二次計算複雜度的同時保持全局信息。表 1 顯示 CA 對 SIDD 貢獻 +0.14 dB，對 GoPro 貢獻 +0.24 dB。值得注意的是，簡化通道注意力（SCA）——將 CA 的非線性激活替換為簡單線性層——性能相當（SIDD +0.03 dB，GoPro +0.09 dB），表明全局聚合機制而非注意實現的複雜度才是重要的。

- **架構設計的簡潔性實現更好性能**：通過維持單階段 UNet 並避免其他最優方法使用的多階段精煉管道，論文以極大較低的計算成本實現更好結果。這挑戰了將圖像復原分解為多個順序階段（如 MPRNet 和 Restormer 變體）是必要的假設。架構簡潔性結合精心的塊設計，在 SIDD（40.30 dB）和 GoPro（33.69 dB）上證明足以達到最先進性能。

- **現有方法未充分探索計算成本與性能的權衡**：圖 1 清楚地說明許多最優方法在性能計算權衡空間中的位置效率極低。NAFNet 以前一最優方法在 GoPro 上 8.4% 的 MACs 和在 SIDD 上不到一半的成本實現更好性能。這表明該領域主要針對性能進行優化，而對效率考慮不足，精心原則性的設計可以恢復這失去的效率。

</div>

<div class="lang-en">

## Key Data & Results

| Dataset | Task | Method | PSNR (dB) | SSIM | MACs (G) | Speedup |
|---------|------|--------|-----------|------|----------|---------|
| SIDD | Denoising | MPRNet | 39.71 | 0.958 | 588 | - |
| SIDD | Denoising | UFormer | 39.89 | 0.960 | 89.5 | - |
| SIDD | Denoising | MAXIM | 39.96 | 0.960 | 169.5 | - |
| SIDD | Denoising | HINet | 39.99 | 0.958 | 170.7 | - |
| SIDD | Denoising | Restormer | 40.02 | 0.960 | 140 | - |
| SIDD | Denoising | **Baseline (ours)** | **40.30** | **0.962** | 84 | - |
| SIDD | Denoising | **NAFNet (ours)** | **40.30** | **0.962** | 65 | 1.09× |
| GoPro | Deblurring | MPRNet | 32.68 | 0.959 | 1235 | - |
| GoPro | Deblurring | HINet | 32.71 | 0.959 | 170.7 | - |
| GoPro | Deblurring | MAXIM | 32.86 | 0.961 | 169.5 | - |
| GoPro | Deblurring | Restormer | 32.92 | 0.961 | 140 | - |
| GoPro | Deblurring | UFormer | 32.97 | 0.967 | 89.5 | - |
| GoPro | Deblurring | DeepRFT | 33.23 | 0.963 | 187 | - |
| GoPro | Deblurring | MPRNet-local | 33.31 | 0.964 | 778.2 | - |
| GoPro | Deblurring | **Baseline (ours)** | **33.40** | **0.965** | 84 | - |
| GoPro | Deblurring | **NAFNet (ours)** | **33.69** | **0.967** | 65 | 1.09× |
| 4Scenes | Raw Denoising | PMRID | 39.76 | 0.975 | 1.2 | - |
| 4Scenes | Raw Denoising | **NAFNet (ours)** | **40.05** | **0.977** | 1.1 | - |
| REDS-val-300 | Deblurring+JPEG | HINet | 28.83 | 0.862 | 170.7 | - |
| REDS-val-300 | Deblurring+JPEG | MAXIM | 28.93 | 0.865 | 169.5 | - |
| REDS-val-300 | Deblurring+JPEG | **NAFNet (ours)** | **29.09** | **0.867** | 65 | - |

**Key Quantitative Results:**

- **Dramatic Efficiency Gains on GoPro**: NAFNet achieves 33.69 dB PSNR, surpassing the previous SOTA (MPRNet-local at 33.31 dB) by 0.38 dB while using only 8.4% of its computational cost (65 vs 778.2 MACs). This represents a ≈12× efficiency improvement per unit performance—a remarkable achievement that challenges the conventional trend of increasing model complexity for performance gains.

- **Competitive Results on SIDD with 2.1× Efficiency Gain**: While the Baseline achieves 40.30 dB and NAFNet also achieves 40.30 dB on SIDD (matching the previous SOTA Restormer at 40.02 dB), NAFNet requires only 65 MACs compared to Restormer's 140 MACs—a 2.1× reduction in computational cost. The 0.28 dB improvement over Restormer (40.30 vs 40.02 dB) with less than half the computation strongly supports the paper's central thesis.

- **Ablation Study Shows Each Component's Contribution**: Table 1 reveals that Layer Normalization alone provides the largest performance boost (+0.46 dB on SIDD, +3.39 dB on GoPro), validating its importance for training stability. GELU provides marginal gains on SIDD (39.71→39.71 dB) but meaningful gains on GoPro (+0.21 dB). Channel Attention adds +0.14 dB on SIDD and +0.24 dB on GoPro, establishing the necessity of global feature weighting.

- **Simplification Without Performance Loss**: Table 2 demonstrates that simplifying the baseline by replacing GELU with SimpleGate and CA with SCA actually improves performance: +0.11 dB on SIDD and +0.50 dB on GoPro. This counterintuitive result—that removing complexity improves performance—is a key finding that challenges conventional design principles. The speedup from these changes is marginal (0.98× to 1.11×), suggesting that the main benefit is in model design clarity rather than inference speed.

- **Generalization Across Multiple Image Restoration Tasks**: NAFNet demonstrates strong performance beyond the primary benchmarks. On raw image denoising (4Scenes), NAFNet achieves 40.05 dB, surpassing PMRID (39.76 dB) with slightly lower MACs. On REDS with JPEG artifacts, NAFNet achieves 29.09 dB, outperforming MAXIM (28.93 dB) and HINet (28.83 dB), confirming the approach's effectiveness across diverse restoration scenarios.

</div>

<div class="lang-zh" style="display:none;">

## 關鍵數據與結果

| 數據集 | 任務 | 方法 | PSNR (dB) | SSIM | MACs (G) | 加速比 |
|-------|------|------|-----------|------|----------|--------|
| SIDD | 去噪 | MPRNet | 39.71 | 0.958 | 588 | - |
| SIDD | 去噪 | UFormer | 39.89 | 0.960 | 89.5 | - |
| SIDD | 去噪 | MAXIM | 39.96 | 0.960 | 169.5 | - |
| SIDD | 去噪 | HINet | 39.99 | 0.958 | 170.7 | - |
| SIDD | 去噪 | Restormer | 40.02 | 0.960 | 140 | - |
| SIDD | 去噪 | **基準（ours）** | **40.30** | **0.962** | 84 | - |
| SIDD | 去噪 | **NAFNet（ours）** | **40.30** | **0.962** | 65 | 1.09× |
| GoPro | 去模糊 | MPRNet | 32.68 | 0.959 | 1235 | - |
| GoPro | 去模糊 | HINet | 32.71 | 0.959 | 170.7 | - |
| GoPro | 去模糊 | MAXIM | 32.86 | 0.961 | 169.5 | - |
| GoPro | 去模糊 | Restormer | 32.92 | 0.961 | 140 | - |
| GoPro | 去模糊 | UFormer | 32.97 | 0.967 | 89.5 | - |
| GoPro | 去模糊 | DeepRFT | 33.23 | 0.963 | 187 | - |
| GoPro | 去模糊 | MPRNet-local | 33.31 | 0.964 | 778.2 | - |
| GoPro | 去模糊 | **基準（ours）** | **33.40** | **0.965** | 84 | - |
| GoPro | 去模糊 | **NAFNet（ours）** | **33.69** | **0.967** | 65 | 1.09× |
| 4Scenes | 原始去噪 | PMRID | 39.76 | 0.975 | 1.2 | - |
| 4Scenes | 原始去噪 | **NAFNet（ours）** | **40.05** | **0.977** | 1.1 | - |
| REDS-val-300 | 去模糊+JPEG | HINet | 28.83 | 0.862 | 170.7 | - |
| REDS-val-300 | 去模糊+JPEG | MAXIM | 28.93 | 0.865 | 169.5 | - |
| REDS-val-300 | 去模糊+JPEG | **NAFNet（ours）** | **29.09** | **0.867** | 65 | - |

**關鍵定量結果：**

- **GoPro 上的戲劇性效率提升**：NAFNet 達到 33.69 dB PSNR，超越先前最優方法（MPRNet-local 33.31 dB）0.38 dB，同時僅使用其計算成本的 8.4%（65 對 778.2 MACs）。這代表每單位性能約 12 倍的效率改進——一項顯著成就，挑戰了增加模型複雜度以追求性能提升的常規趨勢。

- **SIDD 上的競爭性結果與 2.1 倍效率提升**：雖然基準模型在 SIDD 上達到 40.30 dB，NAFNet 也達到 40.30 dB（超越先前最優 Restormer 的 40.02 dB），NAFNet 僅需 65 MACs 相對 Restormer 的 140 MACs——計算成本減少 2.1 倍。0.28 dB 相對 Restormer 的性能改進（40.30 對 40.02 dB）加上不到一半的計算量強烈支持論文的中心主張。

- **消融研究展示各組件的貢獻**：表 1 揭示層歸一化單獨提供最大性能提升（SIDD +0.46 dB，GoPro +3.39 dB），驗證其對訓練穩定性的重要性。GELU 在 SIDD 上提供邊際收益（39.71→39.71 dB）但在 GoPro 上提供有意義的收益（+0.21 dB）。通道注意力在 SIDD 上增加 +0.14 dB，在 GoPro 上增加 +0.24 dB，確立全局特徵加權的必要性。

- **無性能損失的簡化**：表 2 展示通過用 SimpleGate 替換 GELU 和用 SCA 替換 CA 簡化基準模型實際上改進性能：SIDD +0.11 dB，GoPro +0.50 dB。這個違反直覺的結果——移除複雜度改進性能——是挑戰常規設計原則的關鍵發現。這些變化的加速比邊際（0.98× 到 1.11×），表明主要收益在於模型設計清晰性而非推理速度。

- **跨多個圖像復原任務的泛化**：NAFNet 在主要基準以外展示強勁性能。在原始圖像去噪（4Scenes）上，NAFNet 達到 40.05 dB，超越 PMRID（39.76 dB）且 MACs 略低。在帶 JPEG 偽跡的 REDS 上，NAFNet 達到 29.09 dB，超越 MAXIM（28.93 dB）和 HINet（28.83 dB），確認該方法在多樣化復原情景中的有效性。

</div>

<div class="lang-en">

## Strengths

- **Strong Empirical Results with Exceptional Efficiency**: The paper presents compelling experimental evidence across multiple benchmarks (SIDD, GoPro, 4Scenes, REDS) with significantly improved computational efficiency. The 0.38 dB improvement over MPRNet-local on GoPro using only 8.4% of its computational cost is remarkable and well-supported by the data. The consistency of results—outperforming SOTA on SIDD while using less than half the MACs of Restormer—demonstrates the robustness of the approach rather than isolated improvements on a single benchmark.

- **Rigorous Ablation Studies and Design Transparency**: The paper's systematic exploration of design choices (Tables 1-5) is exemplary. Each component's contribution is quantified: Layer Normalization (+0.46 dB SIDD, +3.39 dB GoPro), GELU (+0.21 dB GoPro), Channel Attention (+0.14-0.24 dB), and SimpleGate/SCA replacements. The ablation studies include important details like variant activation functions (Table 5) showing identity function performs comparably to nonlinear alternatives on SIDD (-0.03 to +0.03 dB variation). This transparency makes the paper a valuable reference for the community and facilitates future research.

- **Conceptually Novel Insight About Nonlinear Activations**: The fundamental contribution—demonstrating that nonlinear activation functions are not necessary for SOTA performance—is significant and thought-provoking. The theoretical connection between GELU, GLU, and SimpleGate (showing GELU is a special case of GLU with identity functions, and multiplication alone provides sufficient nonlinearity) is elegant and well-articulated. This challenges a deeply entrenched assumption in deep learning and opens new design possibilities.

- **Practical Value for Practitioners**: By providing a simple, efficient baseline with released code and pre-trained models, the paper offers immediate practical value. The simplicity of NAFNet makes it easy to understand, modify, and build upon. For practitioners deploying image restoration on resource-constrained devices (mobile phones, edge devices), the dramatic efficiency improvements are highly relevant and impactful.

- **Clear Writing and Well-Structured Presentation**: The paper is generally well-written with clear motivation (decomposing SOTA complexity into inter-block and intra-block), systematic progression from PlainNet to Baseline to NAFNet, and good visual aids (Figures 3-4 comparing block structures). The positioning of the work as enabling "convenient analysis and comparison of methods" is well-motivated and the execution supports this claim.

- **Broad Applicability Demonstrated**: The paper validates the approach on diverse image restoration tasks (denoising, deblurring, raw denoising, JPEG artifact removal), showing the method is not specialized for a single task but genuinely effective across the image restoration domain. The scaling experiments (Table 3) show NAFNet performs well with different numbers of blocks (9, 18, 36, 72), indicating flexibility in model capacity and deployment scenarios.

</div>

<div class="lang-zh" style="display:none;">

## 優勢

- **強勁的實證結果與卓越的效率**：論文在多個基準（SIDD、GoPro、4Scenes、REDS）上呈現令人信服的實驗證據且計算效率顯著提高。在 GoPro 上超越 MPRNet-local 0.38 dB 同時僅使用其 8.4% 計算成本是值得注目且充分支持的成就。結果的一致性——在 SIDD 上超越最優方法同時使用 Restormer 不到一半的 MACs——展示方法的穩健性而非單個基準上的孤立改進。

- **嚴格的消融研究與設計透明性**：論文系統化探索設計選擇（表 1-5）是典範性的。每個組件的貢獻被量化：層歸一化（+0.46 dB SIDD，+3.39 dB GoPro）、GELU（+0.21 dB GoPro）、通道注意力（+0.14-0.24 dB）和 SimpleGate/SCA 替換。消融研究包括重要細節如激活函數變體（表 5）顯示恆等函數在 SIDD 上與非線性替代品表現可比（-0.03 到 +0.03 dB 變動）。此透明性使論文成為社區的寶貴參考並促進未來研究。

- **概念上關於非線性激活的新穎洞見**：根本貢獻——展示非線性激活函數對於最先進性能並非必要——是重要和發人深省的。GELU、GLU 與 SimpleGate 之間的理論連接（顯示 GELU 是具有恆等函數的 GLU 的特例，乘法單獨提供足夠非線性）是優雅且表述清晰的。這挑戰了深度學習中根深蒂固的假設並開啟新的設計可能性。

- **對從業者的實踐價值**：通過提供簡單高效的基準及發佈代碼和預訓練模型，論文提供即時實踐價值。NAFNet 的簡潔性使其易於理解、修改和構建。對於在資源受限設備（手機、邊緣設備）上部署圖像復原的從業者，戲劇性的效率改進高度相關且有影響力。

- **清晰的寫作與結構良好的呈現**：論文通常撰寫清晰，具有清晰的動機（將最優方法複雜度分解為塊間和塊內）、從 PlainNet 到基準到 NAFNet 的系統進展，以及優質的視覺輔助（圖 3-4 比較塊結構）。將工作定位為實現「便利分析和比較方法」是恰當動機且執行支持此聲明。

- **廣泛的適用性演示**：論文在多樣化圖像復原任務（去噪、去模糊、原始去噪、JPEG 偽跡移除）上驗證方法，表明方法不是針對單個任務的專用方法而是在圖像復原領域真正有效。縮放實驗（表 3）顯示 NAFNet 在不同塊數（9、18、36、72）上表現良好，表明模型容量和部署情景的靈活性。

</div>

<div class="lang-en">

## Weaknesses

- **Limited Theoretical Justification for SimpleGate Design**: While the paper establishes that SimpleGate (element-wise multiplication) works empirically, the theoretical justification is incomplete. The claim that "the product of two linear transformations raises nonlinearity" is true mathematically but feels somewhat circular—saying multiplication is nonlinear because the output is mathematically nonlinear doesn't explain why this particular form of nonlinearity is optimal for image restoration. The paper would benefit from deeper analysis of what inductive biases SimpleGate provides compared to traditional nonlinearities like ReLU or GELU.

- **Insufficient Analysis of Why Nonlinear Activations Can Be Removed**: The paper demonstrates that nonlinear activations can be removed but provides limited insight into why this works. The connection to GLU is clever, but it doesn't fully explain the mechanism. For instance, Table 5 shows that adding explicit nonlinearities to SimpleGate (ReLU, GELU, Sigmoid variants) slightly degrades or maintains performance on GoPro (-0.35 dB to +0.1 dB variation), but the paper doesn't deeply investigate why. This lack of mechanistic understanding limits the paper's conceptual contribution.

- **Limited Novelty in Individual Components**: While the overall architecture is strong, the individual components are not novel: Layer Normalization, Channel Attention, and depthwise convolution are all existing techniques. The main novelty lies in their combination and the removal of nonlinear activations. For a top-tier venue, the incremental nature of combining existing components (albeit effectively) may be seen as less innovative compared to papers introducing fundamentally new modules or training techniques.

- **Missing Comparisons with Recent Efficient Methods**: The paper doesn't compare against some relevant efficient image restoration methods. For instance, more recent lightweight architectures like MobileNet-based approaches or knowledge distillation methods could provide additional context for the efficiency claims. Additionally, the paper doesn't discuss how NAFNet compares to methods that achieve efficiency through pruning or quantization techniques, which might offer complementary approaches.

- **Unclear Generalization to Other Vision Tasks**: While the paper tests on multiple image restoration tasks, it only evaluates on low-level vision. The fundamental claim that "nonlinear activations are not necessary" would be more compelling if validated on higher-level vision tasks (classification, detection, segmentation). The paper's conclusion that nonlinearities may not be necessary for "SOTA computer vision methods" seems overgeneralized given the limited evaluation scope. Showing results on ImageNet classification with NAFNet would strengthen this claim significantly.

- **Statistical Significance and Confidence Intervals Not Provided**: The paper reports mean PSNR values but doesn't provide error bars, confidence intervals, or significance tests. For improvements of 0.28-0.38 dB, it would be valuable to know if these improvements are statistically significant or within noise margins. The ablation studies (e.g., Table 1) show relatively small differences in some cases (0.02-0.03 dB), and without statistical analysis, it's unclear if all reported improvements are meaningful.

- **Limited Discussion of Failure Cases or Limitations**: The paper doesn't discuss scenarios where NAFNet might underperform or where nonlinear activations might be preferable. For instance, are there specific image degradation types or frequency ranges where the method struggles? What happens at extreme model scales (very small or very large)? This absence of discussion about limitations weakens the paper's scientific rigor.

</div>

<div class="lang-zh" style="display:none;">

## 劣勢

- **SimpleGate 設計的理論正當性有限**：雖然論文以實證證明 SimpleGate（逐元素相乘）有效，理論正當性不完整。「兩個線性變換的乘積提高非線性」的聲明在數學上真實但感覺循環——說乘法是非線性的因為輸出在數學上是非線性的不能解釋為什麼此特定非線性形式對圖像復原最優。論文將受益於相對傳統非線性如 ReLU 或 GELU，SimpleGate 提供什麼歸納偏差的更深層分析。

- **對為何可以移除非線性激活的分析不足**：論文展示非線性激活可被移除但對為何有效提供有限洞見。與 GLU 的連接聰慧，但不完全解釋機制。例如，表 5 顯示向 SimpleGate 添加顯式非線性（ReLU、GELU、Sigmoid 變體）在 GoPro 上略微退化或保持性能（-0.35 dB 到 +0.1 dB 變動），但論文不深入調查為什麼。此機制理解缺失限制論文的概念貢獻。

- **個別組件的創新性有限**：雖然整體架構強勁，個別組件不新穎：層歸一化、通道注意力和深度卷積都是既有技術。主要創新在於其組合和非線性激活的移除。對於頂級場地，組合現有組件的遞進性質（儘管有效）可能被視為比引入基本上新穎模塊或訓練技術的論文創新性較低。

- **缺少與最近高效方法的比較**：論文未比較某些相關高效圖像復原方法。例如，最近輕量級架構如基於 MobileNet 的方法或知識蒸餾方法可提供效率聲稱的額外背景。另外，論文未討論 NAFNet 如何與通過剪枝或量化技術實現效率的方法比較，這些可能提供互補方法。

- **對其他視覺任務的泛化不清晰**：雖然論文在多個圖像復原任務上測試，僅在低級視覺上評估。「非線性激活不必要」的根本聲稱若在更高級視覺任務（分類、檢測、分割）上驗證將更令人信服。論文的結論「非線性對於最先進計算機視覺方法可能不必要」鑑於評估範圍有限似乎過度泛化。在 ImageNet 分類上使用 NAFNet 展示結果將顯著加強此聲稱。

- **未提供統計顯著性和置信區間**：論文報告平均 PSNR 值但不提供誤差棒、置信區間或顯著性測試。對於 0.28-0.38 dB 的改進，知道這些改進是否統計顯著或在噪聲邊界內將有價值。消融研究（例如表 1）在某些情況下顯示相對小差異（0.02-0.03 dB），沒有統計分析，不清楚是否所有報告改進都有意義。

- **缺少對失敗情況或限制的討論**：論文未討論 NAFNet 可能表現不佳或非線性激活可能更優的情景。例如，是否存在網絡表現不力的特定圖像退化類型或頻率範圍？在極端模型尺度（非常小或非常大）時會發生什麼？此對限制討論的缺失弱化論文的科學嚴謹性。

</div>

<div class="lang-en">

## Research Directions

- **Mechanistic Understanding of Nonlinearity-Free Operations**: Investigate why element-wise multiplication of feature maps (SimpleGate) provides sufficient nonlinearity for image restoration. Conduct theoretical and empirical analysis using tools like neural tangent kernels (NTKs), Riemannian geometry of neural networks, or feature correlation analysis to understand the inductive biases of SimpleGate versus traditional activations. A strong paper could characterize the "nonlinearity landscape" of image restoration and show which tasks or degradation types benefit most from different nonlinear mechanisms.

- **Extend Nonlinear-Activation-Free Design to High-Level Vision Tasks**: Test whether the NAFNet principles generalize beyond low-level vision to classification (ImageNet), detection (COCO), and segmentation (ADE20K) tasks. The current claim that nonlinear activations may not be necessary for "SOTA computer vision methods" is not yet validated on these benchmarks. A compelling direction would be to develop activation-free variants of Vision Transformers or CNNs and demonstrate competitive performance on ImageNet with superior efficiency, which would significantly strengthen the paper's conceptual claims.

- **Theoretical Analysis via Linear Algebraic Perspective**: Reformulate image restoration as a linear transformation problem and analyze when nonlinear activations become redundant. Investigate whether the role of nonlinearities in image restoration differs fundamentally from their role in classification tasks. This could involve analyzing the singular value decomposition (SVD) of learned weights, studying rank properties of feature maps, or developing novel analysis tools specific to low-level vision that explain why multiplication suffices.

- **Combination with Emerging Efficient Training Techniques**: Integrate NAFNet with knowledge distillation, neural architecture search (NAS), or dynamic inference mechanisms to push efficiency boundaries further. For instance, explore conditional computing where different network branches (simple vs. complex) are selected based on input degradation severity, or combine NAFNet with mixture-of-experts (MoE) approaches adapted for image restoration. This could achieve sub-8.4% computational cost on GoPro while maintaining or improving PSNR.

- **Adaptation for Practical Deployment on Edge and Mobile Devices**: Extend NAFNet to mobile and edge devices with quantization-aware training, mixed-precision inference, and hardware-specific optimizations. Conduct real-world deployment studies (e.g., on Snapdragon processors, Apple Neural Engine) measuring actual latency, power consumption, and thermal characteristics. This would validate whether the theoretical MACs reduction translates to practical speedups and battery savings on devices users care about.

- **Multi-task Learning and Unified Restoration Framework**: Develop a unified NAFNet variant that handles multiple restoration tasks (denoising, deblurring, super-resolution, inpainting) simultaneously with task-specific adapters or soft routing mechanisms. Investigate whether removing nonlinear activations provides unique advantages for multi-task learning by reducing task interference or improving gradient flow. A successful approach could demonstrate that simplicity enables better transfer learning and multitask generalization.

- **Fundamental Study on Activation Function Necessity**: Conduct comprehensive experiments across diverse architectures (CNNs, Vision Transformers, MLPs) and tasks to characterize when and why nonlinear activations can be removed. Create a taxonomic framework categorizing tasks by their "nonlinearity requirement" (e.g., high for classification, potentially lower for image restoration). This meta-analysis would provide principled guidelines for future architecture design and challenge fundamental assumptions about how neural networks should be structured.

</div>

<div class="lang-zh" style="display:none;">

## 研究方向

- **對無非線性運算的機制理解**：調查為什麼特徵圖的逐元素相乘（SimpleGate）為圖像復原提供充分非線性。使用神經切線核（NTKs）、神經網絡黎曼幾何或特徵相關性分析等工具進行理論和實證分析，理解 SimpleGate 相對傳統激活的歸納偏差。強大論文可刻畫圖像復原的「非線性景觀」並展示哪些任務或退化類型最受益於不同非線性機制。

- **將無非線性激活設計擴展到高級視覺任務**：測試 NAFNet 原則是否超越低級視覺泛化到分類（ImageNet）、檢測（COCO）和分割（ADE20K）任務。當前「非線性激活對於最優計算機視覺方法可能不必要」的聲稱未在這些基準上驗證。令人信服的方向是開發視覺變換器或 CNN 的無激活變體並在 ImageNet 上展示具競爭性能伴優越效率，將顯著加強論文的概念聲稱。

- **通過線性代數視角的理論分析**：將圖像復原重新表述為線性變換問題並分析何時非線性激活變冗餘。調查圖像復原中非線性的角色與分類任務中的根本差異。這可涉及分析學習權重的奇異值分解（SVD）、研究特徵圖秩特性或開發低級視覺特定新穎分析工具解釋為什麼乘法足夠。

- **與新興高效訓練技術結合**：將 NAFNet 與知識蒸餾、神經架構搜索（NAS）或動態推理機制整合以進一步推進效率邊界。例如，探索條件計算其中基於輸入退化嚴重度選擇不同網絡分支（簡單對複雜），或將 NAFNet 與適應圖像復原的專家混合（MoE）方法結合。這可在 GoPro 上達到低於 8.4% 計算成本同時保持或改進 PSNR。

- **適應實際邊緣和移動設備部署**：使用量化感知訓練、混合精度推理和硬體特定優化將 NAFNet 擴展到移動和邊緣設備。進行實世界部署研究（例如在 Snapdragon 處理器、Apple Neural Engine 上）測量實際延遲、功耗消耗和熱特性。這將驗證理論 MACs 減少是否轉化為用戶關心的設備上的實際加速和電池省電。

- **多任務學習與統一復原框架**：開發統一 NAFNet 變體處理多個復原任務（去噪、去模糊、超解析度、修復）同時具有任務特定適配器或軟路由機制。調查移除非線性激活是否通過減少任務干擾或改進梯度流為多任務學習提供獨特優勢。成功方法可展示簡潔性如何實現更好遷移學習和多任務泛化。

- **關於激活函數必要性的基礎研究**：跨多樣化架構（CNN、視覺變換器、MLP）和任務進行全面實驗刻畫何時以及為何非線性激活可被移除。創建分類框架按「非線性需求」分類任務（例如分類高，圖像復原可能較低）。此後設分析提供未來架構設計的原則化指南並挑戰關於神經網絡應如何結構的根本假設。

</div>


