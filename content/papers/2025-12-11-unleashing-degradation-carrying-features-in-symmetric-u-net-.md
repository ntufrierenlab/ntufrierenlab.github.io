---
title: "Unleashing Degradation-Carrying Features in Symmetric U-Net: Simpler and Stronger Baselines for All-in-One Image Restoration"
date: 2025-12-11
authors:
  - "Wenlong Jiao"
  - "Heyang Lee"
  - "Ping Wang"
  - "Pengfei Zhu"
  - "Qinghua Hu"
  - "Dongwei Ren"
source: "arXiv"
arxiv_url: "https://arxiv.org/abs/2512.10581"
pdf_url: "https://arxiv.org/pdf/2512.10581"
one_line_summary: "SymUNet establishes that a simple symmetric U-Net architecture with aligned encoder-decoder hierarchies and additive skip connections achieves state-of-the-art all-in-one image restoration while maintaining superior computational efficiency, with SE-SymUNet further improving performance through lightweight bidirectional semantic guidance from frozen CLIP features."
one_line_summary_zh: "SymUNet通過對齐編碼器-解碼器層級的簡單對稱U-Net架構和加法跳過連接在全合一圖像修復中達到最先進的性能，同時保持卓越的計算效率，SE-SymUNet進一步通過來自凍結CLIP特徵的輕量級雙向語義指導提升性能。"
date_added: 2026-03-09
topics: ["Image Deblurring"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **Paradigm Shift toward Architectural Simplicity**: The paper challenges the prevailing trend of increasingly complex all-in-one image restoration methods (Mixture-of-Experts, diffusion models, agent-based systems) by demonstrating that a symmetric U-Net architecture is fundamentally more effective. The key insight is that well-designed feature extraction inherently encodes degradation-specific information, eliminating the need for external complexity. This represents a significant philosophical departure from recent methods, validated through extensive benchmarking.

- **SymUNet: Symmetric U-Net Baseline**: The paper proposes SymUNet, a streamlined baseline built on strict U-shape architecture with three critical design principles: (1) symmetric encoder-decoder structure with aligned block counts, (2) consistent channel dimensions across corresponding stages (avoiding the "channel doubling" in asymmetric designs), and (3) elimination of auxiliary refinement blocks. This simple architecture achieves state-of-the-art performance on three-task (32.93 dB PSNR) and five-task (30.43 dB PSNR) benchmarks while using fewer parameters than competing methods.

- **SE-SymUNet: Semantic-Enhanced Variant with Bidirectional Guidance**: Building upon SymUNet, SE-SymUNet integrates frozen CLIP Vision Transformer features through a novel bidirectional semantic guidance mechanism. This module consists of (1) Semantic Guidance, where image features query semantic context via cross-attention, and (2) Semantic Refinement, where semantic context is updated based on refined image features, creating an iterative feedback loop. This lightweight enhancement yields consistent improvements (33.08 dB for three-task) without full VLM integration complexity.

- **Comprehensive Multi-Task Evaluation Framework**: The paper establishes rigorous evaluation protocols across diverse degradation scenarios. Three-task setting combines denoising (BSD68 with σ ∈ {15,25,50}), dehazing (SOTS-Outdoor, 500 images), and deraining (Rain100L, 100 images). Five-task setting adds deblurring (GoPro, 1,111 images) and low-light enhancement (LOL, 15 images). Training on 77,479 images (three-task) and 80,067 images (five-task) demonstrates generalization across heterogeneous degradations.

- **Architectural Analysis and Motivation Validation**: The paper provides clear visualization and theoretical motivation for the symmetric design. Figure 3 demonstrates that degradation-carrying features (atmospheric haze, rain streaks, noise patterns, blur boundaries) are effectively isolated in decoder features f₀^dec, directly supporting the claim that architectural simplicity preserves degradation cues. Ablation studies (Table 3) quantify the 0.38 dB performance gap between symmetric and asymmetric baselines, providing empirical validation of the core hypothesis.

- **Strong Reproducibility and Open-Source Contribution**: The authors provide complete implementation details including training configurations (AdamW optimizer with learning rate 1×10⁻³, cosine annealing schedule, batch size 32, combined L1 and frequency-domain loss with weight λ=0.1), model architecture specifications (4-level U-Net with 4/6/6 encoder blocks, 8 bottleneck blocks, 6/6/4 decoder blocks), and promise open-source code, facilitating community adoption and reproducibility.

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **轉向架構簡潔性的典範轉移**: 本文挑戰了當前全合一圖像修復方法日益複雜化的趨勢（專家混合模型、擴散模型、基於代理的系統），通過證明對稱U-Net架構在根本上更有效，改變了該領域的方向。核心洞見是精心設計的特徵提取本身就蘊含降噪特定信息，無需外部複雜性輔助。通過廣泛的基準測試驗證，這代表了相對於近期方法的重大哲學轉變。

- **SymUNet：對稱U-Net基線**: 本文提出SymUNet，一個基於嚴格U形架構的簡化基線，具有三個關鍵設計原則：(1)對稱的編碼器-解碼器結構，具有對齊的塊計數，(2)在相應階段保持一致的通道維度（避免非對稱設計中的「通道加倍」），(3)消除輔助精化塊。這個簡單架構在三任務（32.93 dB PSNR）和五任務（30.43 dB PSNR）基準上達到最先進的性能，同時使用比競爭方法更少的參數。

- **SE-SymUNet：具有雙向指導的語義增強變體**: 在SymUNet基礎上，SE-SymUNet通過新穎的雙向語義指導機制整合凍結的CLIP Vision Transformer特徵。該模塊包含：(1)語義指導，其中圖像特徵通過交叉注意查詢語義上下文，(2)語義精化，其中語義上下文根據精化的圖像特徵進行更新，形成迭代反饋迴路。這個輕量級增強在不需要完整VLM集成複雜性的情況下，產生持續改進（三任務33.08 dB）。

- **全面的多任務評估框架**: 本文在多樣化降噪場景中建立了嚴格的評估協議。三任務設置結合了去噪（BSD68，σ ∈ {15,25,50}）、去霾（SOTS-Outdoor，500張圖像）和去雨（Rain100L，100張圖像）。五任務設置增加了去模糊（GoPro，1,111張圖像）和低光增強（LOL，15張圖像）。在77,479張圖像（三任務）和80,067張圖像（五任務）上的訓練展示了在異質降噪中的泛化能力。

- **架構分析和動機驗證**: 本文為對稱設計提供了清晰的可視化和理論動機。圖3證明了降噪特性信息（大氣霾、雨條紋、噪聲圖案、模糊邊界）在解碼器特徵f₀^dec中得到有效隔離，直接支持架構簡潔性保留降噪線索的主張。消融研究（表3）量化了對稱和非對稱基線之間0.38 dB的性能差距，提供了對核心假設的實證驗證。

- **強大的可重現性和開源貢獻**: 作者提供了完整的實現細節，包括訓練配置（AdamW優化器，學習率1×10⁻³，余弦退火調度，批大小32，結合L1和頻域損失，權重λ=0.1）、模型架構規範（4層U-Net，4/6/6編碼器塊，8個瓶頸塊，6/6/4解碼器塊），並承諾開放源代碼，促進社區採納和可重現性。

</div>

<div class="lang-en">

## Core Insights

- **Intrinsic Degradation Encoding in Well-Designed Features**: The paper reveals that encoder feature extraction, when carefully designed, inherently encodes degradation-specific information without requiring explicit external guidance. This is demonstrated through visualization (Figure 3) showing that decoder features isolate distinct degradation patterns: atmospheric haze produces characteristic spatial patterns, rain creates directional streak signatures, and noise exhibits random structure. This insight suggests that the network learns fundamental degradation properties during feature extraction, making complex auxiliary mechanisms redundant.

- **Semantic Hierarchy Misalignment as Core Bottleneck**: The authors identify that asymmetric "decoder-heavy" architectures fundamentally misalign semantic hierarchy. The encoder captures multi-level features ranging from low-level degradation details to high-level semantics, but asymmetric designs with abrupt channel expansion (e.g., doubling from C to 2C) and auxiliary refinement blocks disrupt this hierarchy. This misalignment is particularly problematic in all-in-one settings where multiple heterogeneous degradations interact, causing training instabilities and conflicting signals across different degradation types that cannot be mitigated by degradation consistency alone.

- **Architectural Symmetry Ensures Robust Signal Preservation**: The paper demonstrates through ablation studies (Table 3) that maintaining strict U-shape symmetry with consistent channel dimensions delivers a 0.38 dB improvement (32.93 dB vs 32.55 dB) over asymmetric designs. This improvement stems from creating a "direct and efficient pathway" where degradation cues from the encoder are seamlessly fused and preserved throughout the decoder without dilution. The symmetric structure acts as an inductive bias that is particularly valuable for multi-degradation scenarios where feature conflicts can destabilize learning.

- **Simple Additive Fusion Sufficiency in Aligned Architectures**: Contrary to the trend toward complex fusion mechanisms (concatenation-based with refinement, dynamic routing), the paper shows that simple element-wise additive skip connections (fi^dec = UP(fi+1^dec) + si) are sufficient for state-of-the-art performance when paired with architectural symmetry. This efficiency gain is crucial for practical deployment, as the simpler fusion mechanism reduces computational overhead while the aligned skip connections preserve degradation-aware information with higher fidelity than concatenation-based approaches.

- **Lightweight Semantic Guidance Effectiveness via Bidirectional Feedback**: SE-SymUNet's bidirectional semantic module (Table 4) shows that even modest semantic enhancement (0.15 dB improvement from one-way to bidirectional guidance) is effective because the underlying symmetric architecture is already sufficiently powerful. The iterative refinement cycle where semantic context queries refined image features (Semantic Refinement) enables adaptive guidance that becomes increasingly accurate as features are cleaned across decoder levels. This contrasts with approaches that require frozen or static external priors, suggesting that dynamic interaction between image features and semantic context is more beneficial.

- **Generalization Across Heterogeneous Degradations**: The paper demonstrates robust generalization from three-task (32.93 dB) to five-task (30.43 dB) settings by adding deblurring and low-light enhancement. The performance degradation of only 2.5 dB PSNR despite substantially increased task complexity suggests the symmetric architecture naturally handles diverse degradation types without task-specific tuning. This is supported by per-task analysis showing SE-SymUNet dominates in dehazing (+0.51 dB over second-best) and remains competitive across all tasks, validating the architecture's universal applicability.

</div>

<div class="lang-zh" style="display:none;">

## 核心洞見

- **精心設計特徵中的內在降噪編碼**: 本文揭示了編碼器特徵提取在精心設計時，本質上蘊含特定降噪信息，無需明確的外部指導。這通過可視化（圖3）得到證明，顯示解碼器特徵隔離了不同的降噪模式：大氣霾產生特徵性的空間圖案，雨水創建定向條紋簽名，噪聲表現出隨機結構。這個洞見表明網絡在特徵提取過程中學習基本的降噪屬性，使複雜的輔助機制變得多餘。

- **語義層次失對齊作為核心瓶頸**: 作者識別出非對稱「解碼器-重型」架構根本上失對齐了語義層次。編碼器捕捉從低級降噪細節到高級語義的多級特徵，但具有突然通道擴展（例如，從C加倍到2C）和輔助精化塊的非對稱設計破壞了這個層次。這種失對齐在全合一設置中特別有問題，其中多種異質降噪相互作用，導致不同降噪類型之間的訓練不穩定性和衝突信號，無法單獨通過降噪一致性進行緩解。

- **架構對稱性確保魯棒信號保留**: 本文通過消融研究（表3）證明了維持嚴格U形對稱性與一致通道維度相比非對稱設計提供0.38 dB改進（32.93 dB vs 32.55 dB）。這個改進來自於創建「直接且有效的通路」，其中來自編碼器的降噪線索在整個解碼器中無縫融合和保留，不受稀釋。對稱結構充當歸納偏差，對於多降噪場景特別有價值，其中特徵衝突可能使學習不穩定。

- **簡單加法融合在對齐架構中的充分性**: 與朝向複雜融合機制（基於連接的精化、動態路由）的趨勢相反，本文顯示簡單的逐元素加法跳過連接（fi^dec = UP(fi+1^dec) + si）與架構對稱性配對時足以達到最先進的性能。這個效率增益對實際部署至關重要，因為更簡單的融合機制減少了計算開銷，而對齐的跳過連接保留降噪信息的保真度比基於連接的方法更高。

- **通過雙向反饋的輕量級語義指導有效性**: SE-SymUNet的雙向語義模塊（表4）顯示，即使適度的語義增強（從單向到雙向指導的0.15 dB改進）也是有效的，因為基礎對稱架構已經足夠強大。迭代精化循環，其中語義上下文查詢精化的圖像特徵（語義精化），可實現自適應指導，隨著特徵在解碼器級別的清潔而變得越來越準確。這與需要凍結或靜態外部先驗的方法形成對比，表明圖像特徵和語義上下文之間的動態交互更有益。

- **跨異質降噪的泛化**: 本文通過添加去模糊和低光增強，從三任務（32.93 dB）到五任務（30.43 dB）設置展示了魯棒的泛化。儘管任務複雜性大幅增加，但僅2.5 dB PSNR的性能下降表明對稱架構自然地處理多種降噪類型而無需特定任務調整。這由按任務分析支持，顯示SE-SymUNet在去霾中主導（比第二好多+0.51 dB）並在所有任務中保持競爭力，驗證了該架構的通用適用性。

</div>

<div class="lang-en">

## Key Data & Results

| Method | Dehazing (SOTS) | Deraining (Rain100L) | Denoising BSD68 σ=25 | Denoising BSD68 σ=15 | Denoising BSD68 σ=50 | Average PSNR | Average SSIM |
|--------|-----------------|----------------------|----------------------|----------------------|----------------------|--------------|--------------|
| Restormer | 27.78/0.958 | 33.78/0.958 | 30.67/0.865 | 33.72/0.865 | 27.63/0.792 | 30.75 | 0.901 |
| NAFNet | 24.11/0.960 | 33.64/0.956 | 30.47/0.865 | 33.18/0.918 | 27.12/0.754 | 29.67 | 0.844 |
| AirNet | 27.94/0.962 | 34.90/0.968 | 31.26/0.888 | 33.92/0.933 | 28.00/0.797 | 31.20 | 0.910 |
| PromptIR | 30.58/0.974 | 36.37/0.972 | 31.31/0.888 | 33.98/0.933 | 28.06/0.799 | 32.06 | 0.913 |
| DFPIR | 31.87/0.980 | 38.65/0.982 | 31.47/0.893 | 34.14/0.935 | 28.25/0.806 | 32.88 | 0.919 |
| MoCE-IR | 31.34/0.979 | 38.57/0.984 | 31.45/0.888 | 34.11/0.932 | 28.18/0.800 | 32.73 | 0.917 |
| SE-SymUNet | 32.02/0.983 | 39.23/0.986 | 31.58/0.895 | 34.23/0.937 | 28.33/0.809 | **33.08** | **0.922** |
| **SymUNet** | **31.40/0.981** | **39.12/0.985** | **31.57/0.894** | **34.22/0.937** | **28.32/0.808** | **32.93** | **0.921** |

*Three-Task Benchmark Results (Table 1): PSNR (dB) / SSIM*

- **State-of-the-Art Performance on Three-Task Benchmark**: SymUNet achieves 32.93 dB average PSNR, outperforming previous SOTA DFPIR (32.88 dB) by 0.05 dB, while SE-SymUNet extends this to 33.08 dB. More importantly, SymUNet accomplishes this with substantially fewer parameters than competing methods, as evidenced by Figure 1 showing SymUNet in the optimal efficiency-performance quadrant with only AirNet having fewer parameters. Per-task analysis reveals SymUNet leads in deraining (39.12 dB) and matches DFPIR in dehazing while delivering competitive denoising across all noise levels.

- **Robust Generalization to Five-Task Setting**: The expansion to five-task restoration (adding deblurring and low-light enhancement) demonstrates consistent methodology. SymUNet achieves 30.43 dB average PSNR while SE-SymUNet reaches 30.73 dB, maintaining state-of-the-art status. The performance degradation from three-task to five-task (2.5 dB) is minimal, suggesting the symmetric architecture generalizes effectively. SE-SymUNet shows dominant performance in dehazing (+0.51 dB over second-best Perceive-IR at 29.84 dB) and provides competitive results across deblurring and low-light tasks.

- **Architectural Symmetry Validation**: Ablation study (Table 3) directly validates the core hypothesis by comparing symmetric vs. asymmetric baselines. The symmetric SymUNet achieves 32.93 dB PSNR / 0.921 SSIM versus asymmetric baseline at 32.55 dB PSNR / 0.919 SSIM, confirming a significant 0.38 dB improvement. This quantitative evidence supports the claim that eliminating decoder-heavy designs and maintaining consistent channel dimensions preserves degradation-carrying information more effectively.

- **Bidirectional Semantic Guidance Contribution**: Table 4 shows progressive improvements from adding semantic components: baseline SymUNet (32.93 dB), one-way semantic guidance (+0.09 dB to 33.02 dB), and full bidirectional guidance (+0.06 dB to 33.08 dB, SE-SymUNet). While individual gains appear modest, the consistent improvement pattern validates that the iterative feedback loop between image features and semantic context provides tangible benefits. The modest incremental gains also support the conclusion that the symmetric architecture is already powerful, making semantic injection a refinement rather than a fundamental necessity.

- **Computational Efficiency Advantage**: Figure 1 demonstrates that SymUNet resides in the optimal top-left quadrant representing the efficiency-performance Pareto frontier. With the sole exception of AirNet, SymUNet uses fewer parameters than all competing methods while achieving superior PSNR and SSIM. The streamlined symmetric design with simple additive skip connections directly translates to reduced FLOPs and faster inference, making SymUNet more practical for real-world deployment in resource-constrained environments compared to MoE-based methods like MoCE-IR or diffusion-based approaches.

</div>

<div class="lang-zh" style="display:none;">

## 關鍵數據與結果

| 方法 | 去霾 (SOTS) | 去雨 (Rain100L) | 去噪 BSD68 σ=25 | 去噪 BSD68 σ=15 | 去噪 BSD68 σ=50 | 平均 PSNR | 平均 SSIM |
|--------|-----------------|----------------------|----------------------|----------------------|----------------------|--------------|--------------|
| Restormer | 27.78/0.958 | 33.78/0.958 | 30.67/0.865 | 33.72/0.865 | 27.63/0.792 | 30.75 | 0.901 |
| NAFNet | 24.11/0.960 | 33.64/0.956 | 30.47/0.865 | 33.18/0.918 | 27.12/0.754 | 29.67 | 0.844 |
| AirNet | 27.94/0.962 | 34.90/0.968 | 31.26/0.888 | 33.92/0.933 | 28.00/0.797 | 31.20 | 0.910 |
| PromptIR | 30.58/0.974 | 36.37/0.972 | 31.31/0.888 | 33.98/0.933 | 28.06/0.799 | 32.06 | 0.913 |
| DFPIR | 31.87/0.980 | 38.65/0.982 | 31.47/0.893 | 34.14/0.935 | 28.25/0.806 | 32.88 | 0.919 |
| MoCE-IR | 31.34/0.979 | 38.57/0.984 | 31.45/0.888 | 34.11/0.932 | 28.18/0.800 | 32.73 | 0.917 |
| SE-SymUNet | 32.02/0.983 | 39.23/0.986 | 31.58/0.895 | 34.23/0.937 | 28.33/0.809 | **33.08** | **0.922** |
| **SymUNet** | **31.40/0.981** | **39.12/0.985** | **31.57/0.894** | **34.22/0.937** | **28.32/0.808** | **32.93** | **0.921** |

*三任務基準結果（表1）：PSNR (dB) / SSIM*

- **三任務基準上的最先進性能**: SymUNet達到32.93 dB平均PSNR，超過之前的SOTA DFPIR（32.88 dB）0.05 dB，而SE-SymUNet將其擴展到33.08 dB。更重要的是，SymUNet以比競爭方法實質上更少的參數實現了這一點，如圖1所示，SymUNet位於最優效率-性能象限，僅AirNet參數更少。按任務分析顯示SymUNet在去雨中領先（39.12 dB），在去霾中與DFPIR相匹配，同時在所有噪聲水平上提供競爭力的去噪。

- **對五任務設置的魯棒泛化**: 擴展到五任務修復（添加去模糊和低光增強）展示了一致的方法論。SymUNet達到30.43 dB平均PSNR，而SE-SymUNet達到30.73 dB，保持了最先進的地位。從三任務到五任務的性能下降（2.5 dB）最小，表明對稱架構有效泛化。SE-SymUNet在去霾中顯示主導性能（比第二最佳Perceive-IR（29.84 dB）多+0.51 dB），並在去模糊和低光任務中提供競爭力結果。

- **架構對稱性驗證**: 消融研究（表3）通過比較對稱與非對稱基線直接驗證了核心假設。對稱SymUNet達到32.93 dB PSNR / 0.921 SSIM，而非對稱基線達到32.55 dB PSNR / 0.919 SSIM，確認了顯著的0.38 dB改進。這個定量證據支持了消除解碼器-重型設計並保持一致通道維度更有效地保留降噪特性信息的主張。

- **雙向語義指導貢獻**: 表4顯示了添加語義成分的漸進式改進：基線SymUNet（32.93 dB）、單向語義指導（+0.09 dB至33.02 dB）和完整雙向指導（+0.06 dB至33.08 dB，SE-SymUNet）。雖然單個增益看起來適度，但一致的改進模式驗證了圖像特徵和語義上下文之間的迭代反饋迴路提供了切實的好處。適度的增量增益也支持對稱架構已經足夠強大的結論，使語義注入成為優化而非根本必要性。

- **計算效率優勢**: 圖1證明了SymUNet位於代表效率-性能帕累托邊界的最優左上象限。除了AirNet，SymUNet使用比所有競爭方法更少的參數，同時達到優越的PSNR和SSIM。簡化的對稱設計與簡單加法跳過連接直接轉化為減少的FLOPs和更快的推理，相比MoCE-IR等基於MoE的方法或基於擴散的方法，使SymUNet在資源受限環境中的實際部署更實用。

</div>

<div class="lang-en">

## Strengths

- **Strong Conceptual Novelty and Clear Motivation**: The paper makes a bold and well-articulated case for architectural simplicity in all-in-one image restoration, directly challenging the dominant paradigm of increasingly complex methods. The motivation is supported by clear visual evidence (Figure 3) showing how symmetric architectures preserve degradation-carrying features, and the observation about semantic hierarchy misalignment provides theoretical grounding for the proposed approach. This represents a meaningful contribution to architectural design philosophy in restoration, offering insights that extend beyond the immediate problem.

- **Comprehensive Experimental Validation**: The evaluation across both three-task and five-task benchmarks with multiple baseline comparisons (9 competing methods in three-task setting) demonstrates thorough experimental rigor. The inclusion of both standard benchmarks (BSD68, SOTS-Outdoor, Rain100L, GoPro, LOL) and large-scale training datasets (77,479 to 80,067 images) ensures statistical significance. The ablation studies systematically isolate the contributions of architectural symmetry (0.38 dB improvement) and semantic guidance components, providing clear evidence for design choices.

- **Excellent Parameter Efficiency and Computational Feasibility**: SymUNet achieves state-of-the-art results with fewer parameters than virtually all competing methods (Figure 1), making it practically valuable for deployment. The streamlined symmetric design with simple additive skip connections directly reduces computational overhead compared to concatenation-based fusion and auxiliary refinement blocks. This efficiency-performance balance addresses a critical practical concern often overlooked in pursuit of marginal PSNR improvements, making the work particularly valuable for real-world applications.

- **Reproducibility and Strong Open-Source Commitment**: The paper provides extensive implementation details including network architecture specifications (exact block counts at each level: 4/6/6 encoder, 8 bottleneck, 6/6/4 decoder), training hyperparameters (learning rate, optimizer settings, loss function weights), and data augmentation protocols. The commitment to releasing source code enhances reproducibility and enables the community to build upon the work. These details facilitate independent verification and future extensions of the methodology.

- **Effective Integration of Semantic Priors without Over-Engineering**: SE-SymUNet demonstrates that lightweight semantic guidance through frozen CLIP features can augment a strong baseline without introducing unnecessary complexity. The bidirectional guidance mechanism is elegantly simple—using standard cross-attention rather than complex fusion networks—yet provides consistent improvements. This design philosophy aligns well with the paper's broader argument about unnecessary complexity and shows how to thoughtfully incorporate external priors when beneficial.

- **Clear Presentation and Well-Structured Narrative**: The paper is well-written with effective use of figures to communicate ideas. Figure 2 clearly contrasts asymmetric and symmetric designs, Figure 3 provides compelling visual evidence of degradation-carrying features, and Figure 5 demonstrates visual quality improvements. The progression from motivation through method to comprehensive evaluation is logical and easy to follow, making the contributions accessible to a broad audience despite the technical sophistication.

</div>

<div class="lang-zh" style="display:none;">

## 優勢

- **強大的概念新穎性和清晰的動機**: 本文針對全合一圖像修復中的架構簡潔性做出了大膽且清晰的論述，直接挑戰了日益複雜方法的主導範式。動機由清晰的視覺證據（圖3）支持，顯示對稱架構如何保留降噪特性信息，而語義層次失對齊的觀察為提議的方法提供了理論基礎。這代表了對修復中架構設計哲學的有意義的貢獻，提供了超越直接問題的洞見。

- **全面的實驗驗證**: 在三任務和五任務基準上的評估，包括多種基線比較（三任務設置中9種競爭方法），展示了徹底的實驗嚴謹性。包括標準基準（BSD68、SOTS-Outdoor、Rain100L、GoPro、LOL）和大規模訓練數據集（77,479至80,067張圖像）確保了統計顯著性。消融研究系統性地隔離了架構對稱性（0.38 dB改進）和語義指導成分的貢獻，為設計選擇提供了清晰證據。

- **優秀的參數效率和計算可行性**: SymUNet以比幾乎所有競爭方法更少的參數達到最先進的結果（圖1），使其對部署實踐有價值。簡化的對稱設計與簡單加法跳過連接相比於基於連接的融合和輔助精化塊直接減少了計算開銷。這種效率-性能平衡解決了在追求邊際PSNR改進中常被忽視的關鍵實際問題，使該工作對實際應用特別有價值。

- **可重現性和強大的開源承諾**: 本文提供了廣泛的實現細節，包括網絡架構規範（每個級別的精確塊計數：4/6/6編碼器、8個瓶頸、6/6/4解碼器）、訓練超參數（學習率、優化器設置、損失函數權重）和數據增強協議。承諾發布源代碼增強了可重現性，使社區能夠基於該工作進行構建。這些細節促進了獨立驗證和方法的未來擴展。

- **語義先驗的有效整合而無過度工程**: SE-SymUNet展示了通過凍結CLIP特徵的輕量級語義指導可以增強強基線而無需引入不必要的複雜性。雙向指導機制優雅地簡單——使用標準交叉注意而非複雜融合網絡——但提供持續改進。這種設計哲學與論文更廣泛的關於不必要複雜性的論述一致，顯示了如何在有益時明智地整合外部先驗。

- **清晰的演示和良好結構化的敘述**: 本文寫得很好，有效利用圖形傳達思想。圖2清晰地對比了非對稱和對稱設計，圖3提供了降噪特性信息的令人信服的視覺證據，圖5展示了視覺質量改進。從動機通過方法到全面評估的進展是邏輯清晰且易於理解的，儘管技術複雜性很高，但使貢獻對廣泛觀眾也容易理解。

</div>

<div class="lang-en">

## Weaknesses

- **Limited Novelty in Core Architecture**: While the symmetric U-Net design is effective, the fundamental architecture itself is relatively standard and not particularly novel. The main contribution is showing that this "simpler" design outperforms complex recent methods rather than introducing architectural innovations. The paper would be stronger if it provided deeper theoretical analysis or novel architectural components that explain *why* symmetry is beneficial beyond intuitive feature flow arguments. The reliance on existing Transformer blocks adapted from Restormer further limits the architectural novelty.

- **Incomplete Analysis of Failure Cases and Limitations**: The paper does not thoroughly discuss scenarios where the proposed methods underperform. Figure 5 shows visual comparisons on successful cases, but there is no discussion of degradations or image types where SymUNet struggles. For instance, low-light enhancement results (LOL benchmark) show more modest improvements, and the paper doesn't analyze why semantic guidance helps some tasks (dehazing) more than others (low-light). A candid discussion of failure modes would strengthen the work's credibility.

- **Limited Theoretical Justification for Design Choices**: While the paper provides empirical validation through ablations, several design choices lack theoretical grounding. For example: (1) Why is additive skip connection inherently better than concatenation beyond empirical results? (2) What is the theoretical basis for why symmetric architecture preserves "degradation-carrying features" compared to asymmetric designs? (3) The choice of patch sizes (2 for bottleneck, 4 for decoder layers) appears arbitrary with no ablation justifying these values. Providing mathematical or theoretical analysis would elevate the contribution.

- **Semantic Guidance Component Shows Marginal Gains with Limited Analysis**: While SE-SymUNet shows improvements over SymUNet, the gains are modest (0.15 dB for bidirectional guidance in Table 4, averaging 0.09-0.15 dB per task). More critically, the paper lacks detailed analysis of when and why semantic guidance helps. The claim that "explicit semantic guidance is particularly beneficial for resolving spatially extensive degradations" (Section 4.3.1) regarding dehazing's 0.62 dB gain is not rigorously validated. An ablation showing which decoder layers benefit most from guidance would strengthen this claim.

- **Limited Evaluation of Generalization to Out-of-Distribution Scenarios**: The evaluation uses paired training-test scenarios with consistent image statistics. The paper does not thoroughly evaluate robustness to distribution shifts—e.g., real-world noise characteristics versus synthetic BSD68 noise, or domain adaptation to real degradations. Given the claimed advantage in handling "unknown degradations," more extensive evaluation on naturally degraded images or cross-dataset generalization would be valuable. This gap is particularly important for "all-in-one" restoration claims.

- **Missing Comparisons with Recent Lightweight Baselines**: While the paper compares with many methods, most competing approaches are relatively complex (MoE-based, diffusion-based, prompt-based). A more comprehensive comparison with other lightweight baselines that similarly prioritize efficiency would strengthen the efficiency claims. Additionally, the comparison with AirNet (which also has fewer parameters) could be more detailed in analyzing where SymUNet's gains come from relative to this closest efficient baseline.

</div>

<div class="lang-zh" style="display:none;">

## 劣勢

- **核心架構中的新穎性有限**: 雖然對稱U-Net設計是有效的，但基礎架構本身相對標準，並不特別新穎。主要貢獻是表明這個「更簡單」的設計優於複雜的近期方法，而不是引入架構創新。如果論文提供了更深入的理論分析或新穎架構成分來解釋*為什麼*對稱性除了直觀特徵流論據之外是有益的，論文會更強。對從Restormer改編的現有Transformer塊的依賴進一步限制了架構新穎性。

- **對失敗案例和局限性的分析不完整**: 論文沒有徹底討論提議方法表現不佳的場景。圖5顯示了成功案例的視覺比較，但沒有討論SymUNet困難的降噪或圖像類型。例如，低光增強結果（LOL基準）顯示更適度的改進，論文沒有分析為什麼語義指導在某些任務（去霾）中比其他任務（低光）更有幫助。對失敗模式的坦誠討論會加強該工作的可信度。

- **設計選擇的理論基礎有限**: 雖然論文通過消融提供了經驗驗證，但幾個設計選擇缺乏理論基礎。例如：(1)為什麼加法跳過連接除了經驗結果之外本質上比連接更好？(2)為什麼相比非對稱設計對稱架構保留「降噪特性信息」的理論基礎是什麼？(3)補丁大小的選擇（瓶頸為2，解碼器層為4）看起來任意，沒有消融來證明這些值。提供數學或理論分析會提升貢獻。

- **語義指導成分顯示邊際增益並分析有限**: 雖然SE-SymUNet顯示相比SymUNet的改進，但增益適度（表4中雙向指導為0.15 dB，平均每任務0.09-0.15 dB）。更關鍵的是，論文缺乏對語義指導何時以及為什麼有幫助的詳細分析。關於「明確語義指導特別有利於解決空間廣泛的降噪」（第4.3.1節）的主張，有關去霾的0.62 dB增益沒有被嚴格驗證。顯示哪些解碼器層從指導中獲益最多的消融會加強這個主張。

- **對分布外場景泛化的評估有限**: 評估使用具有一致圖像統計數據的配對訓練-測試場景。論文沒有徹底評估對分布轉移的魯棒性——例如，實際噪聲特徵與合成BSD68噪聲，或對實際降噪的域自適應。鑒於在處理「未知降噪」方面聲稱的優勢，對自然降噪圖像或跨數據集泛化的更廣泛評估會很有價值。這個差距對於「全合一」修復主張特別重要。

- **與近期輕量級基線的比較缺失**: 雖然論文與許多方法進行了比較，但大多數競爭方法相對複雜（基於MoE、基於擴散、基於提示）。與同樣優先考慮效率的其他輕量級基線進行更全面的比較會加強效率主張。此外，與AirNet（參數也更少）的比較可以在分析SymUNet相對於這個最接近的有效基線的增益來自何處方面更詳細。

</div>

<div class="lang-en">

## Research Directions

- **Theoretical Analysis of Degradation-Carrying Feature Encoding**: A compelling follow-up would provide rigorous mathematical analysis of how and why well-designed encoders inherently encode degradation information. This could involve: (1) Information-theoretic analysis measuring mutual information between encoded features and degradation types, (2) feature visualization and interpretability studies using techniques like Grad-CAM to identify which feature channels correspond to specific degradations, (3) theoretical frameworks explaining why symmetric architectures preserve this information better. Such work would elevate the paper from empirical validation to principled understanding, potentially enabling design of architectures optimized for degradation preservation from first principles.

- **Adaptive Semantic Guidance with Task-Specific Feature Selection**: While SE-SymUNet uses static CLIP features, future work could develop adaptive semantic priors that dynamically select relevant semantic information based on detected degradation types. Specific approach: (1) Design a degradation classifier that operates on encoder features to identify task type with high confidence, (2) condition the semantic guidance mechanism to selectively attend to relevant CLIP semantic tokens, (3) evaluate on mixed-degradation scenarios where multiple degradations co-occur. This would address the observation that semantic guidance helps dehazing more than low-light enhancement, and could yield significantly larger gains than the 0.15 dB currently observed.

- **Extension to Video Restoration and Temporal Consistency**: The paper focuses on image restoration; extending to video restoration would be impactful. The symmetric architecture naturally accommodates temporal information through recurrent or attention-based modules. Specific approach: (1) incorporate temporal skip connections across adjacent frames within the symmetric U-Net, (2) design bidirectional temporal guidance where frame features and semantic context evolve across time, (3) evaluate on video denoising, deraining, and deblurring benchmarks. This would maintain the simplicity philosophy while addressing a practically important problem and could be a strong CVPR/ICCV submission.

- **Composite Degradation Handling with Multi-Scale Degradation Analysis**: Current evaluation uses single or simple combinations of degradations. Real-world images often exhibit complex composite degradations (blur + rain + low-light). Approach: (1) develop interpretable degradation analysis modules that decompose images into component degradations, (2) extend SymUNet with hierarchical multi-degradation branches that operate at different scales, (3) design loss functions that balance restoration quality across degradation components, (4) create large-scale datasets with realistic composite degradations. This addresses a fundamental limitation of current all-in-one methods and would be highly relevant for deployment.

- **Mobile and Edge Deployment with Quantization and Pruning**: While SymUNet is computationally efficient, deploying to mobile/edge devices requires further optimization. Approach: (1) design quantization-aware training schemes that maintain accuracy under 8-bit or 4-bit quantization, (2) conduct structured pruning targeting redundant channels in layers less sensitive to degradation information, (3) develop knowledge distillation methods to train ultra-lightweight student models from SymUNet, (4) benchmark on mobile platforms (iPhone, Android) and edge devices (NVIDIA Jetson). This would make the method practically valuable for real applications and could yield a strong IEEE TPAMI or MobileAI workshop paper.

- **Unified Architecture for Image and Video Processing**: The simplicity of SymUNet suggests it could be unified with other restoration tasks beyond all-in-one degradation removal—e.g., super-resolution, style transfer, inpainting. Approach: (1) analyze which architectural properties (symmetry, additive fusion) generalize across tasks, (2) design a minimal parameterized variant that handles multiple task types through lightweight task-specific adapters, (3) evaluate on comprehensive multi-task benchmarks. This could establish SymUNet as a foundational architecture for diverse restoration tasks, with broader impact on the field.

- **Robustness Evaluation Against Adversarial Degradations and Domain Shifts**: Strengthen claims about handling "unknown degradations" through rigorous evaluation. Approach: (1) evaluate on naturally degraded images from real-world sources (actual photographs taken in adverse weather vs. synthetic benchmarks), (2) conduct cross-dataset evaluation measuring generalization from training benchmarks to held-out datasets, (3) analyze robustness to adversarial degradations designed to fool the restoration network, (4) develop uncertainty quantification methods to identify when the model is confident versus uncertain. This would provide honest assessment of practical applicability and identify failure modes for future improvement.

</div>

<div class="lang-zh" style="display:none;">

## 研究方向

- **降噪特性特徵編碼的理論分析**: 一個引人矚目的後續工作應該提供嚴格的數學分析，說明精心設計的編碼器如何以及為什麼本質上編碼降噪信息。這可能涉及：(1)測量編碼特徵和降噪類型之間互信息的信息論分析，(2)使用Grad-CAM等技術的特徵可視化和可解釋性研究，以識別哪些特徵通道對應於特定降噪，(3)解釋為什麼對稱架構保留這些信息更好的理論框架。這樣的工作會將論文從經驗驗證提升到原則性理解，可能能夠從第一原理設計針對降噪保留優化的架構。

- **具有任務特定特徵選擇的自適應語義指導**: 雖然SE-SymUNet使用靜態CLIP特徵，但未來工作可以開發動態語義先驗，根據檢測到的降噪類型動態選擇相關語義信息。具體方法：(1)設計在編碼器特徵上操作的降噪分類器，以高置信度識別任務類型，(2)條件化語義指導機制以選擇性地關注相關CLIP語義令牌，(3)在多個降噪共現的混合降噪場景上評估。這將解決語義指導在去霾中比在低光增強中幫助更多的觀察，並且可能產生遠大於目前觀察到的0.15 dB的增益。

- **擴展到視頻修復和時間一致性**: 論文重點關注圖像修復；擴展到視頻修復將產生影響。對稱架構通過循環或注意力的模塊自然適應時間信息。具體方法：(1)在對稱U-Net內相鄰幀之間納入時間跳過連接，(2)設計雙向時間指導，其中幀特徵和語義上下文隨時間進化，(3)評估視頻去噪、去雨和去模糊基準。這將維持簡潔哲學，同時解決實踐上重要的問題，可能是強大的CVPR/ICCV提交。

- **具有多尺度降噪分析的複合降噪處理**: 當前評估使用單一或簡單的降噪組合。實際圖像通常表現出複雜的複合降噪（模糊+雨+低光）。方法：(1)開發可解釋的降噪分析模塊，將圖像分解為成分降噪，(2)使用在不同尺度上操作的分層多降噪分支擴展SymUNet，(3)設計在降噪成分之間平衡修復質量的損失函數，(4)創建具有現實複合降噪的大規模數據集。這解決了當前全合一方法的根本限制，對部署高度相關。

- **移動和邊緣部署與量化和修剪**: 雖然SymUNet計算上有效，但部署到移動/邊緣設備需要進一步優化。方法：(1)設計量化感知訓練方案，在8位或4位量化下保持準確性，(2)進行結構化修剪，針對對降噪信息不敏感的層中的冗餘通道，(3)開發知識蒸餾方法，從SymUNet訓練超輕量級學生模型，(4)在移動平台（iPhone、Android）和邊緣設備（NVIDIA Jetson）上進行基準測試。這將使該方法對實際應用實踐有價值，並可能產生強大的IEEE TPAMI或MobileAI研討會論文。

- **圖像和視頻處理統一架構**: SymUNet的簡潔性建議它可以與全合一降噪移除之外的其他修復任務統一——例如，超分辨率、風格轉移、修復。方法：(1)分析哪些架構屬性（對稱性、加法融合）跨任務泛化，(2)設計最小參數化變體，通過輕量級任務特定適配器處理多種任務類型，(3)在綜合多任務基準上評估。這可能會將SymUNet建立為多種修復任務的基礎架構，對該領域有更廣泛的影響。

- **對對抗性降噪和域轉移的魯棒性評估**: 通過嚴格評估加強關於處理「未知降噪」的主張。方法：(1)在實際來源的自然降噪圖像上評估（在不利天氣中拍攝的實際照片與合成基準相比），(2)進行跨數據集評估，測量從訓練基準到保留數據集的泛化，(3)分析對設計為愚弄修復網絡的對抗性降噪的魯棒性，(4)開發不確定性量化方法，以識別何時模型有信心與不確定。這將提供對實踐適用性的誠實評估，並識別未來改進的失敗模式。

</div>

---


