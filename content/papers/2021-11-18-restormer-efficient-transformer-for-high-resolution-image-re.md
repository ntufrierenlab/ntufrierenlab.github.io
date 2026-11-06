---
title: "Restormer: Efficient Transformer for High-Resolution Image Restoration"
date: 2021-11-18
authors:
  - "Syed Waqas Zamir"
  - "Aditya Arora"
  - "Salman Khan"
  - "Munawar Hayat"
  - "Fahad Shahbaz Khan"
  - "Ming-Hsuan Yang"
source: "arXiv"
arxiv_url: "https://arxiv.org/abs/2111.09881"
pdf_url: "https://arxiv.org/pdf/2111.09881"
one_line_summary: "Restormer proposes an efficient Transformer for high-resolution image restoration through transposed channel-dimension attention (MDTA) and gated feed-forward networks (GDFN), achieving state-of-the-art results across six image restoration tasks with significantly reduced computational complexity."
one_line_summary_zh: "Restormer通過轉置通道維度注意力（MDTA）和門控前饋網路（GDFN）提出了一個用於高分辨率圖像復原的高效Transformer，在六個圖像復原任務上達到最先進結果，同時大幅降低計算複雜度。"
date_added: 2026-04-14
topics: ["Image Deblurring"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **Multi-Dconv Head Transposed Attention (MDTA) Module**: The paper introduces a novel attention mechanism that applies self-attention across the feature channel dimension rather than the spatial dimension, reducing computational complexity from O(W²H²) to linear complexity. This is achieved by computing cross-covariance across channels to generate attention maps, while incorporating depth-wise convolutions for local context enrichment before attention computation. This key innovation enables processing of high-resolution images without disintegrating them into local windows, preserving global context crucial for image restoration tasks.

- **Gated-Dconv Feed-Forward Network (GDFN)**: A reformulated feed-forward network that replaces standard fully-connected layers with a gating mechanism based on element-wise multiplication of two projection paths (one with GELU activation). The GDFN incorporates depth-wise convolutions to encode spatial context and controls information flow through hierarchical levels, allowing selective feature propagation. This design differs from conventional Transformer feed-forward networks by combining gating with local convolutions, improving representation learning for image restoration.

- **Efficient Transformer Architecture for Image Restoration**: Restormer is presented as an encoder-decoder Transformer with 4-level hierarchical multi-scale design that maintains computational efficiency while processing high-resolution images without patch division. The architecture incorporates progressive learning strategy where patch sizes gradually increase during training (from 128² to 384²) with corresponding batch size reduction to maintain consistent training time, enabling the network to learn both local and global image statistics.

- **Comprehensive Multi-Task Evaluation**: The paper demonstrates state-of-the-art performance across six image restoration tasks on 16 benchmark datasets: image deraining (5 datasets), single-image motion deblurring (4 datasets), defocus deblurring on single-image and dual-pixel data (1 dataset with 76 scenes), and image denoising on both synthetic (5 datasets with varying noise levels) and real-world data (2 datasets). This breadth of evaluation validates the generalizability of the proposed method across diverse restoration problems.

- **Significant Computational Efficiency Improvements**: Compared to prior state-of-the-art methods, Restormer achieves substantial reduction in computational overhead: 81% fewer FLOPs than MPRNet with 0.26 dB performance gain on deblurring, 3.14× fewer FLOPs than SwinIR with faster inference (13× speedup), and 4.4× fewer parameters than IPT with 29× faster inference on deblurring. This addresses a critical limitation of Transformers in image restoration and makes the approach practical for high-resolution applications.

## Core Insights

- **Channel-Dimension Attention as a Solution to High-Resolution Processing**: The fundamental insight is that self-attention can be reformulated to operate across feature channels rather than spatial positions, implicitly modeling global pixel relationships while maintaining linear complexity. The paper demonstrates through ablation (Table 7c: 0.32 dB PSNR gain) and visual results that this transposed attention maintains the ability to capture long-range dependencies essential for restoration while being computationally tractable. This shifts the paradigm from explicitly modeling pixel-pair interactions to computing feature covariance, a more efficient alternative that does not sacrifice modeling capacity.

- **Local Context Mixing is Complementary to Global Attention**: The integration of depth-wise convolutions in both MDTA and GDFN modules reveals that spatially local processing through convolutions and globally contextualized attention are complementary rather than contradictory. Table 7 shows that removing local mechanisms from MDTA results in performance degradation, suggesting that local feature aggregation provides essential spatial structure information that enhances the effectiveness of covariance-based attention. This insight validates combining CNN and Transformer strengths within a unified framework.

- **Progressive Learning Addresses the Patch-vs-Global-Statistics Dilemma**: The paper identifies that training Transformers on fixed small patches (128²) fails to capture global image statistics needed for high-resolution inference, leading to suboptimal performance. Table 9 demonstrates that progressive learning (increasing patch size from 128² to 384²) improves PSNR by 0.07 dB over fixed training with equivalent computational cost. This insight shows that curriculum-like training strategies are essential for Transformers in image restoration, similar to how curriculum learning benefits complex task learning, and this training strategy transfers well to varying resolutions at test time.

- **Architectural Design Choices Enable Both Quality and Efficiency**: The ablation studies (Tables 7-10) collectively show that the specific combination of architectural components yields compounding benefits: MDTA (0.32 dB), GDFN (0.26 dB), concatenation without channel reduction at level-1 (0.04 dB), and refinement stage (0.05 dB) together contribute 0.51 dB PSNR gain. The choice of deep-narrow architecture over wide-shallow (Table 10) demonstrates that depth is more important than width for accuracy, while wider models achieve faster inference—revealing a fundamental trade-off in Transformer design for restoration.

- **Single-Model Generalization Across Tasks and Datasets**: Notably, Restormer is trained on individual tasks but demonstrates strong cross-dataset generalization. For motion deblurring, the model trained only on GoPro achieves state-of-the-art on unseen datasets (HIDE, RealBlur-R, RealBlur-J), with 36.19 dB PSNR on HIDE compared to 29.99 dB for the previous best method that was specifically trained on that dataset. This suggests that the proposed architecture learns fundamental restoration priors that generalize across different degradation distributions and camera settings, a significant advantage over task-specific or dataset-specific methods.

- **Linear Complexity Enables Practical High-Resolution Processing**: The mathematical reformulation from O(W²H²) standard attention to linear complexity MDTA is crucial for practical application. Processing 256×256 images requires 87.7 B FLOPs for the full Restormer (Table 7g), making it feasible for real-world deployment. In contrast, applying standard Transformer attention to such images would require quadratic memory and computation, making even 256×256 prohibitively expensive. This efficiency gain enables the model to handle the gradual increase in patch sizes during progressive learning without memory constraints.

## Key Data & Results

| Task | Dataset | Metric | Restormer | Previous Best | Improvement |
|------|---------|--------|-----------|---------------|-------------|
| Image Deraining | Test100 | PSNR | 32.00 | 30.35 (SPAIR) | +1.65 dB |
| Image Deraining | Rain100H | PSNR | 31.46 | 30.95 (SPAIR) | +0.51 dB |
| Image Deraining | Rain100L | PSNR | 38.99 | 36.93 (SPAIR) | +2.06 dB |
| Image Deraining | Test2800 | PSNR | 34.18 | 33.34 (SPAIR) | +0.84 dB |
| Image Deraining | Test1200 | PSNR | 33.19 | 33.04 (SPAIR) | +0.15 dB |
| Image Deraining | Average | PSNR | 33.96 | 32.91 (SPAIR) | +1.05 dB |
| Motion Deblurring | GoPro | PSNR | 31.22 | 32.66 (MPRNet) | -1.44 dB |
| Motion Deblurring | HIDE | PSNR | 36.19 | 29.99 (SPAIR) | +6.20 dB |
| Motion Deblurring | RealBlur-R | PSNR | 35.79 | 35.26 (DMPHN) | +0.53 dB |
| Motion Deblurring | RealBlur-J | PSNR | 28.96 | 28.70 (DeblurGAN-v2) | +0.26 dB |
| Defocus Deblurring (Single) | DPDD Combined | PSNR | 25.98 | 25.37 (IFAN) | +0.61 dB |
| Defocus Deblurring (Dual) | DPDD Combined | PSNR | 26.66 | 25.99 (IFAN) | +0.67 dB |
| Gaussian Denoising (σ=50) | Urban100 | PSNR | 28.29 | 28.00 (SwinIR) | +0.29 dB |
| Gaussian Denoising (σ=50) | BSD68 | PSNR | 26.62 | 26.58 (SwinIR) | +0.04 dB |
| Real Denoising | SIDD | PSNR | 40.02 | 39.77 (Uformer) | +0.25 dB |
| Real Denoising | DND | PSNR | 40.03 | 39.96 (Uformer) | +0.07 dB |

**Key Quantitative Results:**

- **Deraining Performance**: Restormer achieves 33.96 dB average PSNR across five deraining benchmarks with 1.05 dB improvement over previous best (SPAIR), with particularly strong gains on Rain100L (2.06 dB). SSIM scores are consistently superior, with 0.935 average SSIM compared to 0.926 for SPAIR.

- **Deblurring with Strong Cross-Dataset Generalization**: On motion deblurring, despite achieving 31.22 dB on GoPro (0.26 dB below MPRNet), Restormer dramatically outperforms on HIDE (36.19 dB vs 29.99 dB for SPAIR—a 6.20 dB gap), RealBlur-R (35.79 dB), and RealBlur-J (28.96 dB). This demonstrates superior generalization despite being trained only on GoPro, with inference 29× faster than IPT and 81% fewer FLOPs than MPRNet.

- **Real Image Denoising Milestone**: Restormer becomes the first method to exceed 40 dB PSNR on both SIDD (40.02 dB) and DND (40.03 dB) real denoising benchmarks. This represents a significant performance milestone compared to previous best Uformer (39.77 dB SIDD, 39.96 dB DND), with 0.25-0.07 dB improvements despite having 3.14× fewer FLOPs and 13× faster inference.

- **Ablation Study Contributions**: Progressive analysis shows individual component contributions—MDTA provides 0.32 dB gain (Table 7c), GDFN provides additional 0.26 dB over standard FN (Table 7f), concatenation without channel reduction at level-1 adds 0.04 dB (Table 8), and refinement stage adds 0.05 dB (Table 8). Combined, these innovations yield 0.51 dB PSNR improvement over baseline UNet with ResBlocks on Gaussian denoising.

- **Computational Efficiency Validation**: Figure 1 comprehensively demonstrates state-of-the-art performance with low computational cost across all tasks. For deblurring, Restormer uses 240 GFLOPs versus 740 GFLOPs for MPRNet while achieving superior performance; for Gaussian denoising, 240 GFLOPs versus 740 for SwinIR; for real denoising, 240 GFLOPs versus 640+ for competing methods. This efficiency-performance trade-off is fundamental to the method's practical applicability.

## Strengths

- **Novel Technical Contribution with Solid Mathematical Foundation**: The transposed attention mechanism elegantly solves the quadratic complexity problem of standard Transformers by reformulating attention computation from spatial dimension (HW×HW matrices) to channel dimension (C×C matrices). This is mathematically sound and well-motivated—the paper clearly shows how local context mixing via 1×1 and 3×3 depth-wise convolutions before covariance computation preserves the ability to capture spatial relationships. The novelty is significant compared to prior work (e.g., Swin Transformer's local windows, IPT's patch division) as it maintains global connectivity while achieving linear complexity.

- **Comprehensive and Fair Experimental Evaluation**: The paper evaluates Restormer across six different restoration tasks on 16 benchmark datasets with multiple metrics (PSNR, SSIM, MAE, LPIPS where applicable). All baselines are contemporary and diverse, including CNNs (DRUNet, MPRNet), recent Transformers (SwinIR, Uformer, IPT), and task-specific methods. The experiments follow standard evaluation protocols, and comparisons use the Y-channel in YCbCr space consistent with prior work. Ablation studies systematically validate each component's contribution (Tables 7-10), including progressive learning, architectural choices, and depth vs. width trade-offs.

- **Consistent State-of-the-Art Performance with Practical Efficiency**: Rather than cherry-picking one or two tasks, Restormer consistently achieves state-of-the-art results across all evaluated tasks, with particularly impressive gains on deraining (1.05 dB average) and real denoising (first to exceed 40 dB). The computational efficiency is exceptional—3.14× fewer FLOPs than SwinIR with 13× faster inference, 81% fewer FLOPs than MPRNet with superior performance. This demonstrates that efficiency gains were not traded for accuracy.

- **Strong Generalization and Reproducibility**: The model trained on GoPro dataset generalizes exceptionally well to unseen deblurring datasets (HIDE, RealBlur), suggesting learned restoration priors are robust. Implementation details are thorough: architecture specifications (number of blocks, heads, channels at each level), training procedures (learning rate schedule, optimizer, loss function, patch sizes during progressive learning), and data augmentation strategies are clearly documented. Code and pre-trained models are promised to be released, enhancing reproducibility.

- **Well-Motivated Design Choices with Supporting Evidence**: The paper provides clear justification for each design decision. Progressive learning is motivated by the curriculum learning principle (Section 3.3) and validated by experiments (Table 9). The combination of gating mechanism in GDFN with depth-wise convolutions is explained through information flow control. The choice of deep-narrow architecture over wide-shallow is empirically validated (Table 10), showing accuracy-speed trade-offs. The paper avoids unjustified architectural decisions.

- **Addresses a Critical Problem in Vision Transformers**: The paper tackles a genuine bottleneck limiting Transformer applicability to high-resolution image tasks—quadratic complexity preventing processing of full-resolution images. The solution is elegant and practical, enabling the community to leverage Transformers' benefits (content-adaptive processing, long-range dependencies) for image restoration without architectural compromises. This opens new research directions and validates Transformers for low-level vision tasks.

## Weaknesses

- **Limited Theoretical Analysis of Transposed Attention**: While the MDTA mechanism is intuitive, the paper lacks formal analysis of what implicit global relationships are captured when applying attention across channels versus spatial dimensions. The claim that covariance-based attention "implicitly models contextualized global relationships between pixels" (Section 3.1) is not rigorously justified. Visualization of learned attention maps or feature correlations would strengthen understanding of how channel-dimension attention differs fundamentally from spatial attention in capturing image structure. The connection between feature covariance and pixel-pixel dependencies could be more explicitly established.

- **Progressive Learning Lacks Systematic Investigation**: The progressive learning strategy (Section 3.3) is presented somewhat superficially. The paper provides limited analysis of how patch size progression affects learning dynamics or why the specific sequence [128², 192², 256², 320², 384²] at iterations [92K, 156K, 204K, 240K, 276K] was chosen. Table 9 shows only one progressive schedule; alternative schedules (e.g., different growth rates, different stopping sizes) are not explored. The improvement of 0.07 dB is modest—systematic ablation of schedule design would clarify how critical specific choices are versus the general principle of progressive training.

- **Missing Comparisons and Analysis on Computational Cost Claims**: While FLOPs comparisons are provided, actual runtime measurements are limited. The claim of "13× faster than SwinIR" (Section 4.4) is mentioned once without detailed analysis. Memory consumption comparisons with baselines are absent—crucial for practical deployment on GPUs with limited memory. For a paper emphasizing computational efficiency, more comprehensive profiling (memory usage, actual inference time on different image resolutions, throughput) would strengthen claims and help practitioners understand practical deployment trade-offs.

- **Incomplete Analysis of Failure Cases and Limitations**: The paper does not discuss scenarios where Restormer underperforms. For motion deblurring on GoPro (Table 2), Restormer achieves 31.22 dB versus 32.66 dB for MPRNet—a notable gap on the training dataset itself. The paper does not explain why the network performs worse on the dataset it trains on, limiting understanding of when the method is inappropriate. No analysis of difficult restoration scenarios (extreme noise levels, highly textured regions, edge artifacts) is provided.

- **Potentially Overclaimed Efficiency Due to Architecture Differences**: Comparisons with SwinIR and other methods are not entirely apples-to-apples. Restormer uses a 4-level encoder-decoder while some baselines may have different depths. The refinement stage (Figure 2) adds additional computation not present in baseline methods. While ablations show refinement adds 0.05 dB (Table 8), the FLOPs accounting for this component in the overall efficiency claims could be clearer. The paper should explicitly decouple efficiency gains from architectural depth versus genuine algorithmic improvements.

- **Limited Analysis of Cross-Task Knowledge Transfer and Generalization Boundaries**: Although Restormer achieves strong results on six tasks, the paper trains separate models for each task (Section 4, "Implementation Details: We train separate models for different image restoration tasks"). The claim of generalization relies mainly on cross-dataset evaluation (e.g., model trained on GoPro tested on HIDE). Whether the learned Transformer modules could benefit multi-task learning, or what factors determine when single-task training is optimal, remains unexplored. This limits understanding of the method's potential for practical systems requiring handling multiple degradations.

## Research Directions

- **Adaptive Spatial-Channel Attention Mechanisms**: Extend the transposed attention paradigm by developing adaptive mechanisms that dynamically select between spatial and channel attention based on image content. For instance, implement a learnable gating that routes each feature block through either MDTA (channel-wise) or standard spatial attention depending on the presence of large-scale structures. This would combine the efficiency of MDTA with the explicit spatial modeling of standard attention when beneficial. A meta-learning approach could learn to select appropriate attention types for different degradation types, advancing toward a unified restoration model.

- **Efficient Multi-Task Restoration Framework**: Design an architecture that can handle multiple restoration tasks (deraining, deblurring, denoising) with a single unified model through task-specific adapters or conditional modules inserted into Restormer blocks. Given the paper's per-task training requirement and strong cross-dataset generalization, a multi-task variant could investigate shared feature representations and task-specific refinements. This would reduce deployment overhead and enable efficient handling of images with multiple degradations, a practical requirement for real-world applications.

- **Hierarchical and Multi-Resolution Progressive Learning Schemes**: Investigate more sophisticated progressive learning strategies beyond simple patch size increase. Consider multi-branch architectures where different resolution streams train progressively, or adaptive curriculum strategies that adjust difficulty based on loss plateaus rather than fixed schedules. Combine with meta-learning to learn optimal progression sequences for different restoration tasks. Empirically validate that learned curricula outperform hand-designed schedules, potentially improving upon the modest 0.07 dB gains observed in Table 9.

- **Theoretical Analysis of Channel-Dimension Attention and Feature Covariance**: Provide formal mathematical analysis connecting covariance-based attention to pixel-pixel relationships. Develop theoretical bounds on what spatial relationships can be captured through feature covariance and how they relate to kernel receptive fields. Conduct feature visualization studies and attention map analysis to empirically demonstrate what global structures are implicitly modeled. This could lead to principled improvements to MDTA and help understand when transposed attention is preferable to spatial attention.

- **Integration with Neural Architecture Search for Efficient Restoration**: Apply AutoML techniques to automatically discover optimal configurations of Restormer components (number of levels, blocks per level, channel configurations, attention head configurations) under different computational budgets. The paper's architecture (Table 7) was manually designed; NAS could explore the design space more comprehensively. Generate efficient variants targeting specific deployment scenarios (mobile devices, edge computing, real-time processing). This addresses the paper's limitation of providing only one model configuration.

- **Extending to Video Restoration and Temporal Modeling**: Adapt Restormer to video restoration tasks (denoising, deblurring, deraining in videos) by incorporating temporal dimensions into the attention mechanism. Explore whether transposed attention can efficiently model temporal correlations across frames. Video restoration is commercially important and presents distinct challenges—local temporal consistency while capturing long-range spatiotemporal dependencies. The efficiency of MDTA makes temporal extension more feasible than standard Transformers.

- **Adversarial Robustness and Uncertainty Quantification**: Investigate robustness of Restormer to adversarial perturbations and distribution shift (e.g., images with different noise types than training data). Implement Bayesian variants of Restormer to quantify uncertainty in predictions, useful for critical applications (medical imaging). Combine with recent work on certified robustness to provide guarantees on restoration performance bounds. This extends the paper's focus on accuracy to broader reliability concerns important for practical deployment.

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **多卷積頭轉置注意力（Multi-Dconv Head Transposed Attention, MDTA）模塊**：本文提出了一種新穎的注意力機制，將自注意力從空間維度改為沿特徵通道維度應用，將計算複雜度從O(W²H²)降低到線性。通過計算通道間的交叉協方差來生成注意力圖，同時在注意力計算前利用深度卷積進行局部上下文豐富化。這一關鍵創新使高分辨率圖像無需分割成局部窗口即可處理，保留了圖像復原任務中至關重要的全局上下文。

- **門控深度卷積前饋網路（Gated-Dconv Feed-Forward Network, GDFN）**：通過基於兩條投影路徑（其中一條使用GELU激活）的逐元素乘法門控機制，重新設計了標準的全連接層前饋網路。GDFN融合了深度卷積來編碼空間上下文，並控制資訊在階層級別間的流動，允許選擇性特徵傳播。這一設計與傳統Transformer前饋網路不同，通過結合門控機制與局部卷積，改進了圖像復原的表示學習能力。

- **適用於圖像復原的高效Transformer架構**：Restormer是一個4層級的編碼器-解碼器Transformer，採用分層多尺度設計，在無需分割高分辨率圖像的情況下保持計算效率。該架構融合了漸進式學習策略，在訓練過程中逐步增大補丁尺寸（從128²增加到384²），同時相應降低批次大小以保持一致的訓練時間，使網絡同時學習局部和全局圖像統計特徵。

- **全面的多任務評估**：本文在6個圖像復原任務的16個基準資料集上展示了最先進的性能：圖像除雨（5個資料集）、單圖像運動去模糊（4個資料集）、焦點去模糊（包括單圖像和雙像素數據，76個場景）以及圖像去噪（5個合成資料集和2個真實資料集，涵蓋多種噪聲等級）。這種廣泛的評估驗證了所提方法在多樣復原問題上的可泛化性。

- **計算效率的顯著提升**：與先前的最先進方法相比，Restormer實現了大幅的計算開銷降低：相比MPRNet減少81% FLOPs且性能提升0.26 dB、相比SwinIR減少3.14倍FLOPs且推理速度快13倍、相比IPT減少4.4倍參數且去模糊推理快29倍。這解決了Transformer在圖像復原中的關鍵限制，使該方法在高分辨率應用中具有實用性。

## 核心洞見

- **通道維度注意力作為高分辨率處理的解決方案**：基本洞見是自注意力可被重新表述為沿特徵通道而非空間位置運行，隱含地模擬全局像素關係而保持線性複雜度。論文通過消融（表7c：PSNR提升0.32 dB）和視覺結果證明，這種轉置注意力在保持捕捉復原必需的長距離依賴關係能力的同時，在計算上可行。這轉變了範式，從顯式模擬像素對相互作用改為計算特徵協方差，一種更高效的替代方案，不犧牲建模能力。

- **局部上下文混合對全局注意力具有互補性**：在MDTA和GDFN模塊中整合深度卷積揭示了通過卷積進行的空間局部處理與全局上下文注意力是互補而非矛盾的。表7顯示，從MDTA中移除局部機制導致性能下降，表明局部特徵聚合提供對空間結構的必要資訊，增強了基於協方差的注意力的有效性。這一洞見驗證了在統一框架內結合CNN和Transformer優勢的合理性。

- **漸進式學習解決補丁與全局統計的困境**：本文指出在固定小補丁（128²）上訓練Transformer無法捕捉高分辨率推理所需的全局圖像統計，導致次優性能。表9表明漸進式學習（將補丁尺寸從128²增加到384²）在計算成本相同的情況下相比固定訓練提升0.07 dB PSNR。這洞見表明課程式訓練策略對圖像復原中的Transformer至關重要，類似課程學習對複雜任務學習的益處，該訓練策略在不同測試解析度上泛化效果良好。

- **架構設計選擇同時實現質量與效率**：消融研究（表7-10）綜合顯示特定組件組合的複合效益：MDTA貢獻0.32 dB、GDFN貢獻額外0.26 dB（相比標準前饋網路）、第1層級無通道縮減的連接貢獻0.04 dB、優化階段貢獻0.05 dB，共計相比ResBlocks基線提升0.51 dB PSNR。深度-窄型架構優於寬型-淺型（表10）的選擇揭示深度對精度比寬度更重要，而寬型模型因並行化實現更快推理，顯示Transformer復原設計的基本折衷。

- **跨任務和資料集的單一模型泛化能力**：值得注意的是，Restormer對不同任務分別訓練，但展示出強跨資料集泛化。對運動去模糊，僅在GoPro上訓練的模型在未見資料集上達到最先進水平（HIDE 36.19 dB相比先前最佳29.99 dB、RealBlur-R 35.79 dB、RealBlur-J 28.96 dB）。這表明所提架構學習了跨不同降質分佈和相機設定的基礎復原先驗，相比特定任務或資料集方法具有顯著優勢。

- **線性複雜度實現高分辨率實際處理**：從O(W²H²)標準注意力到線性複雜度MDTA的數學重新表述對實際應用至關重要。處理256×256圖像需要完整Restormer的87.7 B FLOPs（表7g），使其對實際部署可行。相比之下，將標準Transformer注意力應用於此類圖像需要二次方記憶體和計算，使即使256×256也變得禁止性地昂貴。這一效率增益使模型能夠在漸進式學習期間無記憶體限制地處理逐漸增加的補丁尺寸。

## 關鍵數據與結果

| 任務 | 資料集 | 指標 | Restormer | 先前最佳 | 改進 |
|------|--------|------|-----------|---------|------|
| 除雨 | Test100 | PSNR | 32.00 | 30.35 (SPAIR) | +1.65 dB |
| 除雨 | Rain100H | PSNR | 31.46 | 30.95 (SPAIR) | +0.51 dB |
| 除雨 | Rain100L | PSNR | 38.99 | 36.93 (SPAIR) | +2.06 dB |
| 除雨 | Test2800 | PSNR | 34.18 | 33.34 (SPAIR) | +0.84 dB |
| 除雨 | Test1200 | PSNR | 33.19 | 33.04 (SPAIR) | +0.15 dB |
| 除雨 | 平均 | PSNR | 33.96 | 32.91 (SPAIR) | +1.05 dB |
| 去模糊 | GoPro | PSNR | 31.22 | 32.66 (MPRNet) | -1.44 dB |
| 去模糊 | HIDE | PSNR | 36.19 | 29.99 (SPAIR) | +6.20 dB |
| 去模糊 | RealBlur-R | PSNR | 35.79 | 35.26 (DMPHN) | +0.53 dB |
| 去模糊 | RealBlur-J | PSNR | 28.96 | 28.70 (DeblurGAN-v2) | +0.26 dB |
| 焦點去模糊（單圖像） | DPDD 綜合 | PSNR | 25.98 | 25.37 (IFAN) | +0.61 dB |
| 焦點去模糊（雙像素） | DPDD 綜合 | PSNR | 26.66 | 25.99 (IFAN) | +0.67 dB |
| 高斯去噪 (σ=50) | Urban100 | PSNR | 28.29 | 28.00 (SwinIR) | +0.29 dB |
| 高斯去噪 (σ=50) | BSD68 | PSNR | 26.62 | 26.58 (SwinIR) | +0.04 dB |
| 真實去噪 | SIDD | PSNR | 40.02 | 39.77 (Uformer) | +0.25 dB |
| 真實去噪 | DND | PSNR | 40.03 | 39.96 (Uformer) | +0.07 dB |

**關鍵定量結果：**

- **除雨性能**：Restormer在五個除雨基準資料集上達到33.96 dB平均PSNR，相比先前最佳（SPAIR）提升1.05 dB，在Rain100L上的提升特別突出（2.06 dB）。SSIM指標一貫優越，平均0.935相比SPAIR的0.926。

- **去模糊與強跨資料集泛化**：在運動去模糊上，儘管在GoPro上達到31.22 dB（相比MPRNet低0.26 dB），Restormer在HIDE上大幅超越（36.19 dB相比SPAIR 29.99 dB—超過6.20 dB的差距）、RealBlur-R上35.79 dB、RealBlur-J上28.96 dB。儘管僅在GoPro上訓練，這展示了優越的泛化能力，推理速度相比IPT快29倍，相比MPRNet減少81% FLOPs。

- **真實去噪里程碑**：Restormer成為首個在SIDD（40.02 dB）和DND（40.03 dB）真實去噪基準上均超過40 dB PSNR的方法。相比先前最佳Uformer（39.77 dB SIDD、39.96 dB DND），這代表顯著性能里程碑，同時FLOPs減少3.14倍，推理快13倍。

- **消融研究貢獻**：漸進式分析顯示個別組件貢獻—MDTA提供0.32 dB提升（表7c）、GDFN相比標準前饋網路額外提供0.26 dB（表7f）、第1層級無通道縮減的連接添加0.04 dB（表8）、優化階段添加0.05 dB（表8）。綜合而言，這些創新在高斯去噪上相比ResBlocks基線產生0.51 dB PSNR提升。

- **計算效率驗證**：圖1全面展示了所有任務上的最先進性能與低計算成本。對去模糊，Restormer使用240 GFLOPs相比MPRNet的740 GFLOPs，同時實現優越性能；對高斯去噪，240 GFLOPs相比SwinIR的740；對真實去噪，240 GFLOPs相比競爭方法的640+。這種效率-性能折衷是該方法實用可應用性的基礎。

## 優勢

- **具有紮實數學基礎的新穎技術貢獻**：轉置注意力機制優雅地解決了標準Transformer的二次複雜度問題，通過將注意力計算從空間維度（HW×HW矩陣）改為通道維度（C×C矩陣）重新表述。這在數學上嚴謹且充分動機—論文清楚地展示了如何通過1×1和3×3深度卷積在協方差計算前的局部上下文混合，保護捕捉空間關係的能力。相比先前工作（如Swin Transformer的局部窗口、IPT的補丁分割），該創新具有顯著意義，因其保持全局連接性的同時實現線性複雜度。

- **全面公正的實驗評估**：論文在16個基準資料集上跨6個不同復原任務評估Restormer，採用多種指標（PSNR、SSIM、MAE、LPIPS等）。所有基線均為當代且多樣的，包括CNN（DRUNet、MPRNet）、最新Transformer（SwinIR、Uformer、IPT）及特定任務方法。實驗遵循標準評估協議，比較使用與先前工作一致的YCbCr空間的Y通道。消融研究系統驗證每個組件貢獻（表7-10），包括漸進式學習、架構選擇及深度對寬度的權衡。

- **跨所有任務的一致最先進性能與實用效率**：Restormer不是在一兩個任務上表現突出，而是在所有評估任務上穩定達到最先進結果，在除雨（平均1.05 dB提升）和真實去噪（首次超過40 dB）上特別令人印象深刻。計算效率卓越—相比SwinIR FLOPs減少3.14倍且推理快13倍，相比MPRNet FLOPs減少81%且性能優越。這證明效率提升並未以準確度為代價。

- **強泛化能力與可復現性**：在GoPro資料集上訓練的模型在未見去模糊資料集（HIDE、RealBlur）上表現異常優異，表明學習的復原先驗是魯棒的。實現細節詳盡：架構規範（各層級的塊數、頭數、通道配置）、訓練程序（學習率計劃、優化器、損失函數、漸進式學習中的補丁尺寸）和數據增強策略清晰記錄。承諾發佈代碼和預訓練模型，增強可復現性。

- **設計選擇動機充分且有支撐證據**：論文為每項設計決策提供清晰理由。漸進式學習由課程學習原理動機（第3.3節）且通過實驗驗證（表9）。GDFN中門控機制與深度卷積的結合通過資訊流控制解釋。深度-窄型相比寬-淺型架構的選擇通過實驗驗證（表10），展示準確度-速度權衡。論文避免了無根據的架構決策。

- **解決Vision Transformer中的關鍵問題**：論文解決了限制Transformer在高分辨率圖像任務中應用的真正瓶頸—二次複雜度阻止全分辨率圖像處理。解決方案優雅實用，使社區能利用Transformer的優勢（內容自適應處理、長距離依賴）用於圖像復原，無需架構折衷。這為社區開啟新研究方向，驗證Transformer在低層次視覺任務上的可行性。

## 劣勢

- **轉置注意力缺乏理論分析**：儘管MDTA機制直觀，論文缺乏沿通道維度與空間維度應用注意力時隱含捕捉的全局關係的形式分析。聲稱協方差基注意力「隱含地模擬像素間的上下文全局關係」（第3.1節）缺乏嚴格論證。已學注意力圖或特徵相關性的視覺化將增強理解通道維度注意力與空間注意力在捕捉圖像結構上的根本差異。特徵協方差與像素-像素依賴的連接可更明確建立。

- **漸進式學習缺乏系統調查**：漸進式學習策略（第3.3節）呈現方式相當膚淺。論文對補丁尺寸漸進如何影響學習動態或為何選擇特定序列[128²、192²、256²、320²、384²]在迭代[92K、156K、204K、240K、276K]處提供有限分析。表9僅展示一種漸進計劃；替代計劃（如不同增長率、不同停止尺寸）未探索。0.07 dB的改進謙虛—計劃設計的系統消融將釐清特定選擇相比漸進訓練一般原則有多關鍵。

- **計算成本聲明中缺乏比較和分析**：儘管提供FLOPs比較，實際執行時間測量有限。「相比SwinIR快13倍」的聲明（第4.4節）僅提及一次，無詳細分析。與基線的記憶體消耗比較缺失—對實際GPU部署至關重要。對強調計算效率的論文而言，更全面的分析（記憶體使用、不同解析度上的實際推理時間、吞吐量）將強化聲明並幫助實踐者理解部署權衡。

- **缺乏失敗案例和限制分析**：論文未討論Restormer表現不佳的情景。在運動去模糊GoPro上（表2），Restormer達到31.22 dB相比MPRNet的32.66 dB—在訓練資料集本身上的顯著差距。論文未解釋為何網絡在其訓練資料集上表現更差，限制了對何時該方法不適當的理解。未提供對困難復原場景（極端噪聲等級、高紋理區域、邊界偽跡）的分析。

- **因架構差異潛在過度聲稱效率**：與SwinIR和其他方法的比較並非完全可比。Restormer使用4層級編碼器-解碼器而某些基線可能深度不同。優化階段（圖2）添加未呈現在基線方法中的額外計算。儘管消融顯示優化階段添加0.05 dB（表8），FLOPs在整體效率聲明中對該組件的計算可更清晰。論文應明確分離效率增益（源於架構深度）與真正算法改進。

- **跨任務知識遷移和泛化界限分析有限**：儘管Restormer在6個任務上成就卓著，論文對每個任務訓練單獨模型（第4節，「實現細節：我們為不同圖像復原任務訓練單獨模型」）。泛化聲明主要依賴跨資料集評估（如模型在GoPro上訓練在HIDE上測試）。所學Transformer模塊是否能受益多任務學習，或何種因素決定何時單任務訓練最優，仍未探索。這限制了對該方法在實際系統（需處理多重降質）中潛力的理解。

## 研究方向

- **自適應空間-通道注意力機制**：通過開發動態選擇機制，根據圖像內容在空間和通道注意力間動態選擇，擴展轉置注意力範式。例如，實現學習型門控，基於大尺度結構的存在，將每個特徵塊路由至MDTA（通道式）或標準空間注意力。這將結合MDTA的效率與標準注意力的顯式空間建模優勢（在有益時）。元學習方法可學習為不同降質類型選擇合適注意力類型，邁向統一復原模型。

- **高效多任務復原框架**：通過任務特定適配器或插入Restormer塊的條件模塊，設計能用單一統一模型處理多復原任務（除雨、去模糊、去噪）的架構。鑑於論文的逐任務訓練要求和強跨資料集泛化，多任務變體可調查共享特徵表示與任務特定優化。這將減少部署開銷，實現對包含多重降質的圖像的高效處理，實際應用的實際需求。

- **分層和多解析度漸進式學習方案**：除簡單補丁尺寸增加外，調查更複雜漸進學習策略。考慮具有漸進訓練的多分支架構（不同解析度流），或基於損失平台而非固定計劃調整難度的自適應課程策略。與元學習結合以學習最優漸進序列（針對不同復原任務）。經驗驗證所學課程優於手設計計劃，潛在改進表9觀察到的謙虛0.07 dB提升。

- **通道維度注意力和特徵協方差的理論分析**：提供連接協方差基注意力與像素-像素關係的形式數學分析。開發理論界限，說明通過特徵協方差可捕捉何種空間關係及其與核感受野的關聯。進行特徵視覺化研究和注意力圖分析，經驗展示隱含建模的全局結構。這可導致MDTA原則性改進並幫助理解何時轉置注意力優於空間注意力。

- **與神經架構搜索的整合以實現高效復原**：應用AutoML技術在不同計算預算下，自動發現Restormer組件的最優配置（層級數、各層級塊數、通道配置、注意力頭配置）。論文的架構（表7）為人工設計；NAS可更全面地探索設計空間。生成針對特定部署場景（行動設備、邊緣計算、實時處理）的高效變體。這解決論文僅提供一個模型配置的限制。

- **擴展至視頻復原和時間建模**：通過在注意力機制中納入時間維度，將Restormer適應視頻復原任務（去噪、去模糊、除雨）。探索轉置注意力是否能高效建模幀間時間相關性。視頻復原商業上重要且呈現不同挑戰—局部時間一致性同時捕捉長距離時空依賴。MDTA的效率使時間擴展相比標準Transformer更可行。

- **對抗魯棒性和不確定性量化**：調查Restormer對對抗擾動和分佈轉移（例，圖像含不同訓練資料集的噪聲類型）的魯棒性。實現Restormer的貝葉斯變體以量化預測中的不確定性，對關鍵應用（醫學成像）有用。結合近期認證魯棒性工作以提供復原性能界的保證。這將論文對準確度的焦點擴展至實際部署中重要的更廣泛可靠性關注。

</div>


