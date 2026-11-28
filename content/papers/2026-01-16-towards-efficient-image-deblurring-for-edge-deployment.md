---
title: "Towards Efficient Image Deblurring for Edge Deployment"
date: 2026-01-16
authors:
  - "Srinivas Miriyala"
  - "Sowmya Vajrala"
  - "Sravanth Kodavanti"
source: "arXiv"
arxiv_url: "https://arxiv.org/abs/2601.11685"
pdf_url: "https://arxiv.org/pdf/2601.11685"
one_line_summary: "This paper presents a hardware-aware optimization framework that restructures image deblurring networks (NAFNet) through sensitivity-guided block substitution, surrogate distillation, and training-free Bayesian search to achieve 1.25× on-device latency improvement and 55% GMACs reduction while maintaining competitive accuracy across motion and defocus deblurring tasks."
one_line_summary_zh: "本論文提出硬體感知優化框架，通過敏感性引導區塊替換、代理蒸餾和無訓練貝葉斯搜尋來重組影像去模糊網路，在保持競爭力準確性的同時實現1.25倍設備延遲改進和55% GMACs減少。"
date_added: 2026-02-13
topics: ["Image Deblurring"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **Hardware-Aware Adaptation Framework**: Proposes a black-box optimization method that restructures pre-trained deblurring models (specifically NAFNet) specifically for edge deployment on mobile NPUs through direct device profiling. Unlike traditional efficiency metrics (FLOPs, parameters), this framework uses actual hardware feedback to guide architectural decisions, addressing the critical gap between algorithmic efficiency and real deployment performance on devices like Samsung Galaxy S24 Ultra.

- **Sensitivity-Guided Network Surgery**: Introduces a technique to identify and replace latency-heavy but non-critical blocks in U-Net architectures using multiple saliency metrics (gradient norm, SNIP, GraSP, Fisher information, SynFlow). Low-saliency blocks identified through these metrics are substituted with hardware-friendly alternatives while preserving model accuracy, demonstrating that block-level profiling can effectively guide network restructuring.

- **Surrogate Distillation at Feature Level**: Proposes training lightweight surrogate blocks that approximate original blocks' functionality while being more hardware-amenable, trained via feature-wise knowledge distillation using only 20-25% of the full training set. This decouples accuracy optimization from inference optimization, enabling parallel training of multiple surrogates and reducing overall computational burden compared to end-to-end retraining approaches.

- **Training-Free Multi-Objective Bayesian Search**: Formulates network configuration selection as a combinatorial multi-objective optimization problem (m blocks × n alternatives per block) solved through Bayesian optimization without requiring retraining for each architecture. The method uses pre-trained surrogates to enable plug-and-play evaluation, reducing the 7^8 (~5.7M) search space to tractable evaluations of accuracy drop (PSNR loss) and profiled latency on target hardware.

- **Comprehensive Experimental Validation**: Demonstrates generality across multiple deblurring tasks (motion deblurring on GoPro/RealBlur-J/R/HIDE, defocus deblurring on DPDD) with extensive comparisons to state-of-the-art methods. Achieves 55% GMACs reduction compared to transformer-based SOTA (Restormer: 140 GMACs vs. Ours: 68 GMACs on GoPro) while maintaining competitive accuracy, and importantly demonstrates 1.25× latency improvement on commercial smartphones.

- **Practical Framework for Reconfigurable Deployment**: Establishes a repeatable methodology for adapting networks across different devices and tasks by decoupling accuracy from inference optimization, enabling rapid redeployment when hardware changes or new tasks emerge. This provides actionable insights on module-level bottlenecks and latency trade-offs without requiring architectural redesign from scratch.

## Core Insights

- **FLOPs-Latency Misalignment is Critical**: The paper reveals that traditional efficiency metrics like FLOPs and parameter counts fail to correlate with actual on-device latency. NAFNet, considered "efficient" theoretically, still suffers from latency bottlenecks (177ms baseline) due to channel-attention modules on specialized hardware accelerators. This insight fundamentally motivates the shift from proxy metrics to direct hardware profiling, demonstrating that even 36-block architectures benefit significantly from hardware-aware restructuring without complete redesign.

- **Saliency Estimation Guides Safe Surgery**: The use of multiple saliency metrics identifies that certain blocks (e.g., 4th encoder in NAFNet with 28 NAF blocks) are critical for accuracy despite high computational cost, while others are replaceable with minimal accuracy loss. The authors excluded high-saliency blocks from substitution, demonstrating that accuracy preservation depends on preserving network sensitivity rather than raw parameter counts. This selective approach maintains the original model's robustness while achieving 1.25× speedup.

- **Feature-Level Distillation Enables Efficient Adaptation**: Training surrogate blocks using only 20-25% of the full dataset through feature-wise distillation (rather than end-to-end loss) reduces training overhead while enabling parallel surrogate optimization. This insight decouples accuracy restoration from the original network's training dynamics, making the framework modular and enabling rapid substitution testing across different hardware targets without full retraining cycles.

- **Black-Box Multi-Objective Optimization Scales to Large Networks**: The training-free search formulation converts a combinatorial explosion (7^8 ≈ 5.7M combinations) into tractable Bayesian optimization using pre-trained surrogates. By replacing the evaluation of each configuration with a simple forward pass rather than requiring retraining, the method scales efficiently even for large 36-block architectures. The Pareto front extraction and knee-point selection provides interpretable trade-off choices between accuracy and latency.

- **Generalization Beyond Training Dataset is Strong**: The optimized model trained on GoPro motion deblurring generalizes effectively to unseen motion deblurring datasets (RealBlur-J/R, HIDE) with competitive PSNR scores (26.51 on RealBlur-J vs. NAFNet baseline 26.48, SSIM 0.824 vs. 0.823), and can be fine-tuned for defocus deblurring on DPDD with comparable performance. This demonstrates that hardware-aware optimization doesn't sacrifice generalization capability, contrary to the risk that aggressive optimization might overfit to training conditions.

- **Hardware-Aware Search Outputs Pareto-Optimal Architectures**: Rather than forcing a single architecture choice, Bayesian optimization reveals multiple Pareto-optimal configurations trading off accuracy and latency. The selection of least-latency architecture from the Pareto set (147ms) achieves 1.25× speedup compared to baseline (177ms) while maintaining reasonable accuracy (PSNR 33.76 vs. 35.80 baseline). This insight suggests that the optimal deployment choice depends on application requirements—some scenarios may prefer higher accuracy with 1.15× speedup vs. maximum latency reduction.

## Key Data & Results

| Benchmark | Method | PSNR | SSIM | GMACs | Latency (ms) | Notes |
|-----------|--------|------|------|-------|-------------|-------|
| **GoPro** | SRN | 30.26 | 0.934 | - | - | Baseline |
| | Restormer | 32.92 | 0.961 | 140 | - | SOTA transformer |
| | NAFNet (baseline) | 35.80 | 0.980 | - | 177 | Base model |
| | **Ours (optimized)** | **33.76** | **0.942** | **68** | **147** | **55% GMACs↓, 1.25× speedup** |
| **RealBlur-J** | NAFNet | 26.48 | 0.823 | - | - | Unseen test set |
| | **Ours** | **26.51** | **0.824** | - | - | Maintained performance |
| **DPDD** (Defocus) | Restormer | 28.87 | 0.882 | 141 | - | SOTA |
| | KBNet | 28.89 | 0.883 | 108 | - | Efficient baseline |
| | NAFNet | 24.44 | 0.766 | 65 | - | Base (different domain) |
| | **Ours (fine-tuned)** | **26.87** | **0.818** | **63** | - | **Competitive with 1.5% GMACs↓** |

- **Motion Deblurring Results**: The optimized model achieves a critical 1.25× on-device latency improvement (177ms → 147ms) on Samsung Galaxy S24 Ultra NPU while reducing GMACs by 55% compared to Restormer transformer baseline (140→68 GMACs). PSNR drops from 35.80 (baseline NAFNet) to 33.76 (optimized), representing a 2.04 PSNR trade-off for 1.25× speedup—reasonable for edge deployment where latency is paramount.

- **Generalization Across Motion Blur Datasets**: On unseen RealBlur-J test set, the optimized model maintains near-baseline performance (PSNR 26.51 vs. NAFNet 26.48, SSIM 0.824 vs. 0.823), despite being trained solely on GoPro. This demonstrates that hardware optimization through block substitution doesn't catastrophically reduce generalization. Performance on RealBlur-R remains identical (PSNR 33.82), confirming robustness across motion blur variations.

- **Defocus Deblurring Adaptation**: When fine-tuned on DPDD defocus deblurring dataset, the optimized architecture achieves 26.87 PSNR (indoor) with only 63 GMACs, competitive with specialized methods like KBNet (28.89 PSNR, 108 GMACs). This represents ~42% GMACs reduction compared to KBNet while losing only 2 PSNR, demonstrating that hardware-aware adaptation generalizes to fundamentally different blur types (motion vs. defocus).

- **Saliency Score Distribution**: Table 1 reveals heterogeneous block importance—4th encoder (0.0644 gradient norm) is 6× more salient than early encoders (0.0118), justifying its exclusion from substitution. Middle decoder blocks show lower saliency than later encoders, indicating that features learned in deeper layers are more critical for restoration than intermediate features, aligning with intuition about deblurring requiring long-range context.

## Strengths

- **Addresses a Practically Important Problem**: Image deblurring is a critical mobile ISP stage, and the paper tackles the genuine gap between algorithmic efficiency (FLOPs) and actual hardware deployment (latency). This hardware-reality disconnect is highly relevant for practitioners deploying to commercial smartphones with opaque NPU accelerators, making the work immediately applicable beyond academic interest.

- **Comprehensive Experimental Coverage**: The paper validates across multiple domains (motion blur: GoPro/RealBlur-J/R/HIDE; defocus blur: DPDD) and includes proper comparisons to established baselines (Restormer, KBNet, MPRNet, NAFNet). The ability to maintain competitive accuracy while achieving 55% GMACs reduction and 1.25× latency speedup demonstrates significant practical impact with measurable improvements over prior art.

- **Novel Multi-Stage Framework Combining Multiple Techniques**: The integration of sensitivity analysis, surrogate distillation, and Bayesian optimization is well-motivated. Each component addresses a specific challenge: saliency identifies safe targets for replacement, distillation trains efficient alternatives, and multi-objective search handles combinatorial complexity. The decoupling of accuracy optimization from inference optimization is conceptually clean and practically effective.

- **Training-Free Search Scalability**: Converting a potentially combinatorial search (7^8) into tractable Bayesian optimization through pre-trained surrogates is technically sound. The plug-and-play evaluation without retraining each configuration makes the approach scalable to large 36-block architectures. This efficiency improvement over neural architecture search methods is valuable for practitioners.

- **Clear Reproducibility Details**: The paper provides concrete details: baseline (17.112M parameters NAFNet), hardware target (Samsung Galaxy S24 Ultra), exact saliency metrics used, and quantitative results across metrics. The method's device-agnostic formulation (adapts to any hardware via profiling) enhances reproducibility and enables community adoption.

- **Honest Discussion of Accuracy Degradation**: The authors transparently report accuracy loss (PSNR 35.80→33.76 on GoPro, SSIM 0.980→0.942) alongside speedup claims, rather than overstating results. This honest trade-off presentation helps practitioners understand deployment constraints and builds credibility.

## Weaknesses

- **Limited Novelty in Individual Components**: While the framework cleverly combines existing techniques (saliency-based pruning, knowledge distillation, Bayesian optimization), none of these components is novel independently. The saliency metrics (SNIP, GraSP, Fisher) are from prior work; feature-level distillation is well-established; and Bayesian NAS is standard. The contribution is primarily in their combination rather than methodological innovation. For a top-tier venue, deeper technical novelty would strengthen the paper.

- **Insufficient Ablation Studies**: The paper lacks ablations quantifying the individual contribution of each stage (e.g., How much speedup from saliency guidance alone? How much from distillation vs. surrogate selection?). Table 3 shows final results but provides no comparison removing saliency analysis, distillation, or Bayesian search components. This makes it difficult to assess which stages are critical versus optional, limiting insights into the framework's design.

- **Accuracy-Latency Trade-off Not Fully Explored**: The paper selects only the "least latency" Pareto point (147ms) from the Pareto front for final analysis. Other points on the front likely offer better accuracy-latency trade-offs (e.g., higher PSNR with modest latency increase). The paper should present multiple Pareto solutions and discuss selection criteria for different use cases rather than optimizing purely for minimum latency.

- **Limited Hardware Scope and Generalization Claims**: Optimization and profiling are performed exclusively on Samsung Galaxy S24 Ultra NPU. While the paper claims a "black-box" framework, the hardware-specificity of profiling means results may not transfer to other NPU architectures (Qualcomm Snapdragon, Apple Neural Engine, etc.). Claims of framework generality would require validation on diverse hardware platforms.

- **Incomplete Comparison with Efficient Methods**: The paper compares mainly to full models (Restormer, KBNet, NAFNet) but lacks comparison with other hardware-efficient optimization techniques: quantization (INT8, pruning), knowledge distillation baselines, or neural architecture search methods applied to deblurring. Did the authors try combining hardware profiling with quantization? How does the method compare to simpler baselines like magnitude pruning? These omissions weaken claims about the framework's superiority.

- **Defocus Deblurring Results Concern**: In Table 5, the "Ours" (untuned) model achieves only 19.93 PSNR on DPDD indoor scenes—far below baselines (DPDNet 26.54, IFAN 28.11). While fine-tuning recovers to 26.87 PSNR, the untuned performance suggests significant domain mismatch when adapting from motion to defocus blur. The paper doesn't explain why or provide guidance on when hardware-optimized models can be directly transferred versus requiring task-specific fine-tuning.

## Research Directions

- **Hardware-Agnostic Meta-Learning for Block Substitution**: Develop a meta-learning framework that learns to predict optimal block substitutions for unseen hardware architectures without device-specific profiling. Train a model on (hardware specifications → optimal blocks) pairs across diverse NPU/GPU targets, enabling one-shot adaptation to new devices. This would generalize the framework's hardware-specificity limitation and enable broader deployment. A NeurIPS paper could formulate this as few-shot architecture adaptation using device capability fingerprints.

- **Multi-Task Hardware-Aware Optimization**: Extend the framework to jointly optimize for multiple tasks (motion + defocus deblurring) in a single model by using multi-task learning objectives in the Bayesian search. Rather than separate models per task, develop architectures that balance performance across tasks while minimizing latency. This addresses real mobile ISP pipelines that handle multiple blur types with a single inference model, improving practical applicability.

- **Theoretical Analysis of Saliency-Based Block Substitution**: Provide formal analysis of why saliency metrics successfully identify replaceable blocks—develop bounds on accuracy degradation from substituting low-saliency blocks in terms of saliency score magnitude. Combine with generalization theory to predict which tasks benefit most from hardware optimization. A strong theory paper could provide principled guidelines for practitioners on safe substitution thresholds.

- **Quantization-Aware Block-Level Search**: Combine hardware-aware block substitution with post-training quantization (INT8, mixed-precision) at block granularity. Use Bayesian search to jointly optimize block choices and quantization levels, exploiting that different blocks may have different quantization sensitivity. This could unlock additional 2-4× speedup on mobile accelerators with quantization support, complementing the current approach.

- **Adversarial Robustness of Hardware-Optimized Models**: Investigate whether aggressive hardware optimization through block substitution degrades adversarial robustness. Evaluate optimized deblurring models against adversarial blur perturbations and develop robust hardware-aware search variants. This addresses deployment concerns in safety-critical applications (autonomous driving with deblurred images) where both efficiency and robustness matter.

- **Dynamic Block Switching for Variable Latency Constraints**: Develop inference-time strategies that select block alternatives dynamically based on runtime latency budgets (e.g., switch to faster blocks if system is under thermal constraint). Train models with multiple block options simultaneously and use a lightweight runtime predictor to select alternatives on-device, enabling adaptive deployment without multiple pre-optimized variants.

- **Hardware-Aware Optimization for Other Restoration Tasks**: Generalize the framework to image super-resolution, denoising, and deraining—verify that sensitivity-guided substitution + distillation + Bayesian search works across the restoration pipeline. Each task may reveal unique hardware bottlenecks or require task-specific surrogate designs. A comprehensive study could establish hardware-aware optimization as a general principle for mobile ISP, opening a new subfield.

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **硬體感知自適應框架**：提出一個黑盒優化方法，通過直接設備分析來重組預訓練的去模糊模型（特別是NAFNet），專門針對邊界設備上的移動NPU部署。與傳統效率指標（FLOPs、參數數量）不同，此框架使用實際硬體反饋來指導架構決策，解決演算法效率和實際設備部署效能之間的關鍵差距。在三星Galaxy S24 Ultra上部署時，實現1.25倍的延遲改進。

- **敏感性引導的網路手術**：引入使用多個顯著性指標（梯度範數、SNIP、GraSP、Fisher資訊、SynFlow）識別和替換U-Net架構中延遲高但非關鍵區塊的技術。低顯著性區塊被替換為硬體友好的替代品，同時保持模型準確性，證明區塊級分析能有效指導網路重組。排除高顯著性的第4編碼器區塊（顯著性分數0.0644）同時替換其他區塊。

- **特徵級代理蒸餾**：提出訓練輕量級代理區塊的技術，這些區塊透過特徵級知識蒸餾來近似原始區塊的功能，同時更適應硬體，僅使用完整訓練集的20-25%進行訓練。此方法將準確性優化與推理優化解耦，實現多個代理的平行訓練，減少與端到端重訓練相比的計算負擔。

- **無訓練多目標貝葉斯搜尋**：將網路配置選擇表述為組合多目標優化問題（m個區塊×n個替代品），通過貝葉斯優化求解，無需為每個架構重新訓練。該方法使用預訓練的代理進行插即用評估，將7^8個（約570萬個）搜尋空間縮減為可行的準確度損失（PSNR損失）和目標硬體上分析延遲的評估。

- **全面的實驗驗證**：在多個去模糊任務上展示通用性（GoPro/RealBlur-J/R/HIDE上的運動去模糊、DPDD上的散焦去模糊），與最先進方法進行廣泛比較。與基於Transformer的SOTA（Restormer：140 GMACs vs. 本方法：68 GMACs）相比，實現55% GMACs減少，同時保持競爭力的準確性，重要的是在商用智慧型手機上展示1.25倍延遲改進。

- **用於可重組態部署的實用框架**：通過將準確性與推理優化解耦，為跨不同設備和任務的網路自適應建立可重複的方法論，實現硬體變更或新任務出現時的快速重新部署。這為從業者提供了關於模組級瓶頸和延遲權衡的可操作洞見，無需從頭設計新架構。

## 核心洞見

- **FLOPs-延遲不對齊至關重要**：該論文揭示傳統效率指標如FLOPs和參數數量無法與實際設備延遲相關聯。NAFNet在理論上被認為「高效」，但在專用硬體加速器上仍遭受延遲瓶頸（177毫秒基準），源於通道注意模組。此洞見根本上促進了從代理指標向直接硬體分析的轉變，證明即使是36區塊架構在無需完全重新設計的情況下也能從硬體感知重組中獲益。

- **顯著性估計指導安全的手術**：使用多個顯著性指標可識別某些區塊（例如NAFNet中具有28個NAF區塊的第4編碼器）儘管計算成本高卻對準確性至關重要，而其他區塊可被替換而準確性損失最小。作者排除了高顯著性區塊的替換，證明準確性保留取決於保留網路敏感性而非原始參數計數。此選擇性方法在實現1.25倍加速的同時保留原始模型的魯棒性。

- **特徵級蒸餾實現高效自適應**：通過特徵級蒸餾（而非端到端損失）僅使用完整資料集的20-25%訓練代理區塊，減少訓練開銷同時實現平行代理優化。此洞見將準確性復原與原始網路的訓練動態解耦，使框架模組化，實現跨不同硬體目標的快速替換測試，無需完整重訓練週期。

- **黑盒多目標優化擴展至大型網路**：將潛在的組合爆炸（7^8≈570萬組合）透過預訓練代理轉換為可行的貝葉斯優化。通過將每個配置的評估替換為簡單前向傳遞而非需要重訓練，該方法即使對於大型36區塊架構也能有效擴展。Pareto前沿提取和膝點選擇在準確性和延遲之間提供可解釋的權衡選擇。

- **超越訓練資料集的泛化能力強**：在GoPro運動去模糊上訓練的優化模型有效泛化至未見運動去模糊資料集（RealBlur-J/R、HIDE），具有競爭力的PSNR分數（RealBlur-J上26.51 vs. NAFNet基準26.48，SSIM 0.824 vs. 0.823），並可在DPDD上微調進行散焦去模糊，具有相當效能。此證明硬體感知優化不會犧牲泛化能力，與積極優化可能過度擬合訓練條件的風險相反。

- **硬體感知搜尋輸出Pareto最優架構**：貝葉斯優化揭示多個Pareto最優配置，在準確性和延遲之間權衡，而非強制單一架構選擇。從Pareto集選擇最小延遲架構（147毫秒）與基準（177毫秒）相比實現1.25倍加速，同時保持合理準確性（PSNR 33.76 vs. 35.80基準）。此洞見表明最優部署選擇取決於應用需求——某些場景可能偏好更高準確性且1.15倍加速vs. 最大延遲減少。

## 關鍵數據與結果

| 基準 | 方法 | PSNR | SSIM | GMACs | 延遲(毫秒) | 備註 |
|------|------|------|------|-------|----------|------|
| **GoPro** | SRN | 30.26 | 0.934 | - | - | 基準 |
| | Restormer | 32.92 | 0.961 | 140 | - | SOTA Transformer |
| | NAFNet (基準) | 35.80 | 0.980 | - | 177 | 基礎模型 |
| | **本方法(優化)** | **33.76** | **0.942** | **68** | **147** | **55% GMACs↓, 1.25× 加速** |
| **RealBlur-J** | NAFNet | 26.48 | 0.823 | - | - | 未見測試集 |
| | **本方法** | **26.51** | **0.824** | - | - | 維持效能 |
| **DPDD** (散焦) | Restormer | 28.87 | 0.882 | 141 | - | SOTA |
| | KBNet | 28.89 | 0.883 | 108 | - | 高效基準 |
| | NAFNet | 24.44 | 0.766 | 65 | - | 基礎(不同領域) |
| | **本方法(微調)** | **26.87** | **0.818** | **63** | - | **競爭力，1.5% GMACs↓** |

- **運動去模糊結果**：優化模型在三星Galaxy S24 Ultra NPU上實現關鍵的1.25倍設備延遲改進（177毫秒→147毫秒），同時將GMACs相比Restormer Transformer基準減少55%（140→68 GMACs）。PSNR從35.80（基準NAFNet）下降到33.76（優化），代表2.04 PSNR權衡以換取1.25倍加速——對於延遲至關重要的邊界部署而言是合理的。

- **跨運動模糊資料集泛化**：在未見RealBlur-J測試集上，優化模型保持近基準效能（PSNR 26.51 vs. NAFNet 26.48，SSIM 0.824 vs. 0.823），儘管僅在GoPro上訓練。此證明透過區塊替換的硬體優化不會導致泛化能力的災難性降低。RealBlur-R上的效能保持相同（PSNR 33.82），確認跨運動模糊變體的魯棒性。

- **散焦去模糊自適應**：當在DPDD散焦去模糊資料集上微調時，優化架構以僅63 GMACs實現26.87 PSNR（室內），與KBNet等專用方法競爭（28.89 PSNR，108 GMACs）。這代表相比KBNet約42% GMACs減少，同時僅損失2 PSNR，證明硬體感知自適應泛化至根本不同的模糊型別（運動vs.散焦）。

- **顯著性分數分佈**：表1揭示異質區塊重要性——第4編碼器（0.0644梯度範數）比早期編碼器（0.0118）重要6倍，證明其排除替換的合理性。中間解碼器區塊顯示低於後期編碼器的顯著性，表明深層學習的特徵比中間特徵對復原更關鍵，符合去模糊需要長程上下文的直覺。

## 優勢

- **解決實踐重要問題**：影像去模糊是關鍵的行動ISP階段，該論文解決演算法效率（FLOPs）與實際硬體部署（延遲）之間的真實差距。此硬體現實不對齊對於在不透明NPU加速器上部署至商用智慧型手機的從業者而言高度相關，使工作在學術興趣之外即時可用。

- **全面的實驗涵蓋**：該論文驗證跨多個領域（運動模糊：GoPro/RealBlur-J/R/HIDE；散焦模糊：DPDD）並包括與既定基準的適當比較（Restormer、KBNet、MPRNet、NAFNet）。能在實現55% GMACs減少和1.25倍延遲加速的同時保持競爭力準確性，證明具有超越現有技術的顯著實踐影響。

- **新穎的多階段框架整合多項技術**：敏感性分析、代理蒸餾和貝葉斯優化的整合動機充分。每個元件解決特定挑戰：顯著性識別替換安全目標，蒸餾訓練高效替代品，多目標搜尋處理組合複雜性。準確性優化與推理優化的解耦概念清晰且實踐有效。

- **無訓練搜尋可擴展性**：通過預訓練代理將潛在組合爆炸（7^8）轉換為可行貝葉斯優化在技術上合理。無需為每個配置重訓練就評估每個配置的方式使方法即使對大型36區塊架構也能有效擴展。此效率改進相比神經架構搜尋方法對從業者而言有價值。

- **清晰的可複現性細節**：論文提供具體細節：基準（17.112M參數NAFNet）、硬體目標（三星Galaxy S24 Ultra）、使用的精確顯著性指標及跨指標的定量結果。方法的設備無關公式（透過分析適應任何硬體）增強可複現性並實現社群採納。

- **誠實討論準確性退化**：作者透明地報告準確性損失（GoPro上PSNR 35.80→33.76，SSIM 0.980→0.942）以及加速聲明，而非誇大結果。此誠實權衡呈現幫助從業者理解部署限制並建立信譽。

## 劣勢

- **個別元件的新穎性有限**：儘管框架巧妙結合現有技術（基於顯著性的修剪、知識蒸餾、貝葉斯優化），這些元件中沒有一個獨立而新穎。顯著性指標（SNIP、GraSP、Fisher）來自先前工作；特徵級蒸餾是既定方法；貝葉斯NAS是標準。貢獻主要在於它們的組合而非方法創新。對於頂級場所，更深層的技術新穎性將強化論文。

- **消融研究不足**：論文缺乏量化每個階段個別貢獻的消融（例如，顯著性指導單獨實現多少加速？蒸餾vs.代理選擇多少？）。表3展示最終結果但無法比較移除顯著性分析、蒸餾或貝葉斯搜尋元件。此使難以評估哪些階段至關重要versus可選，限制對框架設計的洞見。

- **準確性-延遲權衡未完全探索**：論文僅從Pareto前沿選擇「最小延遲」點（147毫秒）用於最終分析。前沿上的其他點可能提供更好的準確性-延遲權衡（例如更高PSNR及適度延遲增加）。論文應呈現多個Pareto解決方案並討論不同用例的選擇標準，而非純粹優化最小延遲。

- **硬體範圍和泛化聲明有限**：優化和分析僅在三星Galaxy S24 Ultra NPU上進行。儘管論文聲稱「黑盒」框架，分析的硬體特異性意味著結果可能無法轉移至其他NPU架構（高通Snapdragon、Apple Neural Engine等）。框架泛化的聲明需要在多元硬體平台上驗證。

- **與高效方法的比較不完整**：論文主要與完整模型（Restormer、KBNet、NAFNet）比較，但缺乏與其他硬體效率優化技術的比較：量化（INT8、修剪）、知識蒸餾基準或應用於去模糊的神經架構搜尋方法。作者嘗試結合硬體分析與量化嗎？該方法與更簡單基準如大小修剪相比如何？這些遺漏削弱框架優越性聲明。

- **散焦去模糊結果值得關注**：在表5中，「本方法」（未調整）模型在DPDD室內場景上僅達19.93 PSNR——遠低於基準（DPDNet 26.54、IFAN 28.11）。儘管微調恢復至26.87 PSNR，未調整效能表明從運動至散焦模糊適應時存在顯著領域不匹配。論文未解釋原因或提供硬體優化模型何時可直接轉移vs.需要任務特定微調的指導。

## 研究方向

- **硬體無關元學習用於區塊替換**：開發元學習框架，學習預測未見硬體架構的最優區塊替換，無需設備特定分析。在多元NPU/GPU目標上訓練（硬體規格→最優區塊）對，實現對新設備的一次性自適應。此將泛化框架的硬體特異性限制並實現更廣泛部署。NeurIPS論文可使用設備能力指紋將此表述為少次架構自適應。

- **多任務硬體感知優化**：擴展框架以在單一模型中聯合優化多個任務（運動+散焦去模糊），使用多任務學習目標在貝葉斯搜尋中。而非每個任務的分離模型，開發在任務間平衡效能同時最小化延遲的架構。此解決實際行動ISP管線用單一推理模型處理多個模糊型別，改進實踐可用性。

- **顯著性基區塊替換的理論分析**：提供顯著性指標為何成功識別可替換區塊的形式分析——開發關於在顯著性分數大小方面替換低顯著性區塊的準確性退化界。結合泛化理論預測哪些任務最受益於硬體優化。強理論論文可為從業者提供安全替換閾值的原則指南。

- **量化感知區塊級搜尋**：結合硬體感知區塊替換與訓練後量化（INT8、混合精度）在區塊粒度。使用貝葉斯搜尋聯合優化區塊選擇和量化水準，利用不同區塊可能具有不同量化敏感性。此可在支援量化的行動加速器上解鎖額外2-4倍加速，補充當前方法。

- **硬體優化模型的對抗魯棒性**：調查透過區塊替換的積極硬體優化是否降低對抗魯棒性。評估優化去模糊模型對對抗模糊擾動並開發魯棒硬體感知搜尋變體。此解決安全關鍵應用（具有去模糊影像的自動駕駛）中的部署關注，其中效率和魯棒性均重要。

- **動態區塊切換用於可變延遲限制**：開發推理時策略，根據執行時延遲預算動態選擇區塊替代品（例如，如果系統處於熱限制下則切換至更快區塊）。同時訓練多個區塊選項的模型並使用輕量級執行時預測器在設備上選擇替代品，實現自適應部署無需多個預優化變體。

- **硬體感知優化用於其他復原任務**：將框架泛化至影像超解析度、去噪和除雨——驗證顯著性引導替換+蒸餾+貝葉斯搜尋在復原管線上作用。每個任務可能揭示獨特硬體瓶頸或需要任務特定代理設計。全面研究可在行動ISP中建立硬體感知優化為一般原則，開啟新子領域。

</div>


