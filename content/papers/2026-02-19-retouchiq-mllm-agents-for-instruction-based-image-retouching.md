---
title: "RetouchIQ: MLLM Agents for Instruction-Based Image Retouching with Generalist Reward"
date: 2026-02-19
authors:
  - "Qiucheng Wu"
  - "Jing Shi"
  - "Simon Jenni"
  - "Kushal Kafle"
  - "Tianyu Wang"
  - "Shiyu Chang"
  - "Handong Zhao"
source: "arXiv"
arxiv_url: "https://arxiv.org/abs/2602.17558"
pdf_url: "https://arxiv.org/pdf/2602.17558"
one_line_summary: "RetouchIQ introduces a generalist reward model for instruction-based image editing that dynamically generates case-specific evaluation metrics and uses policy-guided reward training to align reward supervision with policy-generated outputs, achieving superior semantic consistency and perceptual quality compared to MLLM agents and diffusion models."
one_line_summary_zh: "RetouchIQ提出通用獎勵模型用於基於指令的圖像編輯，動態生成案例特定的評估指標，並使用策略引導的獎勵訓練將獎勵監督與策略生成的輸出對齐，在語義一致性和感知質量上超越MLLM代理和擴散模型。"
date_added: 2026-02-21
topics: ["General"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **Generalist Reward Model for Subjective Image Editing**: The paper introduces a novel **generalist reward model (GRM)** that moves beyond conventional pixel-level, rule-based rewards to evaluate image retouching quality. Rather than comparing against a fixed reference image using handcrafted metrics like L1/L2 distance, the GRM is an RL-fine-tuned MLLM that dynamically generates case-specific evaluation metrics and assigns scalar rewards through multimodal reasoning. This directly addresses the fundamental challenge that multiple aesthetically valid edits can satisfy the same user instruction, making single reference-based metrics unreliable for subjective creative tasks.

- **Policy-Guided Reward Training (PGRT) Paradigm**: The paper identifies and solves a critical distribution shift problem in reward model training. Initial perturbation-based weak edits (used to construct strong/weak pairs for reward training) differ systematically from actual weak edits produced by the policy model—they tend to involve simple single-parameter adjustments rather than complex combined edits. The proposed **PGRT** replaces perturbed synthetic weak edits with actual policy-generated ones during RL, using an alternating training scheme to align the reward model with the policy distribution. Figure 5 demonstrates that this achieves the highest accuracy on actual policy-generated data (reaching ~0.75 on the red line) compared to training only on perturbed data.

- **RetouchIQ Framework with Professional Tool Integration**: The paper presents a complete **instruction-based executable image editing agent** that interprets natural language user instructions and generates both interpretable reasoning traces and precise parameter adjustments for professional editing software (Adobe Lightroom). The two-stage training approach (SFT + RL) enables the model to first learn gold reasoning processes and parameter configurations from demonstrations, then refine them through reinforcement learning guided by the generalist reward model. This bridges high-level aesthetic goals with precise, executable parameter control.

- **Large-Scale Curated Dataset with Quality Filtering**: The work contributes a substantial **190K instruction-reasoning pair dataset** collected from real user editing trajectories, supplemented with automatic annotation of user intentions and reasoning processes using an MLLM annotator. The data preparation pipeline applies systematic filtering to ensure quality and consistency. Additionally, the paper introduces **RetouchEval**, the first dedicated benchmark for instruction-based image editing with 300 curated pairs spanning three categories: quality enhancement, style transformation, and local retouching, establishing a new evaluation standard for this task.

- **Comprehensive Evaluation Demonstrating Superior Performance**: Experiments comprehensively demonstrate RetouchIQ's superiority across multiple evaluation protocols. On RetouchEval, RetouchIQ-GRM achieves the best or near-best results on semantic consistency (SC) and perceptual quality (PQ) metrics across all three task categories (quality improving: SC=7.57, PQ=7.48; style changing: SC=7.29, PQ=7.34; local retouching: SC=6.39, PQ=6.92). On the MIT-Adobe5K benchmark, it achieves PSNR=23.14, LPIPS=0.16, and SSIM=0.80, surpassing strong baselines including JarvisArt and general-purpose MLLMs like GPT-5 and Gemini-2.5. Qualitative results in Figure 6 show consistent semantic alignment and professional-quality outputs.

## Core Insights

- **Subjectivity Requires Adaptive Evaluation**: The paper provides compelling intuition that subjective image editing cannot be effectively evaluated by fixed, rule-based metrics tied to a single ground truth reference. Figure 3 elegantly illustrates how a single instruction like "make it feel like a fall scene with autumn vibes" can be satisfied by multiple distinct edits emphasizing different aesthetic dimensions (tone, warmth, color balance). The generalist reward model's ability to generate instruction-specific metrics (e.g., "autumn color vibe" vs "white balance") on a case-by-case basis fundamentally aligns reward design with the inherent subjectivity of creative tasks—an insight that extends beyond image editing to other subjective domains.

- **Distribution Alignment Matters More Than Perturbation Strategy**: A key finding is that the strategy for constructing weak edits for reward training substantially impacts downstream performance. Figure 5 shows that a reward model trained only on perturbed weak edits achieves ~0.65-0.68 accuracy when evaluating actual policy-generated edits, but when trained with PGRT (using policy-generated weak edits), accuracy jumps to ~0.75. Correspondingly, the policy model's overall score improves from 6.89 to 7.51 on RetouchEval. This reveals that distribution alignment between the reward model's training data and actual deployment distribution is critical—a lesson applicable to broader RL-for-agents systems where naive synthetic negative examples may not reflect real failure modes.

- **Reasoning Transparency Improves Both Performance and Explainability**: The two-output design of both the policy model (reasoning + parameters) and reward model (metrics + scores) enables the framework to be inherently interpretable. During SFT, the model learns to explicitly reason about aesthetic goals before selecting parameters, which likely improves parameter coherence and instruction alignment. During RL, the reward model's generated metrics provide interpretable feedback, allowing the policy to learn not just "this edit is better" but "this edit better achieves [specific aesthetic criterion]." This design choice provides both a performance benefit (reflected in superior SC and PQ scores) and transparency that diffusion-based black-box methods cannot offer.

- **Professional Tool Integration Enables Precise Control**: Unlike diffusion-based methods (e.g., Flux-Pro in Table 1) that often unintentionally alter image content and structure, RetouchIQ's reliance on professional editing tool parameters ensures pixel-perfect semantic preservation while allowing aesthetic control. Qualitative results consistently show RetouchIQ preserving object identity and environmental structure while achieving the intended style transformations, whereas Flux-Pro exhibits noticeable distortions. This demonstrates that agent-based tool use, despite being more constrained than generative models, provides superior semantic consistency (RetouchIQ SC: 7.57 vs Flux-Pro SC: 6.12 on quality improving tasks).

- **Scaling Benefits from RL Supervision**: The improvement from RetouchIQ-SFT (MLLM fine-tuned only on demonstrations, L1=35.03, SC=6.71) to RetouchIQ-GRM (with RL and generalist reward, L1=31.41, SC=7.57) on the quality improving task demonstrates that RL with high-quality reward signals systematically improves both pixel-level fidelity (L1 distance drops by 10.4%) and semantic alignment (SC increases by 12.8%). This validates that even with large demonstration datasets (190K pairs), RL-based refinement with well-designed reward models provides substantial gains, suggesting the framework would benefit from further scaling in both data and compute.

## Key Data & Results

| Benchmark | Task | Metric | Flux-Pro | GPT-5 | Gemini-2.5 | MonetGPT | JarvisArt | RetouchIQ-SFT | RetouchIQ-GRM |
|-----------|------|--------|----------|-------|-----------|----------|-----------|---------------|---------------|
| RetouchEval | Quality Improving | L1 ↓ | 64.97 | 36.21 | 39.58 | 32.05 | 32.17 | 35.03 | **31.41** |
| RetouchEval | Quality Improving | SC ↑ | 6.12 | 6.74 | 6.56 | 6.52 | 7.22 | 6.71 | **7.57** |
| RetouchEval | Quality Improving | PQ ↑ | 6.09 | 6.51 | 6.72 | 7.05 | 6.59 | 6.67 | **7.48** |
| RetouchEval | Style Changing | L1 ↓ | 68.20 | 39.17 | 41.54 | 41.03 | 33.90 | 38.80 | **34.08** |
| RetouchEval | Style Changing | SC ↑ | 6.48 | 6.89 | 5.82 | 5.87 | 7.39 | 7.04 | **7.29** |
| RetouchEval | Local Retouching | L1 ↓ | 67.93 | 37.20 | 36.11 | 30.08 | 38.19 | 26.41 | **27.03** |
| RetouchEval | Local Retouching | SC ↑ | 6.19 | 6.07 | 6.01 | 6.27 | 6.45 | 5.98 | **6.39** |
| MIT-Adobe5K | N/A | PSNR ↑ | 20.82 | 23.10 | 21.03 | 22.37 | - | - | **23.14** |
| MIT-Adobe5K | N/A | LPIPS ↓ | 0.72 | 0.82 | 0.76 | 0.84 | - | - | **0.16** |
| MIT-Adobe5K | N/A | SSIM ↑ | 0.85 | 0.80 | 0.75 | 0.70 | - | - | **0.80** |

**Key Quantitative Findings:**

- **Comprehensive Superiority Across Metrics and Task Categories**: RetouchIQ-GRM achieves the best performance on the majority of metrics across all three RetouchEval task categories. Most notably, on the quality improving task, it achieves L1=31.41 (8.7% better than JarvisArt's 32.17), SC=7.57 (4.8% improvement), and PQ=7.48 (13.5% improvement over Flux-Pro's 6.59). On MIT-Adobe5K, the LPIPS score of 0.16 is substantially better than all MLLM baselines, indicating superior perceptual quality.

- **Ablation Validates RL and Generalist Reward Importance**: The progression from RetouchIQ-SFT (L1=35.03, SC=6.71 on quality improvement) to RetouchIQ-GRM (L1=31.41, SC=7.57) demonstrates a consistent 10-12% improvement, confirming that RL with the generalist reward model substantially enhances both pixel-level and semantic quality. Figure 5's ablation comparing off-the-shelf MLLM, perturbed-data-trained reward model, and PGRT-trained reward model shows that PGRT achieves the highest accuracy on actual policy-generated data (~0.75 vs ~0.65 for perturbed-only) and correspondingly the highest policy model score (7.51 vs 6.89).

- **Diffusion and General MLLMs Struggle with Specificity**: Flux-Pro exhibits poor performance across multiple metrics (L1=64.97 on quality improvement, LPIPS=0.72 on MIT-Adobe5K), reflecting qualitative failures where image structure and identity are distorted. General-purpose MLLMs (GPT-5, Gemini-2.5) show better results than Flux-Pro but significantly underperform RetouchIQ, suggesting they lack the specialized reasoning and tool integration necessary for precise, instruction-aligned editing. JarvisArt, the strongest MLLM agent baseline, still trails RetouchIQ notably on several metrics.

- **Distribution Shift Solution (PGRT) Yields Measurable Gains**: Figure 5 quantifies that PGRT improves reward model accuracy on policy-generated data from ~0.65-0.68 (perturbed-only baseline) to ~0.75, a substantial improvement. This translates to policy model improvements: with an off-the-shelf reward model, the policy achieves 6.89; with perturbed-data-trained reward, 7.36; with PGRT, 7.51. The analysis reveals that distribution alignment directly improves downstream policy performance, validating the core technical contribution.

- **Generalization to Standard Benchmarks**: Performance on MIT-Adobe5K (a public benchmark without explicit instructions) demonstrates that RetouchIQ generalizes beyond its curated instruction-based dataset, achieving competitive PSNR (23.14, vs JarvisArt's best results) and particularly strong LPIPS (0.16, substantially better than comparable methods). This suggests the learned aesthetic reasoning and parameter control transfer to general image enhancement tasks.

## Strengths

- **Well-Motivated and Novel Technical Contribution**: The paper addresses a genuine and underexplored problem—how to apply RL to subjective image editing where multiple valid solutions exist. The generalist reward model represents a genuinely novel approach that moves beyond the limitations of pixel-level metrics. The insight that reward training data distribution should match policy deployment distribution (PGRT) is technically sound and well-motivated, addressing a real failure mode in RL agent training. This contribution has clear applications beyond image editing to other subjective domains.

- **Comprehensive and Fair Experimental Evaluation**: The paper compares against three categories of baselines (general-purpose MLLMs, MLLM agents, diffusion models), covering different architectural paradigms. The inclusion of both quantitative metrics (L1/L2, SC, PQ, PSNR, LPIPS, SSIM) and qualitative comparisons (Figure 6) provides a thorough assessment. The creation of RetouchEval benchmark with three task categories (quality enhancement, style transformation, local retouching) ensures evaluation across diverse editing scenarios. Evaluation on both curated (RetouchEval) and public (MIT-Adobe5K) benchmarks strengthens generalization claims.

- **Rigorous Ablation Studies Demonstrating Component Importance**: Figure 5's ablation systematically isolates the contribution of different reward model configurations (off-the-shelf, perturbed-data trained, PGRT-trained), clearly showing how PGRT improves both reward model accuracy and downstream policy performance. The comparison between RetouchIQ-SFT and RetouchIQ-GRM isolates the RL contribution. A variant RetouchIQ-Rule (presumably with rule-based rewards) is included, though its results are not detailed in the main tables. These ablations provide clear evidence that design choices matter.

- **High-Quality Dataset and Reproducibility**: The curation of 190K instruction-reasoning pairs from real user editing trajectories (not synthetic presets) and 10K+ samples for reward model training represents substantial effort. The paper describes the training pipeline, model architectures (Qwen2.5-VL-7B, GLM-4.5V), and hyperparameters. Evaluation metrics and implementation details are clearly specified, supporting reproducibility. The release of RetouchEval benchmark would significantly benefit the community.

- **Clear Presentation and Interpretability**: The paper is well-written with effective visualizations. Figure 2 clearly illustrates the overall pipeline with data preparation, SFT, and RL stages. Figure 3 compellingly demonstrates the problem with verifiable rewards in subjective tasks. Figure 4 concisely shows the generalist reward model and PGRT mechanism. The choice to generate interpretable metrics and reasoning traces (rather than black-box rewards) aligns with emerging community values around explainability in AI systems.

- **Professional Tool Integration Enables Practical Deployment**: Unlike diffusion models that produce images from scratch, RetouchIQ's integration with professional editing software (Adobe Lightroom) ensures compatibility with existing professional workflows, making it practically deployable. The generation of executable parameters (e.g., {exposure=+0.9; contrast=-30}) is more transparent and controllable than generative approaches, valuable for professional photographers.

## Weaknesses

- **Limited Analysis of Reward Model's Generalization Beyond Training Distribution**: While PGRT addresses the distribution shift during RL, the paper doesn't thoroughly analyze whether the reward model's learned metrics generalize to entirely novel editing tasks or instructions outside its training distribution. For instance, how would the reward model perform on editing styles not well-represented in the 190K training pairs? The reliance on GLM-4.5V for metric generation may also introduce biases—does the reward model merely learn to mimic the MLLM's aesthetic preferences rather than capturing objective quality? Sensitivity analysis on this dimension is absent.

- **Incomplete Computational Cost Analysis**: The paper does not discuss training time, inference latency, or memory requirements compared to baselines. PGRT requires alternating optimization of policy and reward models, which likely increases total training time compared to standard RL. Inference-time cost is also unspecified—does generating reasoning + metrics + parameters add significant latency compared to Flux-Pro or simpler baselines? This information is essential for practitioners considering adoption.

- **Evaluation Metrics Partially Dependent on MLLM Annotators**: Semantic Consistency (SC) and Perceptual Quality (PQ) scores are produced by GLM-4.5V, the same MLLM family used in data annotation and reward training. This introduces potential circularity—the method may implicitly optimize for GLM-4.5V's aesthetic preferences, inflating scores on metrics evaluated by the same model family. Human evaluation on a subset of results would provide more trustworthy validation, but is absent. L1/L2 metrics on RetouchEval may also be problematic given that multiple valid edits exist.

- **Limited Scope of Baseline Comparisons**: The paper excludes direct comparison with some recent diffusion-based image editing methods and closed-source systems (acknowledging Google's NanaBanana and Adobe Firefly as popular but unavailable). While the inclusion of Flux-Pro-1.1 provides an open-source diffusion baseline, comparisons with other recent RL-based image editing approaches or diffusion variants (e.g., ControlNet-based editing) would strengthen the empirical assessment. The diffusion baseline comparison is somewhat limited to a single model family.

- **Insufficient Analysis of Failure Cases and Limitations**: The paper provides no discussion of scenarios where RetouchIQ fails or underperforms. Are there instruction types (e.g., complex local edits, extreme style transformations) where the method struggles? What is the error rate in parameter generation, and how does it vary across editing dimensions? The qualitative results in Figure 6 show successes but no failure examples, limiting readers' ability to understand applicability boundaries. Failure analysis would enhance the paper's credibility.

- **Data Filtering Process Lacks Transparency**: The paper mentions "systematic filtering procedures" to remove image pairs with unclear intentions or inconsistencies but provides minimal detail. How many pairs were filtered out? What were the filtering criteria? What inter-annotator agreement was achieved on the annotation task? Without clarity on data quality control, reproducibility of the dataset curation is compromised. Additionally, the filtering procedure itself may introduce biases—if it preferentially removes difficult cases, the resulting dataset may not be fully representative.

- **Limited Exploration of Alternatives to PGRT Solution**: While PGRT effectively addresses the distribution shift problem, the paper doesn't compare to alternative solutions such as: (1) using curriculum learning to gradually increase edit complexity during reward training; (2) importance weighting to adjust for distribution mismatch; (3) meta-learning approaches to adapt the reward model to the policy's distribution. A more comprehensive comparison of distribution alignment strategies would strengthen the contribution's positioning as a general RL principle.

## Research Directions

- **Cross-Domain Generalist Reward Models**: Extend the generalist reward model beyond image editing to other subjective domains such as video generation, 3D modeling, creative writing, or music composition. The core insight—that adaptive, instruction-specific metrics outperform fixed verifiable rewards in subjective tasks—likely applies broadly. A meta-framework that learns to generate task-appropriate metrics across diverse domains would be a significant contribution. Approach: Train a unified MLLM reward model on instruction-specific metric generation tasks across multiple creative domains, with cross-domain evaluation showing consistent improvements over domain-specific reward models.

- **Human-in-the-Loop Reward Model Refinement**: Current PGRT relies entirely on policy-model-generated weak edits for reward training, which may miss edge cases or adversarial examples that humans would recognize as truly bad. Develop an interactive framework where occasional human feedback (e.g., "this metric mismatch reveals an important edge case") is incorporated to refine the reward model. This could employ active learning to selectively request human annotation on high-uncertainty samples. Approach: Implement a bandit-based or uncertainty-sampling strategy that identifies cases where the reward model's confidence is low or policy-generated edits are atypical, prioritizing these for human review.

- **Multimodal Compositional Rewards for Complex Editing Goals**: Many professional editing tasks require balancing multiple, sometimes conflicting objectives (e.g., maximize saturation while preserving natural skin tones). Current scalar rewards may struggle with such trade-offs. Develop a compositional reward framework that decomposes complex objectives into sub-rewards and learns trade-off weights from user preferences. This could enable users to specify priority levels for different aesthetic dimensions. Approach: Use constraint satisfaction or multi-objective RL formulations where the reward model generates both individual metric scores and learned weights reflecting user preferences, with interactive preference elicitation to refine trade-offs.

- **Efficient Reward Model Distillation for Deployment**: PGRT requires alternating training of policy and reward models, increasing computational cost. Develop methods to distill the learned reward model into a more efficient form (e.g., a smaller MLLM, a neural network classifier, or a symbolic rule set) that retains performance while reducing inference latency. This would enable deployment in resource-constrained environments. Approach: Use knowledge distillation or neural-symbolic learning to extract interpretable reward functions from the trained MLLM-based reward model, with efficiency-accuracy trade-off analysis.

- **Real-time Collaborative Editing with Iterative Refinement**: Extend RetouchIQ from single-pass editing to interactive, iterative editing where users can see results and provide real-time feedback (e.g., "increase the warmth more, but preserve highlight details"). This requires developing a fast feedback mechanism and updating the editing trajectory online. Approach: Implement a streaming inference pipeline where partial results are shown incrementally, and user feedback (via text or interactive sliders) is incorporated mid-generation to refine parameters in real-time without full recomputation.

- **Cross-Tool Generalization and Abstraction**: Current RetouchIQ is tightly integrated with Adobe Lightroom's parameter space. Develop methods to abstract the parameter control layer such that the MLLM agent can generalize to different professional editing software (Capture One, DxO OpticsPro, Affinity Photo) or even programmatic image processing libraries (PIL, OpenCV). Approach: Learn tool-agnostic semantic editing intent, then translate to tool-specific parameters via learned mappings. Use domain adaptation techniques to reduce the effort required to port RetouchIQ to new editing tools.

- **Theoretical Analysis of Distribution Shift in RL Reward Training**: Formalize the distribution shift problem that PGRT addresses and provide theoretical guarantees about when and why PGRT converges better than naive perturbation-based reward training. Analyze the trade-off between reward model expressiveness (flexibility to represent diverse metrics) and optimization difficulty (convergence speed). Approach: Develop formal RL theory characterizing the optimality gap when reward model distribution diverges from policy distribution, proving that PGRT reduces this gap. Connect to recent work on off-policy RL and importance sampling.

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **通用獎勵模型應對主觀圖像編輯任務**：本文介紹了一個新穎的**通用獎勵模型（GRM）**，超越了傳統的像素級、基於規則的獎勵設計。與計算與固定參考圖像相似度的手工特徵化指標（如L1/L2距離）不同，該GRM是通過強化學習微調的MLLM，能夠動態生成針對具體情況的評估指標，並通過多模態推理提供標量獎勵。這直接解決了一個根本的挑戰：多個美學上有效的編輯可以滿足相同的用戶指令，使得單一參考圖像的像素級指標在主觀創意任務中不可靠。

- **策略引導的獎勵訓練（PGRT）範式**：本文識別並解決了獎勵模型訓練中的關鍵分佈偏移問題。初始的基於擾動的弱編輯（用於構造強/弱編輯對以進行獎勵訓練）與策略模型實際產生的弱編輯在分佈上存在系統性差異——它們傾向於涉及簡單的單參數調整，而非複雜的聯合編輯。提出的**PGRT**在強化學習階段用實際策略生成的弱編輯取代擾動生成的合成弱編輯，使用交替訓練方案使獎勵模型與策略分佈對齐。圖5表明這在實際策略生成的數據上達到最高精度（紅線達到約0.75），相比僅在擾動數據上訓練有顯著提升。

- **RetouchIQ框架與專業工具整合**：本文提出了一個完整的**基於指令的可執行圖像編輯智能體**，能夠解釋自然語言用戶指令並生成可解釋的推理軌跡和精確的參數調整以控制專業編輯軟體（Adobe Lightroom）。兩階段訓練方法（SFT + RL）使模型首先從演示中學習金標準推理過程和參數配置，然後通過由通用獎勵模型引導的強化學習進行精煉。這橋接了高層次的美學目標與精確、可執行的參數控制。

- **具有質量篩選的大規模策劃數據集**：本工作貢獻了一個包含**190K條指令-推理對的實質性數據集**，這些數據來自真實的用戶編輯軌跡，並輔以使用MLLM註釋者自動標註的用戶意圖和推理過程。數據準備管道應用系統的篩選以確保質量和一致性。此外，本文引入了**RetouchEval**，這是第一個針對基於指令的圖像編輯的專用基準，包含300個策劃的對，涵蓋三個類別：質量增強、風格轉換和局部修飾，建立了這一任務的新評估標準。

- **全面評估展示優越性能**：實驗全面展示了RetouchIQ在多個評估協議上的優越性。在RetouchEval上，RetouchIQ-GRM在所有三個任務類別上的語義一致性（SC）和感知質量（PQ）指標上達到最佳或接近最佳結果（質量改進：SC=7.57，PQ=7.48；風格轉換：SC=7.29，PQ=7.34；局部修飾：SC=6.39，PQ=6.92）。在MIT-Adobe5K基準上，它達到PSNR=23.14、LPIPS=0.16和SSIM=0.80，超過了包括JarvisArt和GPT-5、Gemini-2.5等通用MLLM的強基線。圖6中的質量結果顯示一致的語義對齐和專業級輸出。

## 核心洞見

- **主觀性需要自適應評估**：本文提供了令人信服的直覺，即主觀圖像編輯不能通過固定的、與單一地面真實參考相關的基於規則的指標有效地評估。圖3優雅地說明了單個指令（如「讓它感覺像秋天場景，具有秋日氛圍」）如何可以通過多個不同的編輯滿足，每個都強調不同的美學維度（色調、溫暖度、色彩平衡）。通用獎勵模型在逐個案例基礎上生成指令特定指標（如「秋色氛圍」vs「白平衡」）的能力從根本上使獎勵設計與創意任務的固有主觀性對齐——這一洞見超越圖像編輯，擴展到其他主觀領域。

- **分佈對齐比擾動策略更重要**：一個關鍵發現是用於獎勵訓練的弱編輯構造策略對下游性能有實質影響。圖5表明，僅在擾動弱編輯上訓練的獎勵模型在評估實際策略生成的編輯時達到約0.65-0.68的精度，但使用PGRT訓練時（使用策略生成的弱編輯），精度跳升到約0.75。相應地，策略模型在RetouchEval上的總體得分從6.89提高到7.51。這揭示了獎勵模型訓練數據與實際部署分佈之間的分佈對齐對於關鍵重要性——這一教訓適用於更廣泛的代理強化學習系統，其中天真的合成反例可能無法反映真實故障模式。

- **推理透明性同時改善性能和可解釋性**：策略模型（推理+參數）和獎勵模型（指標+得分）的雙輸出設計使框架本質上可解釋。在SFT期間，模型學習在選擇參數前明確推理美學目標，這可能改善參數的連貫性和指令對齐。在RL期間，獎勵模型生成的指標提供可解釋的反饋，允許策略不僅學習「這個編輯更好」，而是「這個編輯更好地實現了[具體美學標準]」。這一設計選擇既提供了性能優勢（反映在優越的SC和PQ分數中），也提供了擴散型黑箱方法無法提供的透明性。

- **專業工具整合實現精確控制**：與擴散模型（如表1中的Flux-Pro）相比，後者經常無意中改變圖像內容和結構，RetouchIQ對專業編輯工具參數的依賴確保了像素完美的語義保持，同時允許美學控制。質量結果一貫地顯示RetouchIQ保持對象身份和環境結構，同時實現預期的風格轉換，而Flux-Pro表現出明顯的扭曲。這表明儘管代理型工具使用比生成模型更受限制，但它提供了優越的語義一致性（RetouchIQ SC：7.57 vs Flux-Pro SC：6.12在質量改進任務上）。

- **RL監督的擴展收益**：從RetouchIQ-SFT（僅在演示上微調的MLLM，L1=35.03，SC=6.71）到RetouchIQ-GRM（具有RL和通用獎勵，L1=31.41，SC=7.57）在質量改進任務上的改進表明，具有高質量獎勵信號的RL系統改善了像素級保真度（L1距離下降10.4%）和語義對齐（SC增加12.8%）。這驗證了即使擁有大型演示數據集（190K對），具有精心設計獎勵模型的RL精煉也提供實質性收益，表明框架在數據和計算進一步擴展中將受益。

## 關鍵數據與結果

| 基準 | 任務 | 指標 | Flux-Pro | GPT-5 | Gemini-2.5 | MonetGPT | JarvisArt | RetouchIQ-SFT | RetouchIQ-GRM |
|------|------|------|----------|-------|-----------|----------|-----------|---------------|---------------|
| RetouchEval | 質量改進 | L1 ↓ | 64.97 | 36.21 | 39.58 | 32.05 | 32.17 | 35.03 | **31.41** |
| RetouchEval | 質量改進 | SC ↑ | 6.12 | 6.74 | 6.56 | 6.52 | 7.22 | 6.71 | **7.57** |
| RetouchEval | 質量改進 | PQ ↑ | 6.09 | 6.51 | 6.72 | 7.05 | 6.59 | 6.67 | **7.48** |
| RetouchEval | 風格轉換 | L1 ↓ | 68.20 | 39.17 | 41.54 | 41.03 | 33.90 | 38.80 | **34.08** |
| RetouchEval | 風格轉換 | SC ↑ | 6.48 | 6.89 | 5.82 | 5.87 | 7.39 | 7.04 | **7.29** |
| RetouchEval | 局部修飾 | L1 ↓ | 67.93 | 37.20 | 36.11 | 30.08 | 38.19 | 26.41 | **27.03** |
| RetouchEval | 局部修飾 | SC ↑ | 6.19 | 6.07 | 6.01 | 6.27 | 6.45 | 5.98 | **6.39** |
| MIT-Adobe5K | N/A | PSNR ↑ | 20.82 | 23.10 | 21.03 | 22.37 | - | - | **23.14** |
| MIT-Adobe5K | N/A | LPIPS ↓ | 0.72 | 0.82 | 0.76 | 0.84 | - | - | **0.16** |
| MIT-Adobe5K | N/A | SSIM ↑ | 0.85 | 0.80 | 0.75 | 0.70 | - | - | **0.80** |

**關鍵定量發現：**

- **跨指標和任務類別的全面優越性**：RetouchIQ-GRM在所有三個RetouchEval任務類別的大多數指標上達到最佳性能。最值得注意的是，在質量改進任務上，它達到L1=31.41（比JarvisArt的32.17好8.7%），SC=7.57（改進4.8%），PQ=7.48（比Flux-Pro的6.59改進13.5%）。在MIT-Adobe5K上，LPIPS分數0.16實質性好於所有MLLM基線，表明優越的感知質量。

- **消融驗證RL和通用獎勵的重要性**：從RetouchIQ-SFT（質量改進上L1=35.03，SC=6.71）到RetouchIQ-GRM（L1=31.41，SC=7.57）的進展表明持續10-12%的改進，確認了具有通用獎勵模型的RL實質性增強了像素級和語義質量。圖5的消融比較了現成的MLLM、擾動數據訓練的獎勵模型和PGRT訓練的獎勵模型，顯示PGRT在實際策略生成數據上達到最高精度（~0.75 vs ~0.65用於擾動）和相應最高策略模型得分（7.51 vs 6.89）。

- **擴散和通用MLLM在特異性方面困難**：Flux-Pro在多個指標上表現不佳（質量改進L1=64.97，MIT-Adobe5K上LPIPS=0.72），反映了圖像結構和身份扭曲的質量失敗。通用MLLM（GPT-5、Gemini-2.5）顯示比Flux-Pro更好的結果但明顯不如RetouchIQ，表明它們缺乏精確、指令對齐編輯所需的專門推理和工具整合。最強MLLM代理基線JarvisArt在幾個指標上仍明顯落後於RetouchIQ。

- **分佈偏移解決方案（PGRT）產生可測量收益**：圖5量化了PGRT將獎勵模型在策略生成數據上的精度從~0.65-0.68（擾動只有基線）改進到~0.75，實質性改進。這轉化為策略模型改進：使用現成獎勵模型，策略達到6.89；用擾動數據訓練的獎勵，7.36；用PGRT，7.51。分析表明分佈對齐直接改善下游策略性能，驗證了核心技術貢獻。

- **推廣到標準基準**：MIT-Adobe5K上的性能（沒有明確指令的公共基準）表明RetouchIQ推廣超過其策劃指令數據集，達到有競爭力的PSNR（23.14，vs JarvisArt最佳結果）和特別強LPIPS（0.16，實質性好於可比方法）。這表明學習的美學推理和參數控制轉移到一般圖像增強任務。

## 優勢

- **良好動機和新穎的技術貢獻**：本文解決了一個真實且未充分探索的問題——如何將RL應用於主觀圖像編輯，其中存在多個有效解決方案。通用獎勵模型代表了一個真正新穎的方法，超越像素級指標的限制。獎勵訓練數據分佈應匹配策略部署分佈（PGRT）的見解在技術上合理且動機充分，解決了RL代理訓練中的真實故障模式。這一貢獻在圖像編輯之外的其他主觀領域有明確應用。

- **全面和公平的實驗評估**：本文比較了三類基線（通用MLLM、MLLM代理、擴散模型），涵蓋不同的架構範式。納入定量指標（L1/L2、SC、PQ、PSNR、LPIPS、SSIM）和質量比較（圖6）提供了全面評估。創建具有三個任務類別（質量增強、風格轉換、局部修飾）的RetouchEval基準確保跨多樣編輯場景的評估。在策劃（RetouchEval）和公共（MIT-Adobe5K）基準上的評估加強了推廣聲明。

- **嚴格的消融研究展示組件重要性**：圖5的消融系統地隔離了不同獎勵模型配置（現成的、擾動數據訓練、PGRT訓練）的貢獻，清楚地顯示PGRT如何改善獎勵模型精度和下游策略性能。RetouchIQ-SFT和RetouchIQ-GRM之間的比較隔離了RL貢獻。包含了一個RetouchIQ-Rule變體（可能具有基於規則的獎勵），儘管其結果未在主表中詳細說明。這些消融提供清晰證據表明設計選擇重要。

- **高質量數據集和可重複性**：190K指令-推理對的策劃來自真實用戶編輯軌跡（非合成預設）以及10K+獎勵模型樣本代表實質性工作。本文描述了訓練管道、模型架構（Qwen2.5-VL-7B、GLM-4.5V）和超參數。評估指標和實現細節明確指定，支持可重複性。RetouchEval基準的發布將顯著利益社區。

- **清晰的呈現和可解釋性**：論文寫作良好，視覺化有效。圖2清楚地說明了包含數據準備、SFT和RL階段的整體管道。圖3令人信服地展示了主觀任務中可驗證獎勵的問題。圖4簡潔地顯示了通用獎勵模型和PGRT機制。選擇生成可解釋的指標和推理軌跡（而非黑箱獎勵）與關於AI系統可解釋性的新興社區價值對齐。

- **專業工具整合實現實際部署**：與從頭生成圖像的擴散模型不同，RetouchIQ與專業編輯軟體（Adobe Lightroom）的整合確保與現有專業工作流的兼容性，使其實際可部署。生成可執行參數（如{exposure=+0.9; contrast=-30}）比生成方法更透明和可控，對專業攝影師有價值。

## 劣勢

- **超過訓練分佈的獎勵模型泛化分析有限**：儘管PGRT在RL期間解決分佈偏移，本文未徹底分析獎勵模型學習的指標是否推廣到完全新穎的編輯任務或超出其訓練分佈的指令。例如，獎勵模型在190K訓練對中代表不充分的編輯風格上的表現如何？對GLM-4.5V進行指標生成的依賴也可能引入偏見——獎勵模型是否僅學習模仿MLLM的美學偏好而非捕捉客觀質量？此維度的敏感性分析缺失。

- **不完整的計算成本分析**：本文未討論與基線相比的訓練時間、推理延遲或記憶要求。PGRT需要策略和獎勵模型的交替優化，這可能比標準RL增加總訓練時間。推理時成本也未指定——生成推理+指標+參數是否相比Flux-Pro或更簡單基線增加顯著延遲？這些信息對於考慮採用的實踐者至關重要。

- **評估指標部分依賴於MLLM註釋者**：語義一致性（SC）和感知質量（PQ）分數由GLM-4.5V產生，與數據註釋和獎勵訓練中使用的相同MLLM系列。這引入潛在的循環性——方法可能隱含優化GLM-4.5V的美學偏好，在由相同模型系列評估的指標上誇大分數。在結果子集上的人類評估會提供更可信的驗證，但缺失。RetouchEval上的L1/L2指標在給定多個有效編輯存在的情況下也可能有問題。

- **基線比較範圍有限**：本文排除與某些最近擴散型圖像編輯方法和閉源系統的直接比較（承認Google的NanaBanana和Adobe Firefly作為流行但不可用）。儘管納入Flux-Pro-1.1提供開源擴散基線，與其他最近RL型圖像編輯方法或擴散變體（如基於ControlNet的編輯）的比較將加強實驗評估。擴散基線比較某種程度上限於單一模型系列。

- **故障案例和限制分析不充分**：本文未討論RetouchIQ故障或表現不佳的場景。是否存在指令類型（如複雜局部編輯、極端風格轉換）方法困難？參數生成中的錯誤率是多少，它如何跨編輯維度變化？圖6中的質量結果顯示成功但沒有故障示例，限制讀者理解適用性邊界的能力。故障分析將增強論文的可信度。

- **數據篩選過程缺乏透明度**：本文提到「系統篩選程序」以移除意圖不清晰或不一致的圖像對，但提供最少詳細。多少對被篩選出？篩選標準是什麼？在註釋任務上達到什麼內部標註者協議？未獲詳細的數據質量控制，數據集策劃的可重複性受損。此外，篩選程序本身可能引入偏見——如果它優先移除困難案例，結果數據集可能不完全代表。

- **對PGRT解決方案替代品的探索有限**：儘管PGRT有效解決分佈偏移問題，本文未比較替代解決方案如：（1）使用課程學習在獎勵訓練期間逐漸增加編輯複雜性；（2）重要性加權調整分佈不匹配；（3）元學習方法適應獎勵模型到策略分佈。更全面的分佈對齐策略比較將加強貢獻在RL一般原理中的定位。

## 研究方向

- **跨域通用獎勵模型**：將通用獎勵模型從圖像編輯擴展到其他主觀領域如視頻生成、3D建模、創意寫作或音樂組合。核心見識——自適應、指令特定指標在主觀任務中優於固定可驗證獎勵——可能廣泛適用。一個在多個創意領域內學習生成任務合適指標的元框架將是顯著貢獻。方法：在多個創意領域內的指令特定指標生成任務上訓練統一的MLLM獎勵模型，具有跨域評估顯示對域特定獎勵模型的一致改進。

- **人類迴圈中的獎勵模型精煉**：當前PGRT完全依賴策略模型生成的弱編輯進行獎勵訓練，這可能遺漏人類會識別為真正糟糕的邊界案例或對抗示例。開發交互框架，其中偶發的人類反饋（如「這個指標不匹配揭示重要邊界案例」）被納入精煉獎勵模型。這可能採用主動學習在高度不確定樣本上選擇性請求人類標註。方法：實現辨別獎勵模型信心低或策略生成編輯非典型案例的土匪或不確定性採樣策略，優先考慮這些進行人類審查。

- **複雜編輯目標的多模態組合獎勵**：許多專業編輯任務需要平衡多個、有時衝突的目標（如最大化飽和度同時保持自然膚色）。當前標量獎勵可能在此類權衡中困難。開發一個組合獎勵框架，將複雜目標分解為子獎勵並從用戶偏好學習權衡權重。這可能使用戶指定不同美學維度的優先級。方法：使用約束滿足或多目標RL公式，其中獎勵模型生成個別指標得分和反映用戶偏好的學習權重，具有交互偏好徵集精煉權衡。

- **高效獎勵模型蒸餾以進行部署**：PGRT需要策略和獎勵模型的交替訓練，增加計算成本。開發方法將學習的獎勵模型蒸餾為更高效的形式（如更小的MLLM、神經網路分類器或符號規則集）以保持性能同時降低推理延遲。這將在資源受限環境中實現部署。方法：使用知識蒸餾或神經符號學習從訓練MLLM型獎勵模型提取可解釋獎勵函數，具有效率-精度權衡分析。

- **與迭代精煉的實時協作編輯**：將RetouchIQ從單次編輯擴展為交互式迭代編輯，用戶可以看到結果並提供實時反饋（如「增加更多溫暖度，但保持高光細節」）。這需要開發快速反饋機制和線上更新編輯軌跡。方法：實現流推理管道，其中部分結果逐漸顯示，用戶反饋（通過文本或交互滑塊）在生成中途被納入以精煉參數，無需完全重新計算。

- **跨工具推廣和抽象**：當前RetouchIQ緊密整合Adobe Lightroom的參數空間。開發方法抽象參數控制層，使MLLM代理可推廣到不同專業編輯軟體（Capture One、DxO OpticsPro、Affinity Photo）甚至編程圖像處理庫（PIL、OpenCV）。方法：學習工具無關的語義編輯意圖，然後通過學習映射翻譯為工具特定參數。使用領域適應技術減少將RetouchIQ移植到新編輯工具所需的工作。

- **RL獎勵訓練中分佈偏移的理論分析**：形式化PGRT解決的分佈偏移問題並提供關於何時以及為何PGRT收斂優於天真擾動型獎勵訓練的理論保證。分析獎勵模型表達力（表示多樣指標的靈活性）和優化困難（收斂速度）之間的權衡。方法：開發形式RL理論，刻畫獎勵模型分佈與策略分佈偏離時的最優性差距，證明PGRT減少此差距。連接到關於策略外RL和重要性採樣的最近工作。

</div>


