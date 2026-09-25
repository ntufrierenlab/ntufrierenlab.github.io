---
title: "Generating the Past, Present and Future from a Motion-Blurred Image"
date: 2025-12-22
authors:
  - "SaiKiran Tedla"
  - "Kelly Zhu"
  - "Trevor Canham"
  - "Felix Taubner"
  - "Michael S. Brown"
  - "Kiriakos N. Kutulakos"
  - "David B. Lindell"
source: "arXiv"
arxiv_url: "https://arxiv.org/abs/2512.19817"
pdf_url: "https://arxiv.org/pdf/2512.19817"
one_line_summary: "The paper presents a method to recover past, present, and future video frames from motion-blurred images by leveraging large-scale pre-trained video diffusion models with novel exposure interval conditioning, demonstrating state-of-the-art results on benchmarks and enabling applications in tracking, 3D reconstruction, and historical image restoration."
one_line_summary_zh: "本文通過運用大規模預訓練視頻擴散模型和新穎的曝光間隔調節，從動作模糊圖像中恢復過去、現在和未來視頻幀，在基準上展示最先進結果，並啟用跟蹤、3D重建和歷史圖像修復中的應用。"
date_added: 2026-02-13
topics: ["Image Deblurring"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **Repurposing Pre-trained Video Diffusion Models for Motion Blur Analysis**: The paper introduces a novel technique that adapts a large-scale pre-trained video diffusion model (CogVideoX-2B with 2 billion parameters) to recover videos from motion-blurred images. Unlike prior work that relies on handcrafted priors or small training datasets (tens of thousands of examples), this approach leverages internet-scale video datasets and demonstrates superior generalization to complex, in-the-wild scenes including dancers, sports events, and deforming cloth.

- **Precise Exposure Time Control for Past/Present/Future Generation**: The authors develop a novel conditioning mechanism that enables precise control over each generated frame's exposure interval through sinusoidal positional encoding and temporal indexing. This architectural innovation allows the model to generate not just frames during the exposure window (the "present"), but also predict what occurred immediately before (past) and after (future) the moment of capture—a capability absent in prior video-from-blur methods.

- **Comprehensive Evaluation Framework with Bidirectional Patch-Based Metrics**: The paper introduces improved quantitative evaluation metrics (Equation 6) that address a fundamental challenge in motion blur analysis: motion ambiguities (e.g., time-reversed videos are equally valid). The bidirectional patch-based metric $M_p$ evaluates both forward and time-reversed patches independently, providing fine-grained spatial assessment of motion consistency rather than just frame-level metrics.

- **State-of-the-Art Results on Multiple Benchmarks**: The method substantially outperforms prior work on standard benchmarks: on GoPro dataset, it achieves PSNR of 30.01 dB vs. 26.54 dB for MotionETR and 25.23 dB for Jin et al.; on B-AIST++ dataset, PSNR improves from 26.69 dB to 27.37 dB. Most critically, FVD (distribution similarity) improves from 235.53 to 21.46 on GoPro, indicating photorealistic video generation that matches ground truth distribution.

- **Diverse Downstream Applications**: The generated videos enable multiple downstream computer vision tasks including 3D object tracking (using CoTracker), camera trajectory recovery (using MegaSaM), dynamic 3D scene reconstruction (4D reconstruction), and 3D human pose estimation from motion-blurred portrait photos. This demonstrates the geometric consistency and temporal coherence of generated videos beyond simple frame restoration.

- **Generalization to Historical Photographs**: The method successfully processes motion-blurred historical images captured over 80 years ago (e.g., 1944 WWII photos, 1971 boxing match), revealing scene dynamics using subtle motion blur cues. This demonstrates the method's robustness and practical utility for cultural heritage applications and historical image analysis.

## Core Insights

- **Large-Scale Pre-training Captures Motion Blur Priors**: The paper's fundamental insight is that large video diffusion models trained on billions of images and millions of video clips inherently learn strong priors about the relationship between motion blur and scene dynamics, even without explicit supervision on this task. Figure 2 demonstrates this: when given a motion-blurred soccer ball image, an off-the-shelf video model naturally predicts motion consistent with the blur direction, confirming that motion blur information is already encoded in these models.

- **Video Formulation is Superior to Image Restoration**: Rather than treating motion blur analysis as traditional deblurring (recovering a single sharp frame), the paper reformulates it as conditional video generation. This is fundamentally more appropriate because motion blur is inherently a temporal phenomenon—it encodes information about appearance changes over an exposure interval. By generating video sequences with controllable exposure intervals (Equation 1), the method naturally represents and exploits this spatiotemporal structure.

- **Exposure Interval Encoding Enables Precise Temporal Control**: The sinusoidal positional encoding of exposure intervals (Equations 4-5) allows the model to learn a continuous mapping between frame exposure windows and scene dynamics. This design choice is critical for past/present/future generation: by simply adjusting the exposure interval vectors $\tilde{T}_i$, the same model can generate frames spanning different temporal ranges without retraining.

- **Multi-Modal Distribution Reflects Real Motion Ambiguities**: Figures 7-8 reveal that the model learns a realistic multi-modal distribution of plausible videos. For example, a face with horizontal motion blur generates both leftward and rightward motion depending on the sample, accurately capturing genuine ambiguity. However, for disambiguated scenarios (e.g., taxis moving forward), the model consistently generates forward motion, demonstrating that it learns real-world motion priors rather than being uniformly multi-modal.

- **3D Consistency Emerges from Video Coherence**: The paper shows (Section 6.3) that generated videos maintain sufficient 3D geometric consistency to enable structure-from-motion reconstruction and 4D scene recovery. This is non-trivial: generating temporally coherent video sequences with consistent disparity cues across frames requires the model to implicitly learn 3D scene geometry, even though it's trained on 2D video data. Figure 9 validates this through epipolar geometry analysis and homography consistency tests.

- **Scale Matters More Than Architecture**: The paper attributes success primarily to the use of large-scale pre-training (2B parameter model on billions of images) rather than novel architectural innovations. The training uses standard VAE encoding, sinusoidal positional encoding, and diffusion denoising—all established techniques. The key difference from baselines is access to larger and more diverse training data, suggesting that scaling rather than architectural novelty is the primary driver of performance improvements.

## Key Data & Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | EPE↓ |
|---------|--------|-------|-------|--------|------|------|
| GoPro | Jin et al. [2018] | 25.23 | 0.8190 | 0.084 | 235.53 | 3.38 |
| GoPro | MotionETR [2021] | 26.54 | 0.8825 | 0.015 | 94.90 | 1.46 |
| GoPro | **Proposed** | **30.01** | **0.9359** | **0.010** | **21.46** | **0.39** |
| B-AIST++ | Animation from Blur [2022] | 26.69 | 0.9209 | 0.042 | 138.27 | 2.65 |
| B-AIST++ | **Proposed** | **27.37** | **0.9306** | **0.027** | **37.16** | **1.78** |

**Quantitative Results Summary:**
- **Present-only generation (during exposure) significantly outperforms baselines**: On GoPro, PSNR improves by 3.47 dB over the strongest baseline (MotionETR). More dramatically, FVD improves from 94.90 to 21.46 (4.4× improvement), indicating the generated videos are much closer to natural video distributions. EPE (optical flow error) drops from 1.46 to 0.39 pixels, demonstrating accurate motion prediction.

- **Past/Present/Future generation maintains quality across temporal range**: Figure 4 shows PSNR remains approximately 20-30 dB across past, present, and future frames when predicting 13 total frames (3 past, 7 during exposure, 3 future) from 7-frame ground truth. The consistency of quality across the temporal window suggests the motion-blurred image provides equally strong constraints for all time periods.

- **Ablation through fine-tuning strategy**: The paper fine-tunes three model variants: two on specific datasets (GoPro and B-AIST++) for direct baseline comparison, and one on a diverse compilation of high-FPS videos (694 clips from GoPro240, Adobe240, REDS, iPhone240, Sports240) for in-the-wild generalization. The diverse multi-dataset model (used for all in-the-wild results in Section 6.1-6.4) demonstrates superior generalization, validating the importance of training data diversity.

- **Computational efficiency**: Inference requires ~2 minutes on an NVIDIA L40 GPU with 50 diffusion steps. Training uses 16 L40 GPUs for 10 days with batch size 64 (20,000 iterations). While this is computationally expensive, it's a one-time cost; inference, though slow compared to deterministic methods, is reasonable for offline applications like historical photo restoration.

## Strengths

- **Novel Problem Formulation with Strong Motivation**: Reformulating motion blur analysis as conditional video generation (rather than image restoration) is conceptually elegant and well-motivated. Motion blur is fundamentally a temporal phenomenon encoding appearance changes over an exposure interval. Using videos to explain this is physically appropriate and enables past/present/future prediction—a genuinely novel capability absent in prior work. The motivation is clear and the approach is philosophically well-grounded.

- **Leverages Large-Scale Pre-training Effectively**: The paper makes a compelling argument for why large video diffusion models are appropriate for this task: they're trained on billions of images and millions of videos, naturally encoding priors about motion blur (Figure 2). Rather than proposing novel architectures, the authors pragmatically repurpose existing models (CogVideoX-2B), demonstrating that scale and pre-training data matter more than architectural novelty—an important empirical finding for the field.

- **Comprehensive Experimental Validation**: The paper includes quantitative evaluation on two standard benchmarks (GoPro, B-AIST++) with proper baselines, improved metrics (bidirectional patch-based evaluation), diverse downstream applications (tracking, 3D reconstruction, pose estimation), and qualitative results on challenging in-the-wild images. The supplemental webpage with video results (referenced but not shown here) appears thorough. The multi-modal distribution analysis (Figures 7-8) honestly addresses a key challenge: acknowledging that generated videos sample plausible alternatives rather than recovering ground truth.

- **Technical Soundness of Architectural Modifications**: The exposure interval encoding (Equations 4-5) is straightforward but clever—sinusoidal encoding maps continuous exposure intervals to high-dimensional representations, and the design naturally generalizes to any temporal range. The temporal upsampling to 1920 FPS during training (Section 4.2) demonstrates careful consideration of blur simulation fidelity. The model maintains temporal and spatial consistency without explicit geometric constraints, suggesting implicit learning of 3D structure.

- **Practical Impact and Generalization**: Success on diverse in-the-wild scenes (dancers, sports, animals, deforming cloth, cityscapes) and historical photographs demonstrates genuine robustness. The ability to process 80-year-old WWII photographs and recover subtle motion cues shows the method works on real-world, unconstrained images—not just synthetic or controlled datasets. Downstream applications (4D reconstruction, pose estimation) prove utility beyond novelty.

- **Honest Assessment of Limitations**: The paper acknowledges that generated videos represent samples from a multi-modal distribution, not ground truth recovery. Figure 4 shows PSNR degrades outside the exposure window (20-30 dB in past/future vs. 30+ dB in present), honestly reporting reduced accuracy for extrapolation. The paper doesn't overclaim that it "solves" motion blur analysis, but rather shows what can be learned from it.

## Weaknesses

- **Limited Quantitative Evaluation of Past/Present/Future Generation**: Section 5.2 provides minimal quantitative analysis of past/future prediction. Only PSNR per frame (Figure 4) is reported; absent are comparisons against baselines on this task and complementary metrics (SSIM, LPIPS, FVD, EPE). The statement "we omit baseline comparisons in this setting, as the baselines were not trained for this task" is problematic—this is precisely why comparison matters: to establish that this novel capability produces meaningful outputs. At minimum, metrics like FVD would assess whether past/future frames look photorealistic.

- **Insufficient Ablation Studies**: While the paper shows results on different fine-tuning datasets, it lacks systematic ablations of the architectural modifications. Specific missing analyses: (1) impact of exposure interval encoding—would position encoding alone suffice? (2) contribution of the linear projection layer for exposure encoding; (3) effect of sinusoidal frequency choices; (4) sensitivity to the temporal VAE compression ratio. Without these ablations, it's unclear which design choices are essential versus incidental.

- **Overclaimed 3D Consistency**: Sections 6.3 and 7.2 claim "3D consistency," but Figure 9's analysis is limited to a single example. The epipolar geometry visualization in Figure 9 shows consistency for camera motion, but only one frame pair is analyzed. Claims about "geometrically consistent video sequences" enabling 4D reconstruction lack quantitative support: (1) no metrics comparing 3D reconstruction quality versus baselines; (2) no analysis of how often 3D consistency holds (is it scene-dependent?); (3) no evaluation of whether geometric consistency is necessary for downstream tasks (could dense tracks work on non-geometric videos?).

- **Evaluation Metric Concerns**: The bidirectional patch-based metric (Equation 6) is introduced as superior, but no validation that it better correlates with human perception or task performance compared to frame-level bidirectional metrics. Why patch-based? How sensitive are results to patch size choices (1×1 for PSNR, 40×40 for SSIM on GoPro)? The paper doesn't justify these choices or study their impact. Additionally, FVD uses forward direction only ("computed at full resolution using videos played in the forward direction"), which seems to underutilize the bidirectional framework.

- **Insufficient Discussion of Failure Cases and Ambiguities**: While Figures 7-8 show multi-modality, failure modes are barely discussed. The paper mentions PSNR degradation (20-30 dB) in past/future but doesn't characterize when extrapolation fails. Missing analyses: (1) dependence on motion blur amount (is tiny blur handled differently than large blur?); (2) sensitivity to scene complexity or occlusions; (3) performance on highly non-rigid deformations; (4) behavior with multiple independent moving objects at different speeds.

- **Computational Cost and Practical Deployment**: Inference requires ~2 minutes per image on expensive GPU hardware (NVIDIA L40). This is prohibitive for real-time applications. Training requires 16 GPUs for 10 days—resource constraints that limit accessibility and reproducibility. The paper doesn't discuss optimization strategies (distillation, quantization, pruning) or potential for inference speedup. For a method claiming broad practical utility (historical photo restoration, etc.), computational efficiency is a significant practical limitation.

- **Limited Discussion of Generalization Boundaries**: The paper successfully handles diverse scenes but doesn't characterize the scope of applicability. When does the method fail? What properties of motion blur are necessary (e.g., is pure camera shake handled differently than scene motion)? How does camera response function $g$ (Equation 1) affect results if non-linear? Are there saturation effects or clipping that break the model?

## Research Directions

- **Efficient Video Generation for Real-Time Motion Deblurring**: Develop distilled or pruned versions of the diffusion model to achieve sub-second inference (target: 100-500 ms per image). This could involve knowledge distillation from the 2B-parameter CogVideoX model to a smaller student model, or consistency-based approaches (consistency models, adversarial distillation) that reduce diffusion steps from 50 to 10-20. Success would enable deployment in mobile/edge devices and real-time video deblurring applications, opening new markets for motion capture and action photography.

- **Explicit 3D-Aware Video Generation with Differentiable Rendering**: Extend the method by incorporating explicit 3D supervision during fine-tuning. Introduce a differentiable renderer that reconstructs the original blurred image from the generated video frames (verifying Equation 1), and optimize jointly on: (1) diffusion loss, (2) blur reconstruction loss, (3) 3D consistency loss (epipolar geometry, disparity smoothness). This could move beyond implicit 3D learning to provably geometric videos, enabling more reliable 4D reconstruction and pose estimation with quantified geometric error bounds.

- **Multi-Modal Uncertainty Estimation and Mode Selection**: Develop a framework for selecting or ranking samples from the multi-modal distribution (addressing Figures 7-8). Propose uncertainty metrics that quantify motion ambiguity per region, and a learnable selector that picks the most plausible mode given scene context (e.g., optical flow consistency across the generated video, human pose constraints). Combine with probabilistic inference to output distributions of plausible pasts/futures rather than single samples. This addresses a fundamental limitation: users currently can't distinguish ground truth from plausible alternatives.

- **Domain-Specific Fine-Tuning for Specialized Scenes**: Fine-tune separate expert models for high-value domains: sports (athletes in motion), medical imaging (surgical motion capture), autonomous driving (object tracking at high speeds), and cinematography (artistic motion control). For each domain, collect 500-1000 high-quality videos and fine-tune task-specific variants. Combine with domain-specific metrics (pose accuracy for sports, structural similarity for medical, tracking loss for autonomous driving). This mirrors the success of vision transformers on specialized tasks and could unlock industrial adoption.

- **Theoretical Analysis of Motion Blur Ambiguities and Solvability**: Develop formal theory characterizing when motion blur uniquely determines (or ambiguously explains) scene dynamics. Derive sufficient conditions for video recovery (e.g., bounds on motion magnitude, blur kernel properties, occlusion constraints). Analyze the optimization landscape of the conditional diffusion objective. Prove or disprove whether spatially-varying motion blur provides enough constraints to disambiguate between plausible videos. This theoretical grounding would elevate the method from empirical success to principled understanding.

- **Cross-Modal Conditioning for Guided Video Generation**: Extend the conditioning beyond exposure intervals to include additional information: text descriptions ("a person running forward"), optical flow priors, or rough 3D point clouds from classical SfM. Design a multi-modal fusion module that combines these signals with the motion-blurred image to constrain the solution space. Evaluate whether auxiliary information reduces ambiguity (e.g., text "person waving" eliminates motion samples where person runs). This could improve practical utility for interactive deblurring applications.

- **Evaluation on Adversarially-Designed and Real-Captured Blurred Videos**: Create a challenging benchmark of adversarially-selected motion-blurred images that maximize model uncertainty, and capture new real blur videos in controlled settings where ground truth is known (e.g., motion capture with simultaneous high-speed video + intentional blur injection). Evaluate proposed method and baselines on this benchmark to establish performance ceilings and identify systematic failure modes. Publish benchmark for community evaluation, similar to how KITTI advanced autonomous driving research.

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **為動作模糊分析重新用途化預訓練視頻擴散模型**：本文引入了一種新穎的技術，該技術改編了大規模預訓練視頻擴散模型（CogVideoX-2B，擁有20億參數），以從動作模糊圖像中恢復視頻。與依賴於手工製作先驗或小型訓練數據集（數萬個示例）的先前工作不同，該方法利用互聯網規模的視頻數據集，並展示了對複雜且真實場景（包括舞蹈演員、運動賽事和變形布料）的卓越泛化能力。

- **用於過去/現在/未來生成的精確曝光時間控制**：作者開發了一種新穎的調節機制，該機制通過正弦位置編碼和時間索引實現對每個生成幀的曝光間隔的精確控制。這種架構創新允許模型生成不僅是曝光窗口內的幀（"現在"），還可以預測捕捉時刻之前（過去）和之後（未來）的立即發生的情況—這是先前視頻解模糊方法中缺少的功能。

- **具有雙向補丁基準度量的綜合評估框架**：本文引入了改進的量化評估度量（方程式6），以解決動作模糊分析中的根本性挑戰：運動歧義（例如，時間反向視頻同樣有效）。雙向補丁基準度量 $M_p$ 獨立評估正向和時間反向補丁，提供對運動一致性的細粒度空間評估，而不是僅僅進行幀級度量。

- **多個基準上的最先進結果**：該方法在標準基準上大幅優於先前工作：在GoPro數據集上，它達到了30.01 dB的PSNR，而MotionETR達到26.54 dB，Jin等人達到25.23 dB；在B-AIST++數據集上，PSNR從26.69 dB改進到27.37 dB。最為關鍵的是，FVD（分佈相似性）在GoPro上從235.53改進到21.46，表明光真實感視頻生成與地面實況分佈相匹配。

- **多樣化的下游應用**：生成的視頻支持多個下游計算機視覺任務，包括3D對象跟蹤（使用CoTracker）、攝像頭軌跡恢復（使用MegaSaM）、動態3D場景重建（4D重建）和從動作模糊肖像照片中進行3D人類姿態估計。這展示了生成視頻的幾何一致性和時間連貫性，超越了簡單的幀恢復。

- **對歷史照片的泛化**：該方法成功處理了超過80年前拍攝的動作模糊歷史圖像（例如1944年二戰照片、1971年拳擊比賽），使用細微的動作模糊線索揭示場景動態。這展示了該方法的魯棒性和文化遺產應用與歷史圖像分析的實用性。

## 核心洞見

- **大規模預訓練捕捉動作模糊先驗**：本文的基本洞見是，在十億圖像和數百萬視頻片段上訓練的大規模視頻擴散模型本質上學習了關於動作模糊與場景動態之間關係的強先驗，即使沒有明確的此任務監督。圖2展示了這一點：給定運動模糊的足球圖像時，現成的視頻模型自然預測與模糊方向一致的運動，確認動作模糊信息已經編碼在這些模型中。

- **視頻表述優於圖像恢復**：該論文不是將動作模糊分析視為傳統去模糊（恢復單個銳利幀），而是將其重新表述為條件視頻生成。這在根本上更合適，因為動作模糊本質上是時間現象—它編碼了曝光間隔內外觀變化的信息。通過生成具有可控曝光間隔的視頻序列（方程式1），該方法自然地表示和利用這種時空結構。

- **曝光間隔編碼實現精確時間控制**：正弦位置編碼的曝光間隔（方程式4-5）允許模型學習幀曝光窗口與場景動態之間的連續映射。這種設計選擇對於過去/現在/未來生成至關重要：通過簡單地調整曝光間隔向量 $\tilde{T}_i$，同一模型可以生成跨越不同時間範圍的幀，無需重新訓練。

- **多模態分佈反映真實運動歧義**：圖7-8揭示了模型學習了可信視頻的現實多模態分佈。例如，具有水平運動模糊的臉可根據樣本生成向左和向右的運動，準確捕捉真正的歧義。但是，對於消歧場景（例如，計程車向前移動），模型始終生成向前運動，表明它學習了真實世界的運動先驗，而不是均勻多模態。

- **3D一致性從視頻連貫性中呈現**：論文顯示（第6.3節），生成的視頻保持足夠的3D幾何一致性以啟用結構從運動重建和4D場景恢復。這非平凡：生成具有跨幀一致視差線索的時間連貫視頻序列要求模型隱含地學習3D場景幾何，儘管它在2D視頻數據上訓練。圖9通過極線幾何分析和單應一致性測試驗證了這一點。

- **規模比體系結構更重要**：論文將成功主要歸因於使用大規模預訓練（20億參數模型在十億圖像上），而不是新穎的架構創新。訓練使用標準VAE編碼、正弦位置編碼和擴散去噪—所有既定技術。與基準線的主要區別是訪問更大和更多樣化的訓練數據，表明縮放而不是架構新穎性是性能改進的主要驅動因素。

## 關鍵數據與結果

| 數據集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | EPE↓ |
|---------|--------|-------|-------|--------|------|------|
| GoPro | Jin et al. [2018] | 25.23 | 0.8190 | 0.084 | 235.53 | 3.38 |
| GoPro | MotionETR [2021] | 26.54 | 0.8825 | 0.015 | 94.90 | 1.46 |
| GoPro | **提出方法** | **30.01** | **0.9359** | **0.010** | **21.46** | **0.39** |
| B-AIST++ | Animation from Blur [2022] | 26.69 | 0.9209 | 0.042 | 138.27 | 2.65 |
| B-AIST++ | **提出方法** | **27.37** | **0.9306** | **0.027** | **37.16** | **1.78** |

**量化結果總結：**
- **純現在生成（曝光期間）顯著優於基準線**：在GoPro上，PSNR比最強基準線（MotionETR）改進3.47 dB。更戲劇性的是，FVD從94.90改進到21.46（4.4倍改進），表明生成的視頻遠接近自然視頻分佈。EPE（光流誤差）從1.46像素下降到0.39像素，展示了準確的運動預測。

- **過去/現在/未來生成在時間範圍內保持質量**：圖4顯示，當從7幀地面實況預測13個總幀（3個過去、7個曝光期間、3個未來）時，PSNR在過去、現在和未來幀中保持約20-30 dB。質量在時間窗口中的一致性表明動作模糊圖像為所有時間段提供同樣強大的約束。

- **通過微調策略的消融**：論文微調了三個模型變體：前兩個在特定數據集上微調（GoPro和B-AIST++）用於與基準線的直接比較，第三個在高FPS視頻的多樣化編譯上微調（694個視頻片段，來自GoPro240、Adobe240、REDS、iPhone240和Sports240），用於真實泛化。用於所有第6.1-6.4節真實結果的多樣化多數據集模型展示了卓越的泛化性，驗證了訓練數據多樣性的重要性。

- **計算效率**：推理在NVIDIA L40 GPU上需要約2分鐘，使用50個擴散步驟。訓練使用16個L40 GPU，進行10天，批大小為64（20,000次迭代）。雖然這在計算上昂貴，但這是一次性成本；推理儘管與確定性方法相比緩慢，但對於離線應用（如歷史照片修復等）是合理的。

## 優勢

- **新穎的問題表述與強動機**：將動作模糊分析重新表述為條件視頻生成（而不是圖像恢復）在概念上是優雅且有充分動機的。動作模糊本質上是編碼曝光間隔內外觀變化的時間現象。使用視頻來解釋這一點在物理上是合適的，並實現了過去/現在/未來預測—一種先前工作中缺失的真正新穎功能。動機清晰，方法在哲學上是基礎良好的。

- **有效利用大規模預訓練**：論文為為什麼大規模視頻擴散模型適合此任務提出了令人信服的論點：它們在十億圖像和數百萬視頻上訓練，自然編碼關於動作模糊的先驗（圖2）。作者不是提出新穎的架構，而是實用地重新用途化現有模型（CogVideoX-2B），展示了規模和預訓練數據比體系結構新穎性更重要—這是該領域的一個重要實證發現。

- **全面的實驗驗證**：論文包括對兩個標準基準（GoPro、B-AIST++）的量化評估，具有適當的基準線、改進的度量（雙向補丁基準評估）、多樣化的下游應用（跟蹤、3D重建、姿態估計）和對具有挑戰性的真實圖像的定性結果。補充網頁中的視頻結果（在此引用但未顯示）看起來很全面。多模態分佈分析（圖7-8）誠實地解決了一個關鍵挑戰：承認生成的視頻抽樣合理的替代方案，而不是恢復地面實況。

- **架構修改的技術合理性**：曝光間隔編碼（方程式4-5）直接但巧妙—正弦編碼將連續曝光間隔映射到高維表示，設計自然泛化到任何時間範圍。訓練期間的時間上採樣至1920 FPS（第4.2節）展示了對模糊模擬保真度的仔細考慮。模型在沒有明確幾何約束的情況下保持時間和空間一致性，表明3D結構的隱含學習。

- **實際影響與泛化**：在多樣化真實場景（舞蹈演員、運動、動物、變形布料、城市景觀）和歷史照片上的成功展示了真正的魯棒性。能夠處理80年前的二戰照片並恢復細微運動線索展示該方法對真實、不受約束的圖像有效—不僅僅是合成或受控數據集。下游應用（4D重建、姿態估計）證明了超越新穎性的實用性。

- **對局限性的誠實評估**：論文承認生成的視頻表示從多模態分佈中的樣本，而不是地面實況恢復。圖4顯示PSNR在曝光窗口外降低（過去/未來中的20-30 dB對比現在中的30+ dB），誠實地報告外推的降低精度。論文沒有過度聲稱它"解決"動作模糊分析，而是展示可以從中學到什麼。

## 劣勢

- **過去/現在/未來生成的量化評估有限**：第5.2節對過去/未來預測的量化分析極少。只報告了每幀PSNR（圖4）；缺少與基準線的比較和該任務的補充度量（SSIM、LPIPS、FVD、EPE）。"我們在此設定中省略基準線比較，因為基準線未針對此任務訓練"的陳述是有問題的—這正是比較重要的原因：以建立此新穎功能產生有意義的輸出。至少，FVD等度量會評估過去/未來幀是否看起來光真實感。

- **消融研究不足**：雖然論文顯示了不同微調數據集上的結果，但缺少對架構修改的系統消融。特定缺失的分析：（1）曝光間隔編碼的影響—僅位置編碼是否足夠？（2）曝光編碼線性投影層的貢獻；（3）正弦頻率選擇的效果；（4）對時間VAE壓縮比的敏感性。沒有這些消融，不清楚哪些設計選擇是必要的對比偶然的。

- **過度聲稱3D一致性**：第6.3和7.2節聲稱"3D一致性"，但圖9的分析限於單個示例。圖9的極線幾何可視化顯示了攝像頭運動的一致性，但僅分析一對幀。關於"幾何一致視頻序列"實現4D重建的聲稱缺少量化支持：（1）沒有度量比較3D重建質量與基準線；（2）沒有分析3D一致性多久成立（是否取決於場景？）；（3）沒有評估幾何一致性是否對下游任務必要（密集軌跡能否在非幾何視頻上工作？）。

- **評估度量問題**：雙向補丁基準度量（方程式6）被引入為優越的，但沒有驗證它比幀級雙向度量更好地與人類感知或任務性能相關聯。為什麼基於補丁？結果對補丁大小選擇（GoPro上PSNR為1×1，SSIM為40×40）的敏感性如何？論文沒有證明這些選擇或研究它們的影響。此外，FVD僅使用前向方向（"在全分辨率下計算，使用向前播放的視頻），這似乎未充分利用雙向框架。

- **對失敗案例和歧義的討論不足**：雖然圖7-8顯示多模態，但失敗模式幾乎沒有討論。論文提到過去/未來中的PSNR降低（20-30 dB），但沒有表徵外推何時失敗。缺失分析：（1）對運動模糊量的依賴（是否以不同方式處理微小模糊與大模糊？）；（2）對場景複雜性或遮擋的敏感性；（3）高度非剛性變形的性能；（4）以不同速度移動多個獨立物體的行為。

- **計算成本與實際部署**：推理在昂貴GPU硬件（NVIDIA L40）上需要約2分鐘。這對實時應用來說是禁止性的。訓練需要16個GPU進行10天—限制可訪問性和可重現性的資源約束。論文不討論優化策略（蒸餾、量化、修剪）或推理加速的潛力。對於聲稱具有廣泛實際效用的方法（歷史照片修復等），計算效率是一個重要的實際限制。

- **泛化邊界的討論有限**：論文成功處理多樣化場景，但沒有表徵適用範圍。該方法何時失敗？動作模糊的哪些性質是必要的（例如，純攝像頭晃動是否以不同方式處理而不是場景運動）？如果非線性，攝像頭響應函數 $g$（方程式1）如何影響結果？是否存在飽和效應或裁剪會破壞模型？

## 研究方向

- **高效視頻生成用於實時運動去模糊**：開發蒸餾或修剪版本的擴散模型以實現亞秒推理（目標：每圖像100-500毫秒）。這可能涉及從20億參數CogVideoX模型到較小學生模型的知識蒸餾，或基於一致性的方法（一致性模型、對抗蒸餾），從50個擴散步驟減少到10-20個。成功將啟用在移動/邊緣設備上的部署和實時視頻去模糊應用，為運動捕捉和動作攝影打開新市場。

- **具有可微分渲染的顯式3D感知視頻生成**：通過在微調期間合併顯式3D監督來擴展該方法。引入可微分渲染器，從生成的視頻幀重建原始模糊圖像（驗證方程式1），並聯合優化：（1）擴散損失、（2）模糊重建損失、（3）3D一致性損失（極線幾何、視差平滑性）。這可以超越隱含3D學習，實現可證明的幾何視頻，具有量化幾何誤差界的更可靠4D重建和姿態估計。

- **多模態不確定性估計與模式選擇**：開發框架用於從多模態分佈中選擇或排列樣本（解決圖7-8）。提出量化運動歧義的不確定性度量（每個區域），以及可學習選擇器，給定場景上下文選擇最可信模式（例如，生成視頻的光流一致性、人類姿態約束）。與概率推理相結合，以輸出合理過去/未來的分佈，而不是單一樣本。這解決了根本限制：用戶目前無法區分地面實況與合理替代方案。

- **特定領域微調用於專門場景**：為高價值領域微調單獨的專家模型：運動（運動員運動）、醫學成像（手術運動捕捉）、自動駕駛（高速物體跟蹤）和電影製作（藝術運動控制）。對於每個領域，收集500-1000個高質量視頻並微調特定任務變體。與特定領域度量相結合（運動中的姿態精度、醫學中的結構相似性、自動駕駛中的跟蹤損失）。這反映了視覺轉換器在專門任務上的成功，並可能解鎖工業採用。

- **動作模糊歧義和可解性的理論分析**：開發形式理論，表徵動作模糊何時唯一確定（或模糊解釋）場景動態。推導視頻恢復的充分條件（例如，運動幅度邊界、模糊核心性質、遮擋約束）。分析條件擴散目標的優化景觀。證明或反駁空間變化運動模糊是否提供足夠約束以消歧合理視頻。這種理論基礎將該方法從經驗成功提升到原則理解。

- **用於引導視頻生成的跨模態調節**：擴展調節超出曝光間隔，包括附加信息：文本描述（"一個人向前跑步"）、光流先驗或粗略3D點雲（來自經典SfM）。設計多模態融合模塊，將這些信號與動作模糊圖像相結合，以約束解空間。評估輔助信息是否減少歧義（例如，文本"人揮手"消除了人跑的運動樣本）。這可以改進互動去模糊應用的實際效用。

- **對於對抗設計和真實捕捉模糊視頻的評估**：創建一個具有對抗選擇動作模糊圖像的具有挑戰性的基準，以最大化模型不確定性，並在受控設置中捕捉新的真實模糊視頻，其中已知地面實況（例如，動作捕捉與同時高速視頻加意圖模糊注入）。在此基準上評估提出方法和基準線，以建立性能上限並識別系統失敗模式。發佈基準供社區評估，類似於KITTI推進自動駕駛研究的方式。

</div>


