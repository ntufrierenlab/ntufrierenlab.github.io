---
title: "Chromatic Aberration Recovery on Arbitrary Images"
date: 2021-10-08
authors:
  - "Daniel J. Blueman"
source: "arXiv"
arxiv_url: "https://arxiv.org/abs/2110.04030"
pdf_url: "https://arxiv.org/pdf/2110.04030"
one_line_summary: "This paper presents a novel automatic algorithm for correcting lateral chromatic aberration in digital images using polynomial distortion models and spatial frequency-based quantification, with systematic validation across synthetic and real-world photographs."
one_line_summary_zh: "本論文提出了一個新穎的自動算法，利用多項式畸變模型和基於空間頻率的量化方法來校正數位影像中的橫向色差，並在合成和真實世界圖像上進行了系統驗證。"
date_added: 2026-02-21
topics: ["Chromatic Aberration"]
tags: []
---

<div class="lang-en">

## Key Contributions

- **Automatic LCA Minimization Algorithm**: The paper presents a novel **coefficient recovery algorithm** that robustly minimizes lateral chromatic aberration without manual user input. The algorithm uses chromatic correspondences and a **generalized polynomial distortion model** (rdest = arsrc + brsrc² + crsrc³ + drsrc⁴ + ...) to converge correction coefficients. This represents a significant advance over existing manual adjustment methods that rely on subjective visual inspection and typically only use two linear parameters, making the approach more accurate and adaptive to lens-specific variations.

- **Lens Model-Based Correction Database**: The paper introduces a methodology to correlate recovered correction information to specific lens models and parameters (focal length, aperture, focus distance), storing this in a reusable database for offline correction of unseen images. This addresses the practical limitation that manual correction work on individual images does not contribute to a generalizable lens model, enabling efficient batch processing and personalization to specific lens units accounting for manufacturing variance.

- **Spatial Frequency-Based Image Fidelity Quantification**: A novel **quantification algorithm** using spatial frequency analysis (DFT-based) is developed to measure image quality improvement in a way more relevant to Human Visual System (HVS) processing. Rather than relying on subjective assessment, this provides an objective metric that captures how LCA correction preserves or enhances perceptually important spatial frequencies, addressing a critical gap in existing correction validation methods.

- **Comprehensive Validation Framework**: The paper validates the algorithm through multiple levels of increasing complexity: synthetic chequerboard patterns for basic correctness verification, ray-traced office scenes for controlled testing, and diverse real-world photographs (26 images) captured under varying lighting and focus conditions. This systematic validation approach across synthetic and real data provides strong evidence of robustness across different scenarios.

- **Adaptive Correction for Lens Variance**: The methodology explicitly addresses manufacturing tolerance and wear in individual lens units, which cause sub-micron imperfections in element alignment. By optimizing correction coefficients for the specific lens instance from which images were captured (identified via camera body and lens identifiers stored in EXIF data), the approach adapts to real-world lens variation rather than assuming ideal optical designs.

## Core Insights

- **Non-linear Correction is Essential**: The paper demonstrates that LCA exhibits non-linear spatial variation across the image plane, as shown by the asymmetric distortion patterns in Figure 1.4 (chromatic misalignment up to ~10 pixels). Using only linear scaling parameters (as in manual methods) fundamentally cannot capture this behavior. The polynomial distortion model with multiple coefficients (b, c, d terms) is justified by Seidel aberration theory and empirical observation, revealing why previous two-parameter approaches inherently underperform on complex lens systems.

- **Spatial Frequency Analysis Captures Perceptual Quality**: The insight that image quality degradation from LCA manifests in spatial frequency loss is supported by the HVS photopic weighting function (Y = 0.299R + 0.587G + 0.114B) and the observation that LCA creates misaligned color planes which destructively interfere at high spatial frequencies. Figures 4.2, 4.5, and 4.7 demonstrate that DFT-based quantification reveals how correction recovers spatial frequency content that manual methods miss, providing a more principled quality metric than subjective visual inspection.

- **LCA Becomes Increasingly Critical with Sensor Resolution**: The paper establishes an important trend: while other classical aberrations (spherical aberration, astigmatism) can be mitigated through image enhancement like sharpening, LCA's relative impact increases with sensor resolution because the aberration is fundamentally a spatial misalignment that persists regardless of enhancement. This insight explains why historic optical designs adequate for lower-resolution sensors become problematic at modern megapixel levels, motivating computational post-processing solutions.

- **Coefficient Recovery Convergence Through Plane Subtraction**: The algorithm's approach of subtracting one color plane from another to create an error function (as shown in Figure 3.1) provides a direct signal for optimization. By minimizing chromatic correspondences (differences between color planes at corresponding spatial locations), the algorithm converges on physically meaningful distortion coefficients without requiring manual ground truth data. This is more elegant than pixel-level matching and avoids interpolation artifacts from demosaicing.

- **Demosaicing Order Matters for Accuracy**: The paper identifies that LCA correction should ideally occur before demosaicing the Bayer CFA, as advanced demosaicing algorithms can inadvertently introduce false color aliasing that changes the true LCA minima. This insight about preprocessing dependencies distinguishes rigorous post-processing workflows from oversimplified corrections applied after demosaicing, explaining why in-camera implementations (Section 2.5) must make accuracy compromises due to processing order constraints.

## Key Data & Results

| Evaluation Scenario | Method | Key Metric | Result |
|---|---|---|---|
| Synthetic Chequerboard | Proposed Algorithm | Difference map area reduction | Successful convergence to near-zero error |
| Ray-traced Scene | Proposed Algorithm | Spatial frequency preservation | Significant recovery of high-frequency detail |
| Real-world Images (3 tested) | Proposed Algorithm | Spatial frequency improvement | Measurable improvement in spatial frequency magnitude |
| Manual Correction Comparison | Adobe Photoshop CS4 | Difference map area | Larger residual error vs. proposed method |
| Runtime Performance | Proposed System | Execution time | Multiple stages: coefficient recovery + quantification |

- The **difference map area reduction** (Table 4.1 and Figure 4.9) shows the proposed algorithm successfully minimizes the magnitude of chromatic misalignment across test images, with quantitative reductions in error map area demonstrating convergence.

- **Spatial frequency analysis results** (Figures 4.2, 4.5, 4.7, 4.11, 4.13, 4.15, 4.17) reveal that correction recovers meaningful detail across multiple test images captured under diverse conditions (outdoor overcast, indoor low-light, bright daylight with focal lengths 18mm and 200mm equivalent).

- **Comparison with manual correction** (Figures 4.12, 4.14, 4.16) shows the proposed automatic method produces visually superior results to manual Photoshop correction, with smaller residual halos and better edge preservation visible in cropped sections.

- **Runtime cost analysis** (Figure 4.8) breaks down computational components, showing the algorithm is practical for post-processing workflows with acceptable computational overhead.

- **Robustness across real-world conditions**: The 26-image test suite spanning outdoor overcast (ISO 400, f/8.0, 50ms exposure), indoor low-light (ISO 800, f/6.3, up to 1s exposure), and bright daylight (ISO 200, f/11.0, 1-20ms exposure) demonstrates the algorithm handles significant variation in lighting and exposure without requiring parameter tuning.

## Strengths

- **Well-Motivated Problem**: The paper clearly establishes why LCA correction is timely and important: sensor resolution has increased dramatically, optical technology has not kept pace, and LCA's relative impact thus grows more significant. The motivation is supported by practical examples (Figure 1.4 showing ~10 pixel misalignment on professional equipment) rather than being purely theoretical.

- **Principled Technical Approach**: The use of Seidel aberration theory and polynomial distortion models grounds the method in established optical physics rather than ad-hoc heuristics. The generalized model (Equation 2.8) is justified theoretically and extends beyond the linear two-parameter approaches of prior work, providing a more accurate approximation of real lens behavior.

- **Novel Quantification Framework**: The development of a spatial frequency-based quality metric addresses a significant gap in prior literature. Rather than relying on subjective visual assessment or direct pixel-level metrics, the approach leverages HVS spectral sensitivity and provides an objective, interpretable measure of correction quality that can be automated and applied consistently.

- **Systematic Validation Strategy**: The evaluation progresses logically from synthetic validation (chequerboard patterns) to controlled scenes (ray-tracing) to diverse real-world images, providing increasing confidence in robustness. The comparison with a widely-used manual method (Photoshop) provides practical context rather than only comparing against other academic baselines.

- **Practical Design Considerations**: The paper thoughtfully addresses real-world deployment challenges including lens manufacturing variance (Section 2.4), demosaicing order effects (Section 2.5), EXIF metadata usage for parameter retrieval, and the limitations of in-camera implementations. This shows maturity beyond pure algorithm development.

- **Clear Presentation of Methods**: Algorithms are presented with pseudocode (Algorithms 3.1 and 3.2), and the overall workflow is clearly diagrammed (Figure 1.5). The background material adequately explains optical concepts needed to understand the contribution.

## Weaknesses

- **Limited Baseline Comparisons**: While the paper compares against manual Photoshop correction, it provides limited comparison with other computational LCA correction methods from the literature. Section 2.7.3 reviews computational approaches but the empirical evaluation (Section 4.3) appears to only compare against one baseline. A comparison with other automatic methods would strengthen the claims of superiority.

- **Small Test Set for Real-World Evaluation**: While 26 real-world images provide some diversity, this is relatively small for establishing robustness of a general imaging algorithm. The paper would benefit from evaluation on standardized image datasets or publicly available benchmarks (none appear to be used). The three "selected images" chosen for detailed comparison (Figures 4.12-4.17) may represent cherry-picked successes rather than representative performance.

- **Insufficient Ablation Studies**: The paper does not provide ablation studies examining the contribution of individual polynomial terms in the distortion model. How much improvement comes from quadratic vs. cubic vs. quartic terms? Is the full polynomial order necessary or would lower-order approximations suffice? This analysis would clarify the design choices and provide insight into where complexity is justified.

- **Limited Discussion of Failure Cases**: The paper acknowledges limitations (Section 5.1.4) but does not provide detailed analysis of failure modes. When does the algorithm struggle? Are there lens types or image types where it performs poorly? The discussion of limitations (tilt-shift lenses, certain demosaicing algorithms) is brief and would benefit from concrete examples or experimental evidence of failure.

- **Incomplete Parameter Database Evaluation**: While the paper proposes storing correction coefficients in a database indexed by lens model and shooting parameters (focal length, aperture, focus distance), the evaluation does not thoroughly test database lookup and generalization. How well do coefficients learned from one image generalize to other images from the same lens at the same parameters? This is critical for practical deployment but is not empirically evaluated.

- **Lack of Theoretical Justification for Convergence**: The optimization procedure (Section 3.6) uses L-BFGS-B minimizer but does not prove or empirically demonstrate that the error function is unimodal or that convergence to a global optimum occurs. For a robust automatic system, convergence guarantees or sensitivity analysis would strengthen confidence in reliability across diverse images.

## Research Directions

- **Deep Learning-Based Coefficient Prediction**: Develop a convolutional neural network that predicts LCA correction coefficients directly from image patches without iterative optimization. The network could be trained on the parameter database built during this work, learning to map image features and EXIF parameters to optimal coefficients. This would dramatically reduce runtime (seconds to milliseconds) and enable real-time mobile processing, advancing the practical applicability to smartphone photography where optical quality limitations are severe.

- **Multi-Wavelength Spectral Correction**: Extend the three-channel (RGB) approach to handle the full visible spectrum by decomposing LCA effects per wavelength rather than just red-green-blue triplets. This could be particularly valuable for scientific imaging and high-end photography. The method could leverage emerging computational imaging techniques using spectral filter banks or coded apertures to capture richer spectral information for more precise correction.

- **Joint LCA and Demosaicing Optimization**: Instead of assuming a fixed demosaicing algorithm, jointly optimize the demosaicing and LCA correction stages as an end-to-end problem. Train a network or design an algorithm that accounts for how demosaicing artifacts interact with LCA correction, potentially recovering fidelity lost when these operations are performed sequentially. This addresses the insight from Section 2.5 about demosaicing-correction ordering.

- **Cross-Lens Generalization via Domain Adaptation**: Investigate whether correction coefficients learned for one lens model transfer to similar lenses (same manufacturer, similar optical design) through domain adaptation techniques. Develop methods to predict coefficients for unseen lens models by learning a manifold of lens behaviors, reducing the database requirements and enabling better performance on new equipment without retraining.

- **Hardware-Software Co-Design for In-Camera Correction**: Extend the work to real in-camera implementation on embedded processors and FPGAs, with hardware acceleration for the most computationally intensive steps (spatial frequency analysis, polynomial distortion application). Develop a reduced-order approximation of the full algorithm that maintains accuracy while meeting strict computational budgets, advancing practical deployment on actual cameras.

- **Perceptual Loss Functions Beyond Spatial Frequency**: Move beyond DFT-based spatial frequency metrics to incorporate perceptual metrics based on modern understanding of visual perception (e.g., steerable pyramid representations, deep perceptual features). Validate whether more sophisticated perceptual metrics better correlate with human preference judgments for LCA correction, potentially improving the quantification algorithm's alignment with subjective quality assessment.

- **Handling Multiple Aberration Types Jointly**: Extend the method to simultaneously correct not just LCA but also axial chromatic aberration (ACA), geometric distortion, and other monochromatic aberrations in a unified framework. Model the joint error surface and develop optimization procedures that balance corrections across aberration types, since correcting one aberration type can potentially exacerbate others if not carefully managed.

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **自動色差最小化算法**：論文提出一個新穎的**係數恢復算法**，能夠在無需手動使用者輸入的情況下強健地最小化橫向色差（LCA）。該算法使用色度對應關係和**廣義多項式畸變模型**（rdest = arsrc + brsrc² + crsrc³ + drsrc⁴ + ...）來收斂校正係數。相比於現有的基於主觀視覺檢查且通常僅使用兩個線性參數的手動調整方法，該方法代表了重大進展，提供了更高的準確度和對鏡頭特定變化的適應性。

- **基於鏡頭模型的校正數據庫**：論文引入了一種方法論，將恢復的校正信息與特定的鏡頭模型和參數（焦距、光圈、對焦距離）相關聯，將其儲存在可重複使用的數據庫中，以用於離線校正未見過的圖像。這解決了以下實際限制：對單個圖像進行的手動校正工作不能對通用鏡頭模型有所貢獻，該方法使得批量處理和針對特定鏡頭單元的個性化成為可能，考慮到製造公差。

- **基於空間頻率的圖像保真度量化算法**：開發了一個新穎的**量化算法**，使用空間頻率分析（基於DFT）來測量與人類視覺系統（HVS）處理更相關的圖像質量改進。該算法不依賴於主觀評估，而是提供客觀、可解釋的指標，能捕捉色差校正如何保留或增強感知上重要的空間頻率，解決了現有校正驗證方法中的關鍵空白。

- **全面驗證框架**：論文通過多個複雜性遞增的層次驗證算法：合成棋盤圖案用於基本正確性驗證，光線追蹤辦公場景用於受控測試，以及多樣化的真實世界圖像（26張）在不同光線和對焦條件下捕捉。這種跨合成和真實數據的系統驗證方法提供了強有力的證據，證明了算法在不同場景中的穩健性。

- **針對鏡頭變化的自適應校正**：該方法明確解決了個別鏡頭單元中的製造公差和磨損，這些因素在元件對齐中造成亞微米級的不完美。通過為拍攝圖像的特定鏡頭實例優化校正係數（通過儲存在EXIF數據中的相機機身和鏡頭標識符識別），該方法適應真實世界的鏡頭變化，而不是假設理想的光學設計。

## 核心洞見

- **非線性校正是必不可少的**：論文展示了色差在整個圖像平面上表現出非線性的空間變化，如圖1.4所示的不對稱畸變圖案（色度錯位高達≈10個像素）。僅使用線性縮放參數（如在手動方法中）根本上無法捕捉這種行為。多項式畸變模型具有多個係數（b、c、d項）由Seidel像差理論和經驗觀察證明是必要的，揭示了為什麼之前的兩參數方法在複雜鏡頭系統上的性能基本上不足。

- **空間頻率分析捕捉感知質量**：圖像質量因色差而的降低表現在空間頻率損失中這一洞見由HVS光視效率函數（Y = 0.299R + 0.587G + 0.114B）和色差在高空間頻率處會導致錯位色平面的觀察所支持。圖4.2、4.5和4.7展示了基於DFT的量化如何揭示手動方法遺漏的校正恢復的空間頻率內容，提供了一個相比主觀視覺檢查更有原則的質量度量。

- **色差隨著傳感器分辨率的提高而變得越來越重要**：論文建立了一個重要的趨勢：雖然其他經典像差（球面像差、散光）可以通過銳化等圖像增強來減輕，但色差的相對影響隨著傳感器分辨率的增加而增加，因為像差本質上是一種空間錯位，無論增強如何都會持續存在。這一洞見解釋了為什麼在低分辨率傳感器時代設計的光學設計在現代百萬像素級別變得有問題，推動了計算後處理解決方案的必要性。

- **通過平面相減進行係數恢復收斂**：算法通過將一個色平面從另一個相減來創建誤差函數（如圖3.1所示）的方法提供了優化的直接信號。通過最小化色度對應關係（色平面之間相應空間位置的差異），算法收斂到物理上有意義的畸變係數，無需手動真實值數據。這比像素級匹配更優雅，避免了來自馬賽克去除的插值偽影。

- **去馬賽克順序對準確性很重要**：論文指出色差校正理想情況下應在Bayer CFA去馬賽克之前進行，因為先進的去馬賽克算法可能會無意中引入虛假色彩混疊，改變真正的色差最小值。這一關於預處理依賴關係的洞見區分了嚴格的後處理工作流程與在去馬賽克後應用的過度簡化校正，解釋了為什麼機內實現（第2.5節）必須由於處理順序約束而做出準確性妥協。

## 關鍵數據與結果

| 評估場景 | 方法 | 關鍵指標 | 結果 |
|---|---|---|---|
| 合成棋盤 | 提出的算法 | 差異圖面積縮減 | 成功收斂到接近零的誤差 |
| 光線追蹤場景 | 提出的算法 | 空間頻率保留 | 高頻細節的顯著恢復 |
| 真實世界圖像（3張測試） | 提出的算法 | 空間頻率改進 | 空間頻率幅度的可測量改進 |
| 手動校正比較 | Adobe Photoshop CS4 | 差異圖面積 | 相比提出的方法具有更大的殘餘誤差 |
| 運行時性能 | 提出的系統 | 執行時間 | 多個階段：係數恢復+量化 |

- **差異圖面積縮減**（表4.1和圖4.9）顯示提出的算法成功地最小化了測試圖像中的色度錯位幅度，量化的誤差圖面積縮減證明了收斂性。

- **空間頻率分析結果**（圖4.2、4.5、4.7、4.11、4.13、4.15、4.17）顯示校正在多個在不同條件下捕捉的測試圖像中恢復了有意義的細節（室外陰天、室內弱光、明亮日光，焦距為18mm和200mm等效）。

- **與手動校正的比較**（圖4.12、4.14、4.16）顯示提出的自動方法相比於手動Photoshop校正產生了視覺上優越的結果，在裁剪部分中可見更小的殘餘光暈和更好的邊緣保留。

- **運行時成本分析**（圖4.8）分解了計算成分，顯示該算法對於後處理工作流程是實用的，計算開銷可以接受。

- **在真實世界條件下的魯棒性**：26張圖像測試套件跨越室外陰天（ISO 400、f/8.0、50ms曝光）、室內弱光（ISO 800、f/6.3、最多1s曝光）和明亮日光（ISO 200、f/11.0、1-20ms曝光）證明該算法可以處理光線和曝光的巨大變化，無需參數調整。

## 優勢

- **充分的問題動機**：論文清楚地確立了為什麼色差校正是及時且重要的：傳感器分辨率急劇增加，光學技術並未跟上步伐，因此色差的相對影響變得越來越顯著。動機由實用例子（圖1.4顯示專業設備上≈10像素的錯位）支持，而不是純粹的理論考慮。

- **有原則的技術方法**：使用Seidel像差理論和多項式畸變模型將該方法建立在已確立的光學物理基礎之上，而不是臨時啟發式方法。廣義模型（方程2.8）在理論上是有根據的，並超越了先前工作的線性兩參數方法，提供了對真實鏡頭行為的更準確近似。

- **新穎的量化框架**：基於空間頻率的質量度量的開發解決了先前文獻中的重要空白。該方法不依賴主觀視覺評估或直接像素級指標，而是利用HVS光譜敏感性並提供客觀、可解釋的校正質量度量，可以自動化應用並一致地應用。

- **系統的驗證策略**：評估從合成驗證（棋盤圖案）邏輯進展到受控場景（光線追蹤）再到多樣化的真實世界圖像，不斷提高對魯棒性的信心。與廣泛使用的手動方法（Photoshop）的比較提供了實用背景，而不是僅與其他學術基準進行比較。

- **實用的設計考慮**：論文周到地解決了真實世界部署的挑戰，包括鏡頭製造公差（第2.4節）、去馬賽克順序效應（第2.5節）、EXIF元數據用於參數檢索，以及機內實現的限制。這顯示了超越純粹算法開發的成熟度。

- **清晰的方法展示**：算法以偽代碼形式呈現（算法3.1和3.2），整體工作流程清晰地圖示化（圖1.5）。背景材料充分解釋了理解貢獻所需的光學概念。

## 劣勢

- **基準比較有限**：雖然論文與手動Photoshop校正進行了比較，但它與文獻中的其他計算色差校正方法的比較有限。第2.7.3節回顧了計算方法，但經驗評估（第4.3節）似乎僅與一個基準進行比較。與其他自動方法的比較會加強優越性聲明。

- **實際評估中的小測試集**：雖然26張真實世界圖像提供了一定的多樣性，但對於建立通用成像算法的魯棒性而言相對較小。論文將受益於在標準化圖像數據集或公開可用基準上的評估（似乎未使用任何數據集）。為詳細比較選擇的三個"選定圖像"（圖4.12-4.17）可能代表精心選擇的成功案例，而不是代表性的性能。

- **消融研究不足**：論文未提供消融研究來檢查畸變模型中各個多項式項的貢獻。有多少改進來自二次項與三次項與四次項？完整的多項式階數是否必要，還是低階近似就足夠了？這種分析將澄清設計選擇並提供對複雜性正當性的洞見。

- **有限的失敗情況討論**：論文承認了限制（第5.1.4節），但未提供失敗模式的詳細分析。算法何時遇到困難？是否存在它性能不佳的鏡頭類型或圖像類型？關於限制的討論（移軸鏡頭、某些去馬賽克算法）很簡短，將受益於具體例子或失敗的經驗證據。

- **參數數據庫評估不完整**：雖然論文提出將校正係數儲存在由鏡頭模型和拍攝參數（焦距、光圈、對焦距離）索引的數據庫中，但評估並未徹底測試數據庫查找和泛化。從一個圖像學到的係數對於同一鏡頭在相同參數的其他圖像的泛化效果如何？這對實際部署至關重要但未進行經驗評估。

- **收斂缺乏理論正當性**：優化程序（第3.6節）使用L-BFGS-B最小化器，但未證明或經驗證明誤差函數是單峰的或收斂到全局最優值發生。對於強健的自動系統，收斂保證或敏感性分析將增強在不同圖像中可靠性的信心。

## 研究方向

- **基於深度學習的係數預測**：開發一個卷積神經網絡，直接從圖像塊預測色差校正係數，無需迭代優化。該網絡可以在該工作期間建立的參數數據庫上進行訓練，學習將圖像特徵和EXIF參數映射到最優係數。這將大幅減少運行時間（秒到毫秒），並支持實時移動處理，推進對光學質量限制嚴重的智能手機攝影的實際適用性。

- **多波長光譜校正**：將三通道（RGB）方法擴展為通過按波長分解色差效應而不是僅按紅綠藍三元組來處理整個可見光譜。這對科學成像和高端攝影尤其有價值。該方法可以利用新興的計算成像技術，如光譜濾波器組或編碼孔徑，以捕捉更豐富的光譜信息以進行更精確的校正。

- **色差與去馬賽克聯合優化**：不假設固定的去馬賽克算法，而是將去馬賽克和色差校正階段聯合優化為端到端問題。訓練一個網絡或設計一個考慮去馬賽克偽影如何與色差校正相互作用的算法，潛在地恢復這些操作順序執行時損失的保真度。這解決了第2.5節關於去馬賽克-校正順序的洞見。

- **通過域自適應的跨鏡頭泛化**：調查為一個鏡頭模型學到的校正係數是否通過域自適應技術轉移到類似的鏡頭（同一製造商、類似的光學設計）。開發方法通過學習鏡頭行為的流形來為未見過的鏡頭模型預測係數，減少數據庫要求，並在無需重新訓練的情況下在新設備上啟用更好的性能。

- **機內校正的硬體-軟體協同設計**：將該工作擴展至嵌入式處理器和FPGA上的真實機內實現，並對最計算密集的步驟（空間頻率分析、多項式畸變應用）進行硬體加速。開發完整算法的縮減階數近似，保持準確性同時滿足嚴格的計算預算，推進在實際相機上的實際部署。

- **超越空間頻率的感知損失函數**：超越基於DFT的空間頻率指標，採用基於現代視覺感知理解的感知指標（例如可轉向金字塔表示、深層感知特徵）。驗證更複雜的感知指標是否更好地與人類對色差校正的偏好判斷相關聯，潛在地改進量化算法與主觀質量評估的對齐。

- **聯合處理多個像差類型**：將該方法擴展以同時校正不僅是橫向色差，還有軸向色差（ACA）、幾何畸變和其他單色像差在統一框架中。建立聯合誤差表面模型並開發在像差類型之間平衡校正的優化程序，因為校正一種像差類型可能會在管理不當時潛在地加重其他情況。

</div>


