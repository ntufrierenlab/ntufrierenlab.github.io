---
title: "After the Party: Navigating the Mapping From Color to Ambient Lighting"
date: 2025-01-01
authors:
  - "Vasluianu"
  - "Florin-Alexandru"
  - "Seizinger"
  - "Tim"
  - "Wu"
  - "Zongwei"
  - "Timofte"
  - "Radu"
source: "ICCV"
arxiv_url: "https://openaccess.thecvf.com/content/ICCV2025/html/Vasluianu_After_the_Party_Navigating_the_Mapping_From_Color_to_Ambient_ICCV_2025_paper.html"
pdf_url: "https://openaccess.thecvf.com/content/ICCV2025/papers/Vasluianu_After_the_Party_Navigating_the_Mapping_From_Color_to_Ambient_ICCV_2025_paper.pdf"
one_line_summary: "This paper introduces CL3AN, the first dataset for ambient lighting normalization under multi-colored directional lights, and proposes RLN2, a Retinex-inspired dual-stream architecture leveraging HSV guidance that achieves state-of-the-art performance with improved computational efficiency."
one_line_summary_zh: "本論文介紹 CL3AN，首個針對多色方向光下環境光照標準化的數據集，並提出 RLN2，一個利用 HSV 指導的 Retinex 啟發雙流架構，以改進的計算效率實現最先進的性能。"
date_added: 2026-03-11
topics: ["Color Mapping"]
tags: []
---

<div class="lang-en">

## Research Directions

- **Extending ALN to Dynamic Scenes and Video**: The current RLN2 and CL3AN focus on static images. A natural extension is to develop video ALN that enforces temporal consistency while handling dynamic scene changes, camera motion, and flickering artifacts caused by temporal lighting variations. This would involve designing recurrent or attention-based mechanisms to propagate lighting estimates across frames. The core contribution could be a temporal coherence loss that prevents inconsistent normalization between adjacent frames while allowing smooth transitions when lighting conditions genuinely change. Applications in video editing and streaming content correction are significant.

- **Generative Refinement for Out-of-Domain Robustness**: While Figure 8 demonstrates reasonable out-of-domain performance, a hybrid approach combining RLN2's efficiency with generative priors could improve robustness. Consider a two-stage pipeline: RLN2 provides a fast initial normalization, then a lightweight diffusion model refines the result conditioned on the RLN2 output and guidance from the original input's color statistics. This leverages RLN2's interpretability and efficiency while gaining the robustness of generative models. The key innovation would be designing a conditioning mechanism that preserves RLN2's content while fixing only color/lighting artifacts.

- **Learnable Guidance and Adaptive Color Space Selection**: Instead of fixing HSV guidance, develop a meta-learning framework where the network learns which color space guidance is most informative for each image region or material type. This could involve gating multiple color space representations (HSV, YCbCr, YUV, etc.) with learned weights, allowing the model to adaptively select guidance. Another direction is learning task-specific guidance spaces (e.g., representations optimized specifically for disentangling illumination from reflectance) through self-supervised pretext tasks like predicting lighting direction or material reflectance properties.

- **Material-Aware ALN with Explicit Material Recognition**: Extend RLN2 to jointly estimate material properties (reflectance, roughness, specularity) alongside lighting normalization. A multi-task framework could include branches for material prediction, with explicit losses encouraging physically consistent decompositions. Training could leverage physically-based rendering engines (like Mitsuba) to generate synthetic ground truth with known material and lighting parameters. This would improve handling of specular and subsurface-scattering materials, addressing a current weakness in HSV-guided methods.

- **Scaling to Extreme Lighting Conditions and Real-World Capture**: Develop new capture procedures and datasets for more extreme scenarios: (1) mixed natural-artificial lighting, (2) highly directional sunlight combined with colored artificial lights, (3) reflective and specular-dominated scenes. Partner with professional photographers or product photographers to capture real-world challenging cases. This would validate whether CL3AN's studio-based insights generalize and potentially reveal new challenges requiring architectural innovations beyond current approaches.

- **Theoretical Analysis and Interpretability of ALN**: Develop formal theoretical analysis of the ALN problem under the multiplicative decomposition model (Equation 1). Under what conditions is reflectance-illumination decomposition identifiable? What assumptions about light sources (number, directionality, color) enable stable estimation? Derive sufficient conditions for RGB-sufficiency (i.e., when 3-channel information uniquely determines L and R). Complementarily, improve RLN2's interpretability through visualization of learned H and S refinement patterns and analysis of what frequency components CDFFA prioritizes for different material types.

- **Integration with Downstream Tasks and End-to-End Learning**: Evaluate whether ALN preprocessing improves performance on downstream tasks (face recognition, object detection, image segmentation) when tested on images under variable lighting. Develop end-to-end learning frameworks where ALN is a differentiable preprocessing stage optimized jointly with task-specific heads. This pragmatic direction would demonstrate whether studio-based ALN improvements translate to real-world benefits, potentially attracting industrial adoption and funding for further dataset expansion.

</div>

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **CL3AN 數據集**：本論文介紹了第一個針對多色方向光源下的環境光照標準化(ALN)而設計的大規模、高分辨率數據集。包含 105 個雜亂場景（3,667 個訓練樣本、437 個驗證、431 個測試），以 24 MP 分辨率以 RAW 和 RGB 格式捕獲，CL3AN 獨特地捕獲了非均勻 RGB 光照場景及其對應的環境光照地面真值。這解決了先前工作（ISTD+、AMBIENT6K）的關鍵缺口，這些工作要麼使用自然白光，要麼使用白平衡多光源場景，缺乏色度多樣性和材料特定反射率變化，無法研究真實世界的光照標準化。

- **綜合基準測試研究**：論文對 11 種領先的圖像復原方法在 CL3AN 上訓練時進行廣泛的實證分析，揭示關鍵失敗模式：光照不一致、紋理洩漏和色彩失真。這項基準測試貢獻對社區具有價值，為未來工作提供參考點。比較涵蓋多種架構（CNN、Transformer、狀態空間模型）和先驗知識（RGB、頻域），實現公平的架構分析。

- **RLN2 框架**：論文提出一個受 Retinex 理論啟發的雙流架構，明確地並行建模反射率(R)和亮度(L)成分，偏離了先前方法使用的隱式映射。關鍵創新涉及利用 HSV 圓柱色彩空間作為物理基礎的指導：V 通道提供反射率信息（編碼材料特定強度），而 H 和 S 編碼入射光下的色度。這種通過領域特定線索的明確分解使得能夠進行精確的光照-反射率分離，無需要求明確的像素級或頻域監督。

- **跨域特徵融合注意力(CDFFA)**：在 RLN2 精煉塊中，論文引入 CDFFA，通過基於 HSV 導出的指導的相關性增強來融合 RGB 特徵與低頻 DWT 分量。該模塊使用點積相似性連接特徵與指導分量，而不是隱式學習濾波器，相比於先前頻域感知方法（如 SFNet 和 IFBlend）實現更靈活和可解釋的頻率濾波機制。

- **最先進的性能與計算效率**：RLN2-Lf 在 CL3AN 上達到 PSNR 20.523（SSIM 0.746、LPIPS 0.208），在 AMBIENT6K 上達到 PSNR 21.712（SSIM 0.825、LPIPS 0.120），超越 IFBlend（先前 SOTA），同時減少約 15% 的計算成本（22.72G vs. ~26.01G MACs）。關鍵地，輕量級變體如 RLN2-S（3.95G MACs）保持競爭性能，使方法在實際部署中可行，同時表明架構選擇（廣泛感受野優於 Transformer）對 ALN 任務的重要性超過模型規模。

## 核心洞見

- **HSV 色彩空間提供最優的物理基礎指導**：論文的關鍵洞見是圓柱形 HSV 表示自然地與光線-材料相互作用的物理性質對齊：Value (V) 通道編碼反射率信息（與材料亮度和平滑度相關），而 Hue (H) 和 Saturation (S) 捕獲入射光的色度效應。消融研究（表 2）明確驗證了這一點：HSV 指導達到 SSIM 0.731，相比簡單連接的 0.703，以及 CDFFA 注意力的 0.746。這超越替代指導方案（RGB、L*a*b*），表明色彩空間幾何與物理圖像形成之間的對齊對特徵精煉至關重要。

- **明確分解大幅減少伪影**：RLN2 變體的比較表明，明確的反射率和亮度指導是必要的。即使從 R.blocks 中移除注意力機制（表 2：CDFFA）也會將 SSIM 從 0.746 降至 0.718，LPIPS 從 0.208 增至 0.233。論文的洞見是通過平行流聯合學習兩個分量可以防止困擾單流方法的光照和反射率的隱式糾纏。圖 6 和 7 的定性驗證表明，RLN2 在色彩標準化和陰影補償方面優於 Restormer 和 IFBlend。

- **非均勻色光照需要空間變化的校正**：與先前工作（ISTD+、AMBIENT6K）假設均勻或白平衡光照不同，論文表明 RGB 光照在場景中創建空間變化的色度偏移。圖 3 說明了在多色光照下直方圖和色調-飽和度統計的戲劇性變化，使全局色彩校正成為不可能。非均勻性是關鍵：不同場景區域接收來自不同 RGB 光源的光，與材料反射率的相互作用產生局部不同的色偏，只有空間特徵精煉可以解決。

- **頻域信息提供一致的邊際增益**：比較 RLN2-Sf 與 RLN2-S（有/無頻域）以及 RLN2-Lf 與 RLN2-L 顯示一致的改進：CL3AN 上 SSIM 提升 0.012，AMBIENT6K 上提升 0.004；LPIPS 分別改進 0.013 和 0.003。論文指出雖然計算成本增加（0.29G MACs），頻域特徵幫助保留細粒度細節。這與其他頻域感知方法（IFBlend、SFNet）也達到競爭結果的觀察一致，表明聯合空間-頻域處理對 ALN 有益但不是必需的。

- **感受野架構比模型規模更重要**：表 3 揭示令人驚訝的發現：具有廣泛感受野的基於 CNN 的方法（MPRNet、HINet）在兩個數據集上都優於優化局部依賴性的 Transformer 架構（SwinIR、Uformer、Restormer）。儘管比 RLN2-Lf 大 63%-87%，MPRNet 在 CL3AN 上達到較低 LPIPS（0.248 vs. 0.208），但 RLN2-Lf 通過物理基礎設計而非規模超越所有替代方案。這表明 ALN 受益於理解全局光照上下文，但該上下文可以通過雙流分解而非單純的大感受野更有效地捕獲。

- **材料特定反射率變化由 V 通道捕獲**：論文的材料涵蓋導體、介質及變化的透明度/粗糙度。HSV 中的 V 通道自然地編碼與這些材料特性相關的亮度變化，為反射率估計提供隱式先驗。這解釋了為什麼 HSV 優於平坦 RGB 指導：HSV 的亮度分量與物理反射率直接相關，而 RGB 需要網絡隱式學習該關係。

## 關鍵數據與結果

| 方法 | 類型 | 先驗 | CL3AN PSNR↑ | CL3AN SSIM↑ | CL3AN LPIPS↓ | AMBIENT6K PSNR↑ | AMBIENT6K SSIM↑ | AMBIENT6K LPIPS↓ | MACs (G) |
|------|------|------|------------|-----------|------------|-----------------|-----------------|-----------------|----------|
| Unprocessed | - | - | 10.837 | 0.447 | 0.518 | - | - | - | - |
| MPRNet | CNN | RGB | 18.453 | 0.688 | 0.291 | 20.947 | 0.820 | 0.129 | 37.21 |
| NAFNet | CNN | RGB | 19.476 | 0.709 | 0.249 | 20.580 | 0.808 | 0.142 | 15.92 |
| SFNet | Transf. | RGB+頻 | 18.382 | 0.686 | 0.291 | 20.519 | 0.812 | 0.141 | 31.27 |
| SwinIR | Transf. | RGB | 16.386 | 0.643 | 0.372 | 20.528 | 0.817 | 0.131 | 37.81 |
| Uformer | Transf. | RGB | 17.508 | 0.655 | 0.313 | 20.776 | 0.818 | 0.131 | 19.33 |
| Restormer | Transf. | RGB | 19.388 | 0.707 | 0.248 | 21.141 | 0.817 | 0.132 | 42.68 |
| HINet | CNN | RGB | 19.388 | 0.707 | 0.248 | - | - | - | 42.68 |
| IFBlend | CNN | RGB+頻 | 20.370 | 0.720 | 0.228 | 21.443 | 0.819 | 0.128 | 26.01 |
| MAMBAIR | Transf+SSM | RGB | 18.970 | 0.704 | 0.254 | - | - | - | 34.32 |
| GRL | Transf. | RGB | 18.089 | 0.672 | 0.308 | 20.856 | 0.821 | 0.129 | 2.16 |
| Retinexformer | Transf. | RGB | 18.649 | 0.683 | 0.281 | - | - | - | 4.86 |
| **RLN2-S** | **CNN** | **RGB** | **19.992** | **0.718** | **0.236** | **21.181** | **0.815** | **0.131** | **3.95** |
| **RLN2-Sf** | **CNN** | **RGB+頻** | **20.128** | **0.730** | **0.223** | **21.333** | **0.819** | **0.128** | **4.24** |
| **RLN2-L** | **CNN** | **RGB** | **20.383** | **0.723** | **0.222** | **21.553** | **0.821** | **0.123** | **22.45** |
| **RLN2-Lf** | **CNN** | **RGB+頻** | **20.523** | **0.746** | **0.208** | **21.712** | **0.825** | **0.120** | **22.72** |

- **在多色 CL3AN 數據集上的卓越性能**：RLN2-Lf 在 CL3AN 上達到 PSNR 20.523（SSIM 0.746、LPIPS 0.208），超越先前最佳基線 IFBlend（PSNR 20.370、SSIM 0.720、LPIPS 0.228）。改進特別體現在感知質量（LPIPS：-9.7%）和結構相似性（SSIM：+3.6%）方面，表明該方法有效處理了非均勻 RGB 光照的色度複雜性，而 IFBlend 並非針對此設計。

- **計算節省 15% 的情況下 AMBIENT6K 結果具有競爭力**：在白平衡對齊的 AMBIENT6K 基準上，RLN2-Lf 達到 PSNR 21.712 vs. IFBlend 的 21.443，具有更優的 SSIM（0.825 vs. 0.819）和 LPIPS（0.120 vs. 0.128），同時僅消耗 22.72G MACs 相比 IFBlend 的 26.01G。這表明物理基礎指導使方法既能改進對不同光照條件的魯棒性，又能提高計算效率。

- **極輕量級變體保持競爭力**：RLN2-S 以僅 3.95G MACs 達到 PSNR 19.992（≈比完整尺寸基線少 15 倍），優於顯著更大的方法如 GRL（2.16G MACs、PSNR 18.089）。這表明雙流分解策略本質上比擴展單流架構更有效，為部署受限場景提供實用替代方案。

- **消融研究驗證 HSV 指導和 CDFFA**：表 2 顯示完全移除融合/指導得到 PSNR 19.892，簡單 RGB 指導連接僅提供邊際改進（20.018）。關鍵跳躍來自 HSV 指導（20.149→20.216）特別是 CDFFA 注意力（20.523）。HSV 指導和 CDFFA 的聯合貢獻相比基線產生 +0.631 PSNR、+0.057 SSIM 和 -0.063 LPIPS，證明每個分量的必要性。

- **在真實世界和生成圖像上的一致魯棒性**：超越基準數據集，圖 8 顯示 RLN2-Lf 在真實產品攝影和 LSMI 數據集樣本上恢復合理的色彩，其中 IFBlend 產生明顯的色彩失真。圖 9 展示 ALN 作為神經圖像編輯（ICLight）的有效預處理步驟，RLN2 改進層分離並減少語義不連續性，表明超越受控工作室設置的實際適用性。

## 優勢

- **解決真實且未充分探索的問題**：論文確定了環境光照標準化研究中的關鍵缺口：缺乏用於處理空間變化彩色光源的數據集和方法。先前工作（ISTD+、AMBIENT6K）通過使用白平衡或均勻光照而過度簡化。動機清晰明確：真實場景涉及複雜多色光照（LED 燈、彩色濾光片、霓虹燈等），現有方法無法捕獲這一點。這是值得解決的問題，在攝影、視覺效果和圖像編輯中具有清晰的實際應用。

- **全面的基準測試和分析**：論文對在 CL3AN 上訓練的 11 種領先圖像復原方法提供廣泛的實證研究，提供關於失敗模式（紋理洩漏、色彩失真、不完整標準化）的診斷見解。這項基準測試貢獻對社區有價值，為未來工作提供參考點。比較包括多樣化的架構（CNN、Transformer、狀態空間模型）和先驗知識（RGB、頻域），實現公平的架構分析。

- **良好動機的技術設計與物理基礎**：RLN2 並非提出另一個黑箱架構，而是將其設計基於 Retinex 理論和明確的 HSV 色彩空間特性。V 編碼反射率而 H 和 S 編碼色度的洞見是直觀的且充分利用。CDFFA 模塊的設計（使用相似性到 HSV 導出的指導進行特徵加權）是創意的且比純粹學習注意力更可解釋，解決了先前頻域感知方法在隱式學習合適濾波器時的真實局限。

- **具有適當消融的穩健實驗驗證**：消融研究（表 2）系統地驗證了 HSV 指導和 CDFFA 的貢獻，與更簡單替代方案比較（連接、RGB 指導、L*a*b* 指導）。結果清晰顯示每個分量的必要性。包括多個 RLN2 變體（S、Sf、L、Lf）允許針對相似規模基線的公平複雜度-性能權衡分析，增強比較的透明度。

- **通過數據集和方法的實際影響**：CL3AN 數據集（105 個場景，4,535 個總樣本）是解決真實需求的寶貴社區資源。結合代碼和模型的發佈，論文既提供了評估未來方法的基準，也提供了強大的基線。該方法展示了強勁的性能-效率權衡：RLN2-Lf 超越計算成本高 63%-87% 的方法，同時 RLN2-S 對資源受限部署實用。

- **清晰的呈現和強大的實驗嚴謹性**：論文撰寫良好，具有信息豐富的可視化（圖 2-3 有效溝通問題；圖 6-9 令人信服地展示定性改進）。實驗設置徹底文檔化（表 1 指定所有捕獲參數）。培訓細節在補充材料中提供。比較包括已確立的基準（AMBIENT6K）和新數據集，驗證泛化性。

## 劣勢

- **數據集範圍有限且潛在領域差距**：儘管 CL3AN 代表寶貴貢獻，數據集完全在受控工作室環境中使用專業光照配置捕獲，引發對真實野外場景泛化的疑問。105 個場景（≈4,500 個總樣本）相比現代圖像復原數據集（ISTD+ 有數千圖像，通用 IR 數據集有數萬）規模有限。五個白色軟箱的環境參考設置可能無法忠實代表所有真實世界環境光照（例如混合自然-人工光照、直射陽光與彩色人工燈結合、高度鏡面反射）。論文缺乏關於該方法如何泛化到此受控設置外場景的分析，限制了關於實際部署的聲稱。

- **材料表徵不足和鏡面表面處理**：論文聲稱場景包括材料具有「變化的透明度、顏色和表面粗糙度」但未提供系統的材料清單或特性規格。這是問題所在因為反射率建模關鍵依賴材料特性（例如擴散 vs. 鏡面 vs. 次表面散射）。HSV 指導假設 V 通道主要編碼反射率，但對於具有強高光的鏡面材料，V 可能由鏡面反應而非內在反射率主導。論文未討論 RLN2 如何處理這些情況或是否在光澤或半透明材料上遇到困難。

- **Transformer 架構性能不佳的未解釋原因**：表 3 揭示 Transformer（SwinIR、Uformer、Restormer、GRL）顯著低於 CNN，但論文提供對原因的有限分析。作者指出 ALN「傾向於具有廣泛感受野的解決方案」但未調查這是內在架構限制、超參數/訓練問題還是數據集特性。這很重要因為 Transformer 已被證明對許多圖像復原任務有效。缺乏更深入的調查削弱了廣泛感受野必要性的聲稱，並冒著錯過重要洞見的風險。

- **頻域處理的邊際增益**：儘管 RLN2-Lf 優於 RLN2-L，頻域貢獻是有限的：CL3AN 上 +0.140 PSNR、+0.023 SSIM、-0.014 LPIPS，計算成本增加 0.27G MACs（總數的 1.2%）。在 AMBIENT6K 上，改進甚至更小（+0.159 PSNR、+0.004 SSIM、-0.003 LPIPS）。對於 1.2% 計算開銷在 LPIPS 和 SSIM 中產生如此邊際增益，成本-益處權衡值得商榷。論文未說明為何頻域特徵超越經驗結果是必要的；了解這是否針對該任務或 ALN 的一般特性將很有價值。

- **缺乏失敗情況和方法限制分析**：論文展示成功結果但提供關於失敗場景的最小討論。RLN2 何時失敗？存在模型類型、光照配置或遮擋模式導致方法崩潰嗎？圖 1 顯示成功結果，但未討論失敗情況。圖 9 顯示在 AI 生成圖像上良好結果，但邊界情況如何：非常高的鏡面性、極端色飽和、深陰影或複雜遮擋？未理解失敗模式，未來研究人員無法有效地基於此工作進行構建。

- **替代指導機制的探索有限**：消融研究僅針對 RGB 和 L*a*b* 比較 HSV 指導，但其他表示可能提供信息。例如，YCbCr（亮度-色度）用於許多圖像處理管道；YUV 和其他亮度-色度空間可提供類似益處。此外，未探索學習的指導（使用輕量級注意力模塊學習哪個指導最相關每個特徵）。論文 HSV 最優的聲稱基於針對兩個替代方案的比較，這是有限的證據。探索更多指導選項或基於學習的指導機制可增強貢獻。

## 研究方向

- **將 ALN 擴展到動態場景和視頻**：當前 RLN2 和 CL3AN 專注於靜態圖像。自然擴展是開發視頻 ALN 在處理動態場景變化、相機運動和時間光照變化引起的閃爍伪影時強制時間一致性。這涉及設計遞歸或基於注意力的機制在幀間傳播光照估計。核心貢獻可以是時間一致性損失，防止相鄰幀間不一致的標準化，同時允許光照條件真實變化時的平滑轉換。視頻編輯和流內容校正中的應用很重要。

- **用於域外魯棒性的生成精煉**：儘管圖 8 展示合理的域外性能，混合方法結合 RLN2 的效率與生成先驗可改進魯棒性。考慮兩階段管道：RLN2 提供快速初始標準化，然後輕量級擴散模型在原始輸入的 RLN2 輸出和色彩統計指導條件下精煉結果。這利用 RLN2 的可解釋性和效率同時獲得生成模型的魯棒性。關鍵創新將是設計條件機制保留 RLN2 內容同時僅修復色彩/光照伪影。

- **可學習指導和自適應色空間選擇**：與固定 HSV 指導不同，開發元學習框架其中網絡學習對於每個圖像區域或材料類型最信息的色彩空間。這可涉及以學習權重對多個色彩空間表示（HSV、YCbCr、YUV 等）進行門控，允許模型自適應選擇指導。另一方向是學習任務特定指導空間（例如特別優化用於將光照從反射率分離的表示）通過自監督前置任務如預測光照方向或材料反射率特性。

- **具有明確材料認知的材料感知 ALN**：擴展 RLN2 以在光照標準化旁聯合估計材料特性（反射率、粗糙度、鏡面性）。多任務框架可包括材料預測分支，具有明確損失鼓勵物理一致分解。培訓可利用物理渲染引擎（如 Mitsuba）生成具有已知材料和光照參數的合成地面真值。這將改進鏡面和次表面散射材料的處理，解決 HSV 指導方法中的當前弱點。

- **縮放到極端光照條件和真實世界捕獲**：開發新捕獲程序和數據集用於更極端場景：(1) 混合自然-人工光照，(2) 高度方向性日光結合彩色人工燈，(3) 反射和鏡面主導的場景。與專業攝影師或產品攝影師合作捕獲真實世界具有挑戰性的情況。這將驗證 CL3AN 基於工作室的洞見是否泛化並可能揭示超越當前方法所需架構創新的新挑戰。

- **ALN 的理論分析和可解釋性**：開發乘法分解模型下 ALN 問題的形式理論分析（方程 1）。在什麼條件下反射率-光照分解是可識別的？光源的什麼假設（數量、方向性、色彩）實現穩定估計？推導 RGB 充分性的充分條件（即 3 通道信息何時唯一確定 L 和 R）。互補地，通過所學 H 和 S 精煉模式的可視化和分析 CDFFA 對不同材料類型優先化哪些頻域分量來改進 RLN2 的可解釋性。

- **與下游任務和端到端學習的整合**：評估當在可變光照下測試的圖像上測試時，ALN 預處理是否改進下游任務性能（人臉識別、物體檢測、圖像分割）。開發端到端學習框架其中 ALN 是與任務特定頭聯合優化的可微預處理階段。這個實用方向將展示基於工作室的 ALN 改進是否轉化為真實世界益處，可能吸引工業採用和資金用於進一步數據集擴展。

</div>


