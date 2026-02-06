---
title: "When the brightest is not the best: illuminant estimation from the geometry of specular highlights"
date: 2026-01-24
authors:
  - "Takuma Morimoto"
  - "Robert J. Lee"
  - "Hannah E. Smithson"
  - "Takuma Morimoto"
  - "Robert J. Lee"
  - "Hannah E. Smithson"
arxiv_url: "https://www.biorxiv.org/content/biorxiv/early/2026/01/24/2026.01.22.700600"
pdf_url: "https://www.biorxiv.org/content/10.64898/2026.01.22.700600v1.full.pdf"
one_line_summary: " Human observers use geometric regularities of specular highlights rather than simply the "brightest is white" heuristic to achieve color constancy, performing better when highlights appear on dark textured regions."
one_line_summary_zh: " 人類觀察者利用鏡面高光的幾何規律性而非簡單的「最亮即白」啟發式來實現顏色恆常性，當高光出現在暗色紋理區域時表現更佳。"
topics: ["Auto White Balance"]
tags: []
---

Let me generate the structured summary based on the paper content provided:

<div class="lang-en">

## Key Contributions

- **Challenges the "brightest is white" heuristic** - demonstrates that human observers do not simply rely on the brightest element to estimate illuminant color, but instead utilize geometric regularities of specular highlights
- **Novel experimental paradigm** - introduces computer-rendered spheres with high-contrast textures to systematically vary the reliability of "brightest element" vs "highlight geometry" cues
- **Computational modeling** - implements ideal observer models to compare "brightest element" and "center of specular highlight" strategies, showing human performance exceeds predictions based on brightness alone
- **Phase scrambling manipulation** - demonstrates the critical role of spatial structure in specular highlight detection by showing performance drops significantly when geometric information is disrupted

## Core Insights

- Human observers achieve **better color constancy** when specular highlights appear on dark regions of textured surfaces (spatially-separated condition), even though the "brightest element" heuristic predicts worse performance
- Performance improved significantly with increasing specularity levels (0%, 2%, 4%), rising from **chance level (d' ≈ 0) for matte surfaces to d' > 1.5** for mid-specularity in spatially-separated conditions
- When specular geometries were disrupted via **phase scrambling**, observer performance dropped dramatically, confirming that structured spatial information is essential for illuminant estimation
- The visual system appears to identify and extract chromatic information from **highlight centers** based on geometric patterns rather than simply selecting the brightest pixels

## Key Data & Results

| Condition | Specularity | Spatially-Aligned d' | Spatially-Separated d' |
|-----------|-------------|---------------------|----------------------|
| Sphere | 0.00 | ~0.1 | ~0.1 |
| Sphere | 0.02 | ~0.7 | ~0.9 |
| Sphere | 0.04 | ~1.0 | ~1.5 |
| Phase-scrambled | 0.02 | ~0.5 | ~0.5 |
| Phase-scrambled | 0.04 | ~0.6 | ~0.6 |

- **Significant main effects** found for spatial alignment (F(1,5) = 33.0, η² = 0.868, p = 0.00225) and specularity (F(2,10) = 43.2, η² = 0.896, p = 0.000012)
- In spatially-separated conditions at specularity 0.04, observers achieved **d' ≈ 1.5**, substantially outperforming the brightest element model
- Phase scrambling eliminated the advantage of spatially-separated conditions, with no significant effect of spatial alignment (F(1,5) = 0.36, p = 0.575)
- Response bias (C) remained close to **zero across all conditions**, indicating unbiased classification between illuminant and reflectance changes

## Strengths

- **Rigorous experimental design** with systematic manipulation of multiple factors (specularity, spatial alignment, geometric structure) across 2,000 trials per observer
- **Physically-based rendering** using PBRT ensures realistic light-material interactions following the Ward reflectance model
- **Strong statistical analysis** with appropriate repeated-measures ANOVA, Bonferroni corrections, and effect size reporting (η²)
- **Computational modeling** provides clear mechanistic comparisons between competing hypotheses about visual strategies
- **High inter-observer consistency** across six observers, strengthening the generalizability of findings
- **Clever stimulus design** matching chromatic changes in diffuse components between illuminant and reflectance change conditions, forcing reliance on specular information

## Weaknesses

- **Limited to single spherical objects** with uniform spectral reflectance - does not test generalization to complex multi-object scenes or objects with multiple surface reflectances
- **Simplified lighting environment** with only three point lights may not capture the complexity of natural illumination
- **Phase scrambling** destroys all spatial structure, making it difficult to isolate which specific geometric features are critical (e.g., gradient smoothness, highlight shape)
- **Computational models exceed human performance** (especially "center of specular highlight" model), suggesting missing factors like viewing time constraints or spatial uncertainty
- **No eye-tracking data** to verify whether observers actually fixate on specular regions or use peripheral vision
- **Limited specularity range** (0-4%) may not represent highly glossy materials commonly encountered in real environments

## Potential Improvements

- **Expand to multi-object scenes** with varying reflectances to test whether cone-ratio methods remain effective with heterogeneous surface properties
- **Vary lighting complexity** including extended light sources, multiple illuminant colors, and natural illumination maps to assess ecological validity
- **Systematic geometric manipulations** (e.g., blur levels, shape distortions) instead of complete phase scrambling to identify specific spatial features used by the visual system
- **Incorporate temporal dynamics** by varying animation duration and number of frames to understand integration time for highlight detection
- **Add noise and uncertainty** to computational models to better match human performance, including factors like fixation patterns, peripheral pooling, and estimation variability
- **Test other material properties** such as subsurface scattering, transparency, or anisotropic reflections to understand limitations of specular-based strategies
- **Include real-world validation** with photographs or physical objects to verify findings transfer beyond computer graphics

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- **挑戰「最亮即白」啟發式方法** - 證明人類觀察者不僅僅依賴最亮元素來估計光源顏色，而是利用鏡面反射高光的幾何規律性
- **新穎的實驗範式** - 引入具有高對比度紋理的電腦渲染球體，系統性地改變「最亮元素」與「高光幾何」線索的可靠性
- **計算模型** - 實作理想觀察者模型來比較「最亮元素」和「鏡面高光中心」策略，顯示人類表現超越僅基於亮度的預測
- **相位擾亂操作** - 透過顯示幾何資訊被破壞時表現顯著下降，證明空間結構在鏡面高光檢測中的關鍵作用

## 核心洞見

- 當鏡面高光出現在紋理表面的暗區（空間分離條件）時，人類觀察者實現了**更好的顏色恆常性**，儘管「最亮元素」啟發式預測表現會更差
- 隨著鏡面反射度增加（0%、2%、4%），表現顯著改善，從**霧面表面的機率水準（d' ≈ 0）上升到空間分離條件下中等鏡面反射度的 d' > 1.5**
- 當透過**相位擾亂**破壞鏡面幾何時，觀察者表現急劇下降，證實結構化空間資訊對於光源估計至關重要
- 視覺系統似乎根據幾何模式識別並從**高光中心**提取色度資訊，而不是簡單地選擇最亮的像素

## 關鍵數據與結果

| 條件 | 鏡面反射度 | 空間對齊 d' | 空間分離 d' |
|------|-----------|------------|------------|
| 球體 | 0.00 | ~0.1 | ~0.1 |
| 球體 | 0.02 | ~0.7 | ~0.9 |
| 球體 | 0.04 | ~1.0 | ~1.5 |
| 相位擾亂 | 0.02 | ~0.5 | ~0.5 |
| 相位擾亂 | 0.04 | ~0.6 | ~0.6 |

- 空間對齊（F(1,5) = 33.0, η² = 0.868, p = 0.00225）和鏡面反射度（F(2,10) = 43.2, η² = 0.896, p = 0.000012）發現**顯著主效應**
- 在鏡面反射度 0.04 的空間分離條件下，觀察者達到 **d' ≈ 1.5**，大幅超越最亮元素模型
- 相位擾亂消除了空間分離條件的優勢，空間對齊無顯著效應（F(1,5) = 0.36, p = 0.575）
- 反應偏差（C）在**所有條件下均接近零**，顯示光源與反射率變化分類無偏差

## 優勢

- **嚴謹的實驗設計**，系統性操作多個因素（鏡面反射度、空間對齊、幾何結構），每位觀察者完成 2,000 次試驗
- **基於物理的渲染**使用 PBRT 確保遵循 Ward 反射模型的真實光-材料交互作用
- **強大的統計分析**，包含適當的重複測量 ANOVA、Bonferroni 校正和效應量報告（η²）
- **計算模型**提供競爭假說之間視覺策略的清晰機制比較
- **高度觀察者間一致性**，六位觀察者結果一致，增強發現的普遍性
- **巧妙的刺激設計**，在光源和反射率變化條件之間匹配漫反射成分的色度變化，強制依賴鏡面資訊

## 劣勢

- **僅限於單一球形物體**，具有均勻光譜反射率 - 未測試對複雜多物體場景或具有多種表面反射率物體的泛化能力
- **簡化的照明環境**，僅有三個點光源可能無法捕捉自然照明的複雜性
- **相位擾亂**破壞所有空間結構，難以分離哪些特定幾何特徵是關鍵（例如梯度平滑度、高光形狀）
- **計算模型超越人類表現**（特別是「鏡面高光中心」模型），暗示缺失觀看時間限制或空間不確定性等因素
- **無眼動追蹤數據**來驗證觀察者是否實際注視鏡面區域或使用周邊視覺
- **有限的鏡面反射度範圍**（0-4%）可能無法代表真實環境中常見的高光澤材料

## 可改進方向

- **擴展至多物體場景**，具有不同反射率，測試錐細胞比率方法在異質表面屬性下是否仍然有效
- **改變照明複雜度**，包括擴展光源、多種光源顏色和自然照明圖，以評估生態效度
- **系統性幾何操作**（例如模糊程度、形狀扭曲）而非完全相位擾亂，以識別視覺系統使用的特定空間特徵
- **納入時間動態**，透過改變動畫持續時間和幀數來理解高光檢測的整合時間
- **在計算模型中加入噪聲和不確定性**，以更好地匹配人類表現，包括注視模式、周邊池化和估計變異性等因素
- **測試其他材料屬性**，如次表面散射、透明度或各向異性反射，以了解基於鏡面反射策略的限制
- **包含真實世界驗證**，使用照片或物理物體來驗證發現是否能轉移到電腦圖形之外

</div>
