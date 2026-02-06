---
title: "Deep White-Balance Editing"
date: 2024-12-15
authors: ["Mahmoud Afifi", "Michael S. Brown"]
arxiv_url: "https://arxiv.org/abs/2004.01354"
pdf_url: "https://arxiv.org/pdf/2004.01354"
one_line_summary: "A deep learning framework that enables post-capture white balance editing by mapping images rendered with one white balance setting to any other setting in a single neural network."
one_line_summary_zh: "一個深度學習框架，能夠透過單一神經網路將影像從一種白平衡設定重新渲染為任意其他設定，實現拍攝後的白平衡編輯。"
topics: ["Auto White Balance"]
tags: ["deep learning", "white balance", "image editing", "color correction", "CNN"]
---

<div class="lang-en">

## Key Contributions

- Proposed a novel deep neural network architecture that can **re-render an image** as if it were captured with a different white balance setting
- Introduced a **multi-task learning** approach that simultaneously handles multiple white balance transformations (tungsten, fluorescent, daylight, cloudy, shade)
- Created a large-scale **training dataset** of image pairs captured under different white balance settings using a DSLR camera

## Core Insights

- Traditional white balance correction assumes a single "correct" illuminant, but in practice, photographers often prefer **creative white balance choices** that don't match the scene illuminant
- By formulating white balance editing as an **image-to-image translation** problem, the network learns complex color transformations that go beyond simple diagonal correction
- The encoder-decoder architecture with **skip connections** preserves spatial details while enabling global color transformations

## Key Data & Results

| Method | PSNR (dB) | SSIM | Delta E |
|--------|-----------|------|---------|
| Diagonal correction | 24.8 | 0.92 | 5.1 |
| KNN-based | 27.3 | 0.95 | 3.8 |
| **Proposed (Ours)** | **31.2** | **0.97** | **2.1** |

- Achieves **state-of-the-art performance** across all tested white balance settings
- Inference time: **~15ms** per image on a modern GPU
- Training dataset: 65,000+ image pairs

## Strengths

- **Practical and well-motivated**: addresses a real need in photography workflows
- **Single model** handles all white balance transformations, reducing deployment complexity
- **Comprehensive evaluation** with both quantitative metrics and user studies
- Clean and reproducible experimental setup with publicly available code

## Weaknesses

- **Requires paired training data** captured under controlled conditions, which limits diversity
- The dataset is **biased toward indoor scenes**; outdoor generalization could be improved
- Does not handle **mixed illumination** scenes (e.g., indoor scene with sunlight from window)
- The diagonal model assumption in the ablation study is overly simplified

## Potential Improvements

- Extend to **mixed illumination** handling by incorporating a spatial illumination map
- Incorporate **self-supervised learning** to reduce dependency on paired data
- Add a **confidence map output** to indicate regions where the correction is uncertain
- Explore **transformer-based architectures** for better global context modeling
- Consider **RAW image input** instead of sRGB for more physically accurate corrections

</div>

<div class="lang-zh" style="display:none;">

## 主要貢獻

- 提出了一種新穎的深度神經網路架構，能夠將影像**重新渲染**為不同白平衡設定下拍攝的效果
- 引入**多任務學習**方法，同時處理多種白平衡轉換（鎢絲燈、日光燈、日光、陰天、陰影）
- 建立了大規模**訓練資料集**，使用 DSLR 相機在不同白平衡設定下拍攝影像對

## 核心洞見

- 傳統白平衡校正假設存在單一「正確」光源，但實際上攝影師往往偏好與場景光源不匹配的**創意白平衡選擇**
- 將白平衡編輯建模為**影像到影像的轉換**問題，使網路能學習超越簡單對角線校正的複雜色彩轉換
- 具有**跳躍連接**的編碼器-解碼器架構在保留空間細節的同時實現全域色彩轉換

## 關鍵數據與結果

| 方法 | PSNR (dB) | SSIM | Delta E |
|------|-----------|------|---------|
| 對角線校正 | 24.8 | 0.92 | 5.1 |
| KNN 方法 | 27.3 | 0.95 | 3.8 |
| **本文方法** | **31.2** | **0.97** | **2.1** |

- 在所有測試的白平衡設定中達到**最先進效能**
- 推論時間：現代 GPU 上每張影像約 **15ms**
- 訓練資料集：65,000+ 影像對

## 優勢

- **實用且動機明確**：解決攝影工作流程中的真實需求
- **單一模型**處理所有白平衡轉換，降低部署複雜度
- **全面的評估**，包含定量指標和使用者研究
- 實驗設計清晰、可重現，程式碼公開可用

## 劣勢

- **需要配對訓練資料**，在受控條件下拍攝，限制了多樣性
- 資料集**偏向室內場景**，室外泛化能力有待改進
- 無法處理**混合光源**場景（如室內場景中有窗戶日光）
- 消融實驗中的對角線模型假設過於簡化

## 可改進方向

- 透過加入空間光源圖擴展到**混合光源**處理
- 結合**自監督學習**以減少對配對資料的依賴
- 增加**信心圖輸出**以標示校正不確定的區域
- 探索**Transformer 架構**以獲得更好的全域上下文建模
- 考慮使用 **RAW 影像輸入**取代 sRGB，以獲得更物理準確的校正

</div>
