---
title: 英伟达dam实测
author: Miao HanCheng
date: 2025-04-28 16:00:00
lang: zh-CN
summary: "简要记录英伟达 DAM 模型的本地测试结果，以及输入图片后的描述效果。"
featured: false
cover:
tags:
  - python
  - vision
  - captioning
categories:
  - [AI]

---

英伟达最新发布的[Describe Anything: Detailed Localized Image and Video Captioning](https://www.arxiv.org/abs/2504.16072)（DAM），展示文案上效果非常好，下载了一下模型进行测试，效果确实不错

### 输入图片（要求描述整个图片）

<img src="https://github.com/miaohancheng/picx-images-hosting/raw/master/pics/DAM_test.73u5gc223d.jpg" style="zoom: 25%;" />



### 输出结果

```json
The sky is a clear, vibrant blue with a few scattered, fluffy white clouds. The clouds are primarily concentrated towards the left side of the image, with one larger cloud near the top left corner and smaller clouds dispersed around it. The right side of the sky is mostly clear with a few smaller clouds.
```
