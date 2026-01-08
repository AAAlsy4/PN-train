# PN-train
通过模式神经元引导的训练提升时空预测模型在特殊模式（如节假日）下的性能。PN-Train的核心思想是识别并微调模型中对特定模式敏感的神经元，从而在不损害整体性能的情况下提升特定模式下的预测精度。

## 原始PN-Train方法包含三个主要阶段：
1.  Detection通过特定检测函数识别与节假日模式相关的神经元
2.  Verification验证检测到的神经元对模式的重要性
3.  Finetune仅微调检测到的模式神经元

## 改进方向：
1. 优化节假日样本的定义标准
2. 基于时空特征训练节假日/常规日分类器
3. 更有效的微调-重训练策略

## 核心参数范围：
1. select_ratio: 0.5，0.7
2. finetune_sample_num: 10，30
3. detect_sample_num: 30，50
4. finetune_learning_rate: 0.0001，0.002

https://github.com/cwang-nus/PN-Train
