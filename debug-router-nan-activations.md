# [OPEN] debug-router-nan-activations

## Symptom
- Validation forward hits `MoERouterError: Router input contains NaN/Inf values [EfficientSpatialRouter]`.

## Hypotheses
1. 특정 backbone/neck layer 输出非有限激活（finite 权重但 activation overflow），并在验证路径更容易触发。
2. 某个 batch/sample 在预处理后包含 NaN/Inf（图像张量或 label 派生张量），导致后续激活污染。
3. eval 路径与 train 路径存在数值差异（BN/Dropout/GC/AMP/autocast），使得验证更容易产生 NaN。
4. MoE block 内部的 gating/normalization 在 eval 形态下产生除零或 softmax 溢出，导致 router 输入非有限。
5. 训练过程中偶发溢出导致 buffer/running stats 异常，随后在 eval 使用 running stats 时放大为 NaN。

## Evidence Needed
- 第一个产生非有限输出的模块 index/name（来自 `_predict_once`）。
- Router 输入张量的统计（dtype/shape/min/max/非有限占比）。
- 是否仅发生在验证（self.training=False）路径。

## Next
- Add minimal instrumentation (only triggers when nonfinite is detected).
