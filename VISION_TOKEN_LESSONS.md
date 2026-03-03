# Qwen3.5-2B 视觉 Token 计数：经验教训总结

本笔记记录一次围绕 `Qwen3.5-2B/` 的图像输入 token 计数实验的关键结论、坑点和可复用公式。

## 1. 模型能力边界先澄清

- `Qwen3.5-2B` 在这里使用的是“图像/视频输入 + 文本输出”的 VLM 路径。
- 它不是图像/视频生成模型，也不是语音生成模型；因此“语音/图像/视频生成能力”这个方向需要换模型族或换任务栈。

## 2. 图像输入是否 tokenized（离散/连续）

- 图像输入在该路线下是 **离散 token**，不是连续 latent。
- Chat template 里图像通常表示为：`<|vision_start|><|image_pad|><|vision_end|>`。
- 真正计入上下文长度的是 `<|image_pad|>` 被 processor 展开后的个数；`vision_start/end` 只额外贡献 2 个 token。

## 3. “等效图像 token 数”最准的定义

推荐把“等效图像 token 数”定义为：

- `T_pad`：processor 展开 `<|image_pad|>` 后的数量（也就是视觉 token）。
- `T_total = T_pad + 2`：再加上 `vision_start` 与 `vision_end` 的开销（若你把它们也算在视觉段里）。

经验：不要只看“输入图像分辨率”，一定要以 processor 的预处理后尺寸/网格为准。

## 4. 关键公式（比拟合更可靠）

设 `H, W` 为输入的高宽（可以来自任意缩放版本/任意长宽比）。

1) 先按模型预处理得到 `H', W'`：

- `H', W' = smart_resize(H, W, factor=patch_size*merge_size, min_pixels, max_pixels)`
- 在本实验中：`patch_size=16`，`merge_size=2`，因此 `factor=32`。

2) token 计数等价关系（核心）：

- `grid_h = H'/patch_size`
- `grid_w = W'/patch_size`
- `T_pad = (grid_h * grid_w) / (merge_size^2)`

因为 `smart_resize` 后 `H',W'` 通常是 `factor=32` 的倍数，上式可进一步化简为：

- `T_pad = (H' * W') / (patch_size^2 * merge_size^2) = (H' * W') / 1024`
- `T_total = T_pad + 2`

经验：对 `resized_pixels = H'*W'` 做“线性拟合”得到 `R²=1` 是正常的，因为本质是恒等式验证；应当把它写成“等价公式/identity check”，不要包装成经验回归。

## 5. smart_resize 带来的“台阶/平台”现象

`smart_resize` 会同时引入：

- 像素夹紧：`min_pixels` 与 `max_pixels`（小图被放大，大图被缩小）。
- 尺寸取整：高宽会被调整到 `factor=32` 的倍数。
- 长宽比约束：同一像素预算下，不同长宽比会落到不同的 `(H',W')`，导致 token 曲线分叉。

结果：`T_pad` 对“原始输入像素 `H*W`”不是严格线性，而是分段 + 量化台阶。

## 6. 粗估公式怎么写才不骗人

如果你只想快速估算而不复现完整 `smart_resize`：

- `P_in = H*W`
- `P_in' = clip(P_in, 65536, 16777216)`（本实验读到的上下界；以实际 processor 配置为准）
- `T_pad ≈ P_in' / 1024`

经验：这是“粗估”，误差主要来自长宽比约束和 factor 量化，尤其在边界夹紧附近更明显。

## 7. “拟合”相关的经验

- 对 `P_resized = H'*W'` 拟合没有信息增量，应该直接报告等价公式 `T_pad = P_resized/1024`。
- 真正可能有用的拟合是 `P_raw = H*W -> T_pad` 的粗估，但建议：
- 只在未触发 `min_pixels/max_pixels` 的中间区间拟合，或者分段拟合。
- 或者只报告低 token 区（例如 `T<=2000`）的 pooled 拟合，并明确它是“跨长宽比混合”的统计近似。

## 8. 工程实现上的坑点

- Transformers 版本要够新：旧版本可能没有对应的 processor/fast image processor，导致行为不一致。
- 计算 token 最省事的方式不是“真的把图片喂进去 tokenization”，而是复用同一套 `smart_resize` + 网格公式，速度快且更可控。
- 生成 HTML 时，把 plot 以 base64 嵌入可以让产物单文件可分享。
- Python 的 f-string + HTML 很容易踩转义/花括号问题；尽量避免在 f-string 里出现无意义的 `{}`。

## 9. 本实验产物与复现

- 脚本：`/home/zsc/Downloads/try_qwen3.5/qwen35_image_token_curve.py`
- 输出：`/home/zsc/Downloads/try_qwen3.5/qwen35_image_token_curve.html`
- 复现命令：

```bash
/home/zsc/Downloads/try_qwen3.5/.venv/bin/python /home/zsc/Downloads/try_qwen3.5/qwen35_image_token_curve.py
```

