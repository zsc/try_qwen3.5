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

## 10. Image token 的生成与“点位”（序列位置与 MRoPE）

这一节回答两个问题：

- 图像 token（`<|image_pad|>` 展开后的视觉 token）是怎么来的？
- 这些 token 在输入序列里怎么插入？在位置编码里怎么“定位”（点位）？

### 10.1 从图像到 `image_pad`（视觉 token）

- 预处理：输入图 `(H,W)` 先走 `smart_resize` 得到有效尺寸 `(H',W')`。
- 其中会发生：像素夹紧到 `[min_pixels,max_pixels]`、并把高宽对齐到 `factor = patch_size*merge_size` 的倍数（本实验为 `16*2=32`）。
- Patch 网格：`grid_h = H'/patch_size`，`grid_w = W'/patch_size`。processor 返回的 `image_grid_thw` 本质上就是 `[T, grid_h, grid_w]`（单图通常 `T=1`）。
- 空间 merge：视觉编码器先做 `16×16` patch embedding，然后把相邻 `merge_size×merge_size`（这里 `2×2`）patch token 合并成 1 个 LLM 视觉 token。
- 视觉 token 数（`image_pad` 展开数量）：
- `T_pad = (T*grid_h*grid_w)/(merge_size^2)`（单图 `T=1`）。
- 在本实验配置下等价为：`T_pad = H'*W'/1024`（因为有效步长是 `32×32`）。
- 若把边界也算进去：`T_total = T_pad + 2`（`vision_start/end` 各 1 个）。

### 10.2 序列“点位”：这些 token 在 input_ids 里怎么排

- Chat template 会把每张图放在 `vision_start`/`vision_end` 之间：`<|vision_start|><|image_pad|><|vision_end|>`。
- processor 在编码时把那个单个 `<|image_pad|>` 展开成 `T_pad` 个 `image_token_id`，因此在 `input_ids` 中图像是一段连续区间。

### 10.3 位置编码“点位”：MRoPE 的 3D position_ids

Qwen3-VL 路线下的 RoPE 位置编码不是单路 1D，而是 3 路（可理解为对 `(t,h,w)` 三个轴分别做 rotary）：

- 文本 token：三路 position 都是同一个递增的 1D 序号（等价于“在对角线上走”）。
- 视觉 token：为视觉块分配 `(t_index,h_index,w_index)` 三个坐标。
- 对图像：`t_index` 恒定（通常为 0），`h_index∈[0,llm_h)`、`w_index∈[0,llm_w)`，按 `w` 最快、再 `h` 的顺序展平。
- 对视频：在当前实现里 LLM 侧 `llm_grid_t` 也被视为 1（注释里明确写了 temporal 信息用 timestamps 编码），因此这里的 `t_index` 同样恒定，主要靠 text/timestamp token 表达时间。

### 10.4 token 索引如何对应到图像格子/区域

设 LLM 侧网格宽为 `llm_w = grid_w/merge_size`，视觉 token 在该图像块内的索引为 `k = 0..T_pad-1`：

- `w = k % llm_w`
- `h = k // llm_w`

在本实验配置（`patch_size=16, merge_size=2`）下，它对应 `smart_resize` 后图像上的一个 `32×32` 区域：

- `x ∈ [w*32,(w+1)*32)`, `y ∈ [h*32,(h+1)*32)`

### 10.5 一个可对照的例子

以 `1920×1080` 为例：

- processor 返回 `image_grid_thw = [1, 68, 120]`，对应 `H'=68*16=1088`、`W'=120*16=1920`。
- LLM 侧网格：`llm_h=68/2=34`，`llm_w=120/2=60`。
- `T_pad = 34*60 = 2040`，因此 `T_total = 2040 + 2`（如果把 `vision_start/end` 也计入视觉段总 token）。
