# Qwen3.5-2B 视觉 Token 计数：经验教训总结

本项目从零开始验证了 Qwen3.5-2B（本地 `Qwen3.5-2B/`）对图像输入的“等效 token”计数方式，并用 `~/d/genshin_qiasika.png` 相关的分辨率/长宽比扫描生成了曲线 HTML。

## 1. 先搞清模型能力边界

- `Qwen3.5-2B/` 这类 VL 模型的主路径是：图像/视频输入 + 文本输出（理解与生成文本）。
- 它不是“图像生成/视频生成/语音生成”模型（不会像扩散模型那样直接生成图像/视频，也不会做 TTS/语音合成）。

## 2. 图像输入是离散 token，不是连续方案

- 在 Transformers 的实现里，图像会被预处理成视觉网格，然后用离散的占位 token（典型是 `<|image_pad|>`）在文本序列中展开。
- 因此“图像输入是否 tokenize”的答案是：会，以离散视觉 token 的形式进入序列（随后映射成 embedding）。

## 3. `image_pad` 数就是你要的“等效 token”

- 对单张图像，处理器会给出 `image_grid_thw`（T,H,W）。
- 有效图像 token 计数遵循：`num_image_tokens = prod(image_grid_thw) / merge_size^2`。
- 对 Qwen3.5-2B 这一套配置（我们实测）：`patch_size=16`、`merge_size=2`，所以每个 token 覆盖 `32x32` 像素。
- 视觉边界 token 开销是常数：每张图像通常还有 `vision_start` 与 `vision_end` 各 1 个（总共 +2）。

一次 sanity check（用真实图片）：

- `~/d/genshin_qiasika.png` 原图 `1920x1080` 经处理后 `image_grid_thw = [1, 68, 120]`。
- `image_pad` 数为 `1*68*120/4 = 2040`。
- 若模板为 `<|vision_start|><|image_pad|><|vision_end|>`，总视觉相关 token 为 `2040 + 2 = 2042`（再加上你文本的 token）。

## 4. `smart_resize` 才是曲线形状的真正原因

你以为在扫“输入分辨率”，实际在扫：

- `H',W' = smart_resize(H,W, factor=patch_size*merge_size, min_pixels, max_pixels)`
- 其中 `factor = 16*2 = 32`，所以 `H',W'` 会被对齐到 32 的倍数。
- 同时像素数会被夹紧到 `[min_pixels, max_pixels]` 的范围（本机配置实测为 `65536` 到 `16777216`）。

这会造成两类现象：

- 小图 token 不会无限变小：会被最小像素限制“抬高”。
- 大图 token 不会无限变大：会被最大像素限制“压平”，曲线会出现高端平台。

## 5. 更简洁且可迁移的公式（推荐写法）

推荐把“估算 token”拆成两步：

1. 先算模型端有效尺寸：`(H',W') = smart_resize(H,W, factor=32, min_pixels=65536, max_pixels=16777216)`
2. 再算 token（单图）：`T_pad = H' * W' / 1024`，`T_total = T_pad + 2`

等价的“旧版风格”写法（更贴近 patch/token 的直觉）：

- `T_pad = ceil(H'/16) * ceil(W'/16) / 4`

说明：

- 之所以能写成 `H'*W'/1024`，是因为这里 `H',W'` 通常是 32 的倍数，且 `merge_size=2`。
- 如果换模型/换配置，核心仍是：`T_pad = (H'/patch_size) * (W'/patch_size) / merge_size^2`（前提是 `H',W'` 已对齐到 patch 网格）。

## 6. “拟合”这件事：哪些是恒等式，哪些才是经验模型

关键教训：

- 用 `resized_pixels` 去“拟合” `image_pad_tokens` 得到的线性关系不是经验回归，而是恒等式验证。
- 因为你已经用同一套离散规则从 `smart_resize` 结果计算出了 token，`T = resized_pixels/1024` 必然严格成立。

真正有经验意义的拟合是：

- 直接用 `raw_pixels = H*W` 去拟合 `T`。
- 但它只能当粗估：因为 `smart_resize` 的像素夹紧、factor 对齐、以及长宽比都会引入分段/台阶效应。

本项目里我们对 `T<=2000` 做了 pooled 拟合（混合多种 ratio），得到一个可用的粗估线：

- `T ≈ a * raw_pixels + b`

用途：

- 快速预算 token 上限，或为采样/缩放策略提供直觉。
- 不适合作为“可跨配置迁移的通用公式”。

## 7. 工程落地建议（复现与复用）

- 直接跑脚本重生成 HTML：`/home/zsc/Downloads/try_qwen3.5/.venv/bin/python /home/zsc/Downloads/try_qwen3.5/qwen35_image_token_curve.py`
- 输出：`/home/zsc/Downloads/try_qwen3.5/qwen35_image_token_curve.html`
- 若你要迁移到别的 Qwen VL 版本，优先做三件事：
- 读 `processor.image_processor` 拿到 `patch_size`、`merge_size`、`min_pixels`、`max_pixels`。
- 用同一个库里的 `smart_resize` 做尺寸对齐和像素夹紧。
- 用公式 `T_pad = (H'/patch_size)*(W'/patch_size)/merge_size^2` 算“等效视觉 token”，再 +2（若模板含 start/end）。

## 8. 小坑记录（避免重复踩）

- Transformers 版本要足够新才能加载对应的 Qwen3 VL Processor；版本太老会缺类或行为不一致。
- HTML 的 `f"""..."""` 大字符串里要谨慎处理转义与 `{}`，否则容易打出 `SyntaxError`。
- “拟合结果”建议明确标注：哪些是 identity check，哪些是 heuristic fit，不然读者会误解结论可信度来源。

