from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import base64
import io

from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


MODEL_DIR = Path("/home/zsc/Downloads/try_qwen3.5/Qwen3.5-2B")
IMAGE_PATH = Path("/home/zsc/d/genshin_qiasika.png")
OUTPUT_HTML = Path("/home/zsc/Downloads/try_qwen3.5/qwen35_image_token_curve.html")


@dataclass
class ImageTokenPoint:
    ratio: str
    ratio_h: int
    ratio_w: int
    target_long: int
    input_w: int
    input_h: int
    resized_w: int
    resized_h: int
    resized_pixels: int
    image_pad_tokens: int


def _build_size_targets(min_long: int = 256, max_long: int = 8192, step: int = 128) -> List[int]:
    return list(range(min_long, max_long + 1, step))


def _build_aspect_ratios() -> List[Tuple[str, int, int]]:
    return [
        ("1:1", 1, 1),
        ("4:3", 4, 3),
        ("16:9", 16, 9),
        ("9:16", 9, 16),
        ("3:2", 3, 2),
        ("2:3", 2, 3),
        ("21:9", 21, 9),
    ]


def _extract_pixel_bounds(image_processor) -> Dict[str, int]:
    size_cfg = getattr(image_processor, "size", {})
    if isinstance(size_cfg, int):
        min_pixels = getattr(image_processor, "min_pixels", 65536)
        max_pixels = getattr(image_processor, "max_pixels", 16777216)
    else:
        min_pixels = size_cfg.get("shortest_edge", size_cfg.get("min_pixels", 65536))
        max_pixels = size_cfg.get("longest_edge", size_cfg.get("max_pixels", 16777216))
    return {"min_pixels": int(min_pixels), "max_pixels": int(max_pixels)}


def _build_raw_size(target_long: int, ratio_h: int, ratio_w: int) -> Tuple[int, int]:
    if ratio_w >= ratio_h:
        input_w = target_long
        input_h = max(1, round(target_long * ratio_h / ratio_w))
    else:
        input_h = target_long
        input_w = max(1, round(target_long * ratio_w / ratio_h))
    return int(input_h), int(input_w)


def _fit_linear(x_values: List[int], y_values: List[int]) -> Tuple[float, float, float, float]:
    n = len(x_values)
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    sxx = sum((x - x_mean) ** 2 for x in x_values)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    if sxx == 0:
        return 0.0, y_mean, 0.0, 1.0
    a = sxy / sxx
    b = y_mean - a * x_mean
    y_pred = [a * x + b for x in x_values]
    mse = sum((y1 - y2) ** 2 for y1, y2 in zip(y_values, y_pred)) / n
    rmse = mse ** 0.5
    sst = sum((y - y_mean) ** 2 for y in y_values)
    r2 = 1.0 if sst == 0 else 1.0 - mse * n / sst
    return a, b, rmse, r2


def collect_points() -> List[ImageTokenPoint]:
    Image.open(IMAGE_PATH).convert("RGB")

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    image_processor = processor.image_processor
    patch_size = int(image_processor.patch_size)
    merge_size = int(image_processor.merge_size)
    size_cfg = _extract_pixel_bounds(image_processor)
    min_pixels = size_cfg["min_pixels"]
    max_pixels = size_cfg["max_pixels"]
    long_short_factor = patch_size * merge_size

    target_longs = _build_size_targets()
    aspect_ratios = _build_aspect_ratios()
    points: List[ImageTokenPoint] = []

    for ratio_name, ratio_h, ratio_w in aspect_ratios:
        for target_long in target_longs:
            input_h, input_w = _build_raw_size(target_long, ratio_h, ratio_w)
            resized_h, resized_w = smart_resize(input_h, input_w, long_short_factor, min_pixels, max_pixels)
            grid_h = resized_h // patch_size
            grid_w = resized_w // patch_size
            num_tokens = (grid_h * grid_w) // (merge_size * merge_size)
            points.append(
                ImageTokenPoint(
                    ratio=ratio_name,
                    ratio_h=ratio_h,
                    ratio_w=ratio_w,
                    target_long=target_long,
                    input_h=input_h,
                    input_w=input_w,
                    resized_w=resized_w,
                    resized_h=resized_h,
                    resized_pixels=resized_h * resized_w,
                    image_pad_tokens=num_tokens,
                )
            )

    return points


def make_plot_and_html(points: List[ImageTokenPoint]) -> None:
    by_ratio: Dict[str, List[ImageTokenPoint]] = defaultdict(list)
    for point in points:
        by_ratio[point.ratio].append(point)

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=140)
    markers = ["o", "s", "^", "D", "v", "<", ">"]
    for idx, ratio_name in enumerate(_build_aspect_ratios()):
        grouped = sorted(by_ratio[ratio_name[0]], key=lambda p: p.target_long)
        ax1.plot(
            [p.target_long for p in grouped],
            [p.image_pad_tokens for p in grouped],
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            label=f"ratio {ratio_name[0]}",
        )

    ax1.set_title("Qwen3.5-2B image token count with aspect ratio sweep", fontsize=13)
    ax1.set_xlabel("Target long edge (pixels)")
    ax1.set_ylabel("image_pad token count")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="upper left", ncol=2)

    first_ratio = _build_aspect_ratios()[0][0]
    reference = sorted(by_ratio[first_ratio], key=lambda p: p.target_long)
    ax2 = ax1.twinx()
    ax2.plot(
        [p.target_long for p in reference],
        [p.resized_pixels for p in reference],
        linestyle=":",
        linewidth=1,
        alpha=0.4,
    )
    ax2.set_ylabel("Resized pixels (1:1)")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    rows = [
        "<tr><th>ratio</th><th>target_long</th><th>input(h×w)</th><th>resized(h×w)</th>"
        "<th>resized_pixels</th><th>image_pad</th><th>start+pad+end</th></tr>"
    ]
    for point in sorted(points, key=lambda p: (p.ratio, p.target_long)):
        rows.append(
            f"<tr><td>{point.ratio}</td><td>{point.target_long}</td><td>{point.input_h}×{point.input_w}</td>"
            f"<td>{point.resized_h}×{point.resized_w}</td><td>{point.resized_pixels}</td>"
            f"<td>{point.image_pad_tokens}</td><td>{point.image_pad_tokens + 2}</td></tr>"
        )

    all_resized = [point.resized_pixels for point in points]
    all_tokens = [point.image_pad_tokens for point in points]
    a_resized, b_resized, rmse_resized, r2_resized = _fit_linear(all_resized, all_tokens)
    resized_fit_row = (
        f"<tr><td>resized 全量</td><td>{a_resized:.12f}</td><td>{b_resized:.8f}</td>"
        f"<td>{rmse_resized:.6f}</td><td>{r2_resized:.6f}</td><td>{len(points)}</td></tr>"
    )

    low_points = [point for point in points if point.image_pad_tokens <= 2000]
    low_resized = [point.resized_pixels for point in low_points]
    low_tokens = [point.image_pad_tokens for point in low_points]
    if low_points:
        a_low, b_low, rmse_low, r2_low = _fit_linear(low_resized, low_tokens)
        low_fit_row = (
            f"<tr><td>resized 低区间(T≤2000)</td><td>{a_low:.12f}</td><td>{b_low:.8f}</td>"
            f"<td>{rmse_low:.6f}</td><td>{r2_low:.6f}</td><td>{len(low_points)}</td></tr>"
        )
        low_raw_pixels = [point.input_h * point.input_w for point in low_points]
        a_lr, b_lr, rmse_lr, r2_lr = _fit_linear(low_raw_pixels, low_tokens)
        low_raw_fit_row = (
            f"<tr><td>raw 低区间(T≤2000)</td><td>{a_lr:.12f}</td><td>{b_lr:.6f}</td>"
            f"<td>{rmse_lr:.6f}</td><td>{r2_lr:.6f}</td><td>{len(low_points)}</td></tr>"
        )
    else:
        low_fit_row = (
            "<tr><td>resized 低区间(T≤2000)</td><td colspan=\"4\">无可用点</td><td>0</td></tr>"
        )
        low_raw_fit_row = (
            "<tr><td>raw 低区间(T≤2000)</td><td colspan=\"4\">无可用点</td><td>0</td></tr>"
        )

    ratio_parts: List[str] = []
    for ratio, grouped in sorted(by_ratio.items()):
        x = [p.input_h * p.input_w for p in sorted(grouped, key=lambda p: p.target_long)]
        y = [p.image_pad_tokens for p in sorted(grouped, key=lambda p: p.target_long)]
        a, b, rmse, r2 = _fit_linear(x, y)
        ratio_parts.append(
            f"<tr><td>{ratio}</td><td>{a:.12f}</td><td>{b:.3f}</td>"
            f"<td>{rmse:.3f}</td><td>{r2:.6f}</td><td>{len(grouped)}</td></tr>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Qwen3.5-2B image token count curve</title>
  <style>
    body {{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 24px; max-width: 1200px;}}
    h1 {{font-size: 1.4rem;}}
    table {{border-collapse: collapse; width: 100%; margin-top: 16px;}}
    th, td {{border: 1px solid #ccc; padding: 6px 8px; font-size: 0.85rem; text-align: right;}}
    th {{background: #f4f4f4;}}
    .mono {{font-family: ui-monospace, Menlo, monospace;}}
  </style>
</head>
<body>
  <h1>Qwen3.5-2B 图像输入等效 token 计数（长宽比探索）</h1>
  <p>等效 token 计算基于 <span class=\"mono\">smart_resize</span> 后尺寸和
  <span class=\"mono\">image_grid_thw.prod() / merge_size^2</span>，并比较多组目标长宽比。</p>
  <img src=\"data:image/png;base64,{image_b64}\" alt=\"token curve\" style=\"max-width: 100%; border: 1px solid #ddd;\" />
  <h2>拟合结果</h2>
  <p>说明：resized 拟合实际上是模型图像 token 的等价关系验证；raw 拟合仅用于粗估。</p>
  <table>
    <tr><th>范围</th><th>斜率 a</th><th>截距 b</th><th>RMSE</th><th>R²</th><th>点数</th></tr>
    {resized_fit_row}
    {low_fit_row}
    {low_raw_fit_row}
  </table>
  <details>
    <summary>raw 像素按长宽比分组的线性拟合</summary>
    <table>
      <tr><th>ratio</th><th>斜率 a</th><th>截距 b</th><th>RMSE</th><th>R²</th><th>点数</th></tr>
      {''.join(ratio_parts)}
    </table>
  </details>
  <p><b>结论公式（推荐）</b>：先将原图输入经过 <span class=\"mono\">smart_resize</span> 得到
    <span class=\"mono\">H', W'</span>，则图像视觉 token 近似为
    <span class=\"mono\">T_pad = ceil(H'/16) * ceil(W'/16) / 4</span>。
    在 Qwen3.5 路径中，<span class=\"mono\">H',W'</span> 通常是 32 的倍数，因此等价为
    <span class=\"mono\">T_pad = H' * W' / 1024</span>。最终加入视觉边界后为
    <span class=\"mono\">T_total = T_pad + 2</span>。</p>
  <p><b>粗略近似（按原始输入像素）</b>：若直接估算，可用
    <span class=\"mono\">P_in = H×W</span>，先做像素截断
    <span class=\"mono\">P_in' = clip(P_in, 65536, 16777216)</span>，再用
    <span class=\"mono\">T_pad ≈ P_in' / 1024</span>。</p>
  <table>
    {''.join(rows)}
  </table>
</body>
</html>"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    points = collect_points()
    make_plot_and_html(points)
    first_ratio = _build_aspect_ratios()[0][0]
    first = [p for p in points if p.ratio == first_ratio][0]
    print(f"Generated {len(points)} points to: {OUTPUT_HTML}")
    print(
        f"Example: ratio={first.ratio}, target_long={first.target_long}, "
        f"resized={first.resized_h}x{first.resized_w}, image_pad_tokens={first.image_pad_tokens}"
    )


if __name__ == "__main__":
    main()
