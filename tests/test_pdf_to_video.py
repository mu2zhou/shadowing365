import os
import sys
import yaml
from pathlib import Path
# Ensure repository root is importable for local test run
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_to_video import ShadowingVideoGenerator
from PIL import Image, ImageDraw


def make_config(tmp_path: Path) -> str:
    cfg = {'output_dir': str(tmp_path)}
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
    return str(cfg_path)


def test_clean_text_hyphenation(tmp_path):
    cfg_path = make_config(tmp_path)
    gen = ShadowingVideoGenerator(cfg_path)

    raw = "This is a ha-\nving test. Also a soft\u00ADhyphen test."
    cleaned = gen._clean_text(raw)

    # Soft hyphen removed
    assert '\u00AD' not in cleaned
    # Broken word joined
    assert 'having' in cleaned
    # No lingering broken pattern
    assert 'ha- ' not in cleaned


def test_layout_non_overlap(tmp_path):
    cfg_path = make_config(tmp_path)
    gen = ShadowingVideoGenerator(cfg_path)

    resolution = (800, 600)
    img = Image.new('RGB', resolution)
    draw = ImageDraw.Draw(img)

    long_en = " ".join(["extraordinary"] * 40)
    long_cn = "这是一个非常长的中文句子。" * 10
    item = {'source': long_en, 'target': long_cn}

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    layout = gen._compute_text_layout(item, draw, resolution, font_path, 0, 70, 45)

    src_pos = layout['src_pos']
    tgt_pos = layout['tgt_pos']
    src_bbox = layout['src_bbox']
    tgt_bbox = layout['tgt_bbox']

    src_h = src_bbox[3] - src_bbox[1]
    tgt_h = tgt_bbox[3] - tgt_bbox[1]

    # Source bottom should be at or above target top
    assert src_pos[1] + src_h <= tgt_pos[1], "Source and target text overlap vertically"

    total_h = src_h + gen.config.get('line_spacing', 24) + tgt_h
    assert total_h <= resolution[1] - 200 + 1  # within available content height
