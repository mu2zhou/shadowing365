import os
from pathlib import Path
import yaml
from PIL import Image, ImageDraw, ImageFont
import wave
import struct
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_to_video import ShadowingVideoGenerator


def make_config(tmp_path):
    cfg = {'output_dir': str(tmp_path / 'out')}
    p = tmp_path / 'cfg.yaml'
    with open(p, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
    return str(p)


def test_wrap_text_cjk(tmp_path):
    cfg_path = make_config(tmp_path)
    gen = ShadowingVideoGenerator(cfg_path)

    resolution = (400, 200)
    img = Image.new('RGB', resolution)
    draw = ImageDraw.Draw(img)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 24)
    except Exception:
        font = ImageFont.load_default()

    # Long Chinese string with no spaces
    long_cn = '这是一个非常长的中文句子，用来测试自动换行功能' * 5
    wrapped = gen._wrap_text(long_cn, font, max_width=resolution[0] - 40, draw=draw)

    # Ensure each line fits within max_width
    for line in wrapped.split('\n'):
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2]
        assert w <= resolution[0] - 40, f"CJK wrap line too wide: width {w} > max"


def test_layout_scales_to_fit_width(tmp_path):
    cfg_path = make_config(tmp_path)
    gen = ShadowingVideoGenerator(cfg_path)

    resolution = (400, 200)
    img = Image.new('RGB', resolution)
    draw = ImageDraw.Draw(img)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 24)
    except Exception:
        font = ImageFont.load_default()

    # Construct an extremely long target (Chinese) that would normally overflow
    long_cn = '这是一个非常长的中文句子，用来测试自动换行功能' * 8
    item = {'source': 'Some short source', 'target': long_cn}
    layout = gen._compute_text_layout(item, draw, resolution, font_path, 0, 40, 28)

    max_width = resolution[0] - 200
    src_lines = layout['src_text'].split('\n')
    tgt_lines = layout['tgt_text'].split('\n')

    for line in src_lines:
        bbox = draw.textbbox((0, 0), line, font=layout['font_src'])
        assert (bbox[2] - bbox[0]) <= max_width

    for line in tgt_lines:
        bbox = draw.textbbox((0, 0), line, font=layout['font_tgt'])
        assert (bbox[2] - bbox[0]) <= max_width


def test_audio_validation_rejects_silence(tmp_path):
    cfg_path = make_config(tmp_path)
    gen = ShadowingVideoGenerator(cfg_path)

    # Write a 1s silent WAV
    out = tmp_path / 'silent.wav'
    sample_rate = 22050
    duration = 1.0
    n_samples = int(sample_rate * duration)
    with wave.open(str(out), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        silence = struct.pack('<h', 0)
        for _ in range(n_samples):
            wf.writeframes(silence)

    # Validate
    stats = gen._is_audio_valid(str(out))
    assert not stats.get('valid', False), "Silent audio should be rejected by validator"
    assert stats.get('rms', 0.0) == 0.0 and stats.get('non_silent_ratio', 1.0) == 0.0, "Silent audio should have zero RMS and non-silent ratio"