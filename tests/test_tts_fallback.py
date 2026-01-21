import os
import yaml
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_to_video import ShadowingVideoGenerator


def make_cfg(tmp_path, provider, fallbacks=None, attempts=1):
    cfg = {
        'output_dir': str(tmp_path / 'out'),
        'project_name': 'fallback_test',
        'resolution': [320, 240],
        'font_size_source': 20,
        'font_size_target': 12,
        'fps': 12,
        'tts_provider': provider,
        'tts_retry_attempts': attempts,
        'tts_fallback_providers': fallbacks or [],
        # Ensure mock fallback provides sufficient duration for short English sentences
        'mock_tts_duration': 1.5
    }
    cfg_path = tmp_path / 'cfg.yaml'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
    return str(cfg_path)


def test_fallback_to_mock(tmp_path):
    cfg_path = make_cfg(tmp_path, provider='fish_speech', fallbacks=['mock'], attempts=1)
    gen = ShadowingVideoGenerator(cfg_path)

    bilingual = [
        {'source': 'This will use mock fallback', 'target': '[译] 这将使用回退'}
    ]

    # Should succeed using mock fallback (since fish_speech will fail due to missing ref)
    audio_data = gen.generate_audio(bilingual)
    assert len(audio_data) == 1
    assert audio_data[0].get('audio_path'), 'Fallback did not produce audio file'
    assert os.path.exists(audio_data[0]['audio_path']), 'Audio file from fallback provider not found'


def test_no_fallback_raises(tmp_path):
    cfg_path = make_cfg(tmp_path, provider='fish_speech', fallbacks=[], attempts=1)
    gen = ShadowingVideoGenerator(cfg_path)
    bilingual = [{'source': 'This will fail', 'target': '[译] 失败'}]

    try:
        gen.generate_audio(bilingual)
        raised = False
    except RuntimeError:
        raised = True

    assert raised, 'Expected RuntimeError when no fallback and provider fails'