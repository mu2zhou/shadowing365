import os
from pathlib import Path
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_to_video import ShadowingVideoGenerator


def test_generate_audio_raises_when_tts_unavailable(tmp_path):
    # Create config that points to a tmp output dir and points fish_speech ref to something missing
    cfg = {
        'output_dir': str(tmp_path / 'out'),
        'project_name': 'fail_test',
        'resolution': [320, 240],
        'font_size_source': 20,
        'font_size_target': 12,
        'fps': 12,
        'tts_provider': 'fish_speech',
        'fish_speech_ref_audio': 'input/nonexistent_ref.wav',
        'tts_retry_attempts': 1,
        'tts_fallback_providers': []
    }
    cfg_path = tmp_path / 'cfg.yaml'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)

    gen = ShadowingVideoGenerator(str(cfg_path))

    bilingual = [
        {'source': 'This will fail', 'target': '[译] 这会失败'}
    ]

    # Expect generate_audio to raise because fish_speech cannot find ref
    try:
        gen.generate_audio(bilingual)
        raised = False
    except RuntimeError:
        raised = True

    assert raised, 'generate_audio did not raise when TTS was unavailable'