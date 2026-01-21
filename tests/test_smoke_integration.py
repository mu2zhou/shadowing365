import os
import shutil
import yaml
import pytest
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_to_video import ShadowingVideoGenerator

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "temp_smoke_config.yaml"

def test_smoke_pipeline_runs_and_outputs(tmp_path):
    # Use repo-provided temp_smoke_config.yaml but override output dir to tmp path
    with open(SMOKE_CONFIG, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg['output_dir'] = str(tmp_path / 'smoke_output')
    cfg_path = tmp_path / 'smoke_config.yaml'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)

    # Ensure no prior outputs (remove before instantiating so __init__ can recreate)
    if os.path.exists(cfg['output_dir']):
        shutil.rmtree(cfg['output_dir'])

    gen = ShadowingVideoGenerator(str(cfg_path))

    # Run the pipeline (may skip heavy external services because config uses mock providers)
    gen.run()

    # Check that step files were produced
    step1 = os.path.join(cfg['output_dir'], 'step1_translated.json')
    step2 = os.path.join(cfg['output_dir'], 'step2_audio.json')

    assert os.path.exists(step1), "step1_translated.json not created"
    assert os.path.exists(step2), "step2_audio.json not created"

    # Either an mp4 exists, or temp_audio contains generated files
    mp4s = list(Path(cfg['output_dir']).glob('*.mp4'))
    temp_audio_dir = os.path.join(cfg['output_dir'], 'temp_audio')

    if mp4s:
        # If mp4 created, test passes
        assert any(p.exists() for p in mp4s)
    else:
        # Otherwise ensure some temp audio files exist
        assert os.path.isdir(temp_audio_dir) and any(Path(temp_audio_dir).iterdir()), "No mp4 and no temp audio present"