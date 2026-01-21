import os
import json
import pytest
import wave
import struct
import subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_to_video import ShadowingVideoGenerator


def make_cfg(tmp_path):
    cfg = {
        'output_dir': str(tmp_path / 'out'),
        'fish_retry_attempts': 3,
        'fish_retry_backoff_base': 0.1,
        'fish_retry_jitter': 0.0,
        'fish_chunk_word_limit': 6  # small for tests
    }
    p = tmp_path / 'cfg.yaml'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    return str(p)


def make_sample_mp3(path, duration=0.5):
    # Create a short WAV and convert to mp3 via ffmpeg
    wav = path.with_suffix('.wav')
    rate = 22050
    n = int(duration * rate)
    with wave.open(str(wav), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for i in range(n):
            t = i / rate
            val = int(0.6 * 16000.0 * __import__('math').sin(2 * __import__('math').pi * 440.0 * t))
            wf.writeframes(struct.pack('<h', val))
    mp3 = path
    subprocess.check_call(['ffmpeg', '-y', '-hide_banner', '-nostats', '-i', str(wav), '-codec:a', 'libmp3lame', '-q:a', '4', str(mp3)])
    return mp3.read_bytes()


def test_fish_chunk_fallback_success(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    gen = ShadowingVideoGenerator(cfg)

    # Provide a fake reference audio so generator doesn't early-exit
    ref = tmp_path / 'ref.wav'
    make_sample_mp3(ref.with_suffix('.mp3')).__len__()
    # write a tiny ref file
    open(ref, 'wb').write(b"RIFF")
    gen.config['fish_speech_ref_audio'] = str(ref)

    long_text = 'Part one; Part two; Part three; Part four; Part five; Part six; Part seven.'

    # Simulate _fish_post behavior: returns 500 for full text, but 200+mp3 bytes for short chunks
    async def fake_fish_post(payload):
        class R:
            def __init__(self, code, content, text=''):
                self.status_code = code
                self.content = content
                self.text = text
        t = payload.get('text', '')
        if len(t.split()) > 6:
            return R(500, b'', 'Simulated server error')
        else:
            # return small mp3 content
            content = make_sample_mp3(tmp_path / f'tmp_chunk_{len(t.split())}.mp3')
            return R(200, content, '')

    monkeypatch.setattr(gen, '_fish_post', fake_fish_post)

    out = tmp_path / 'out_seg.mp3'
    import asyncio
    ok = asyncio.run(gen._generate_audio_fish_speech(long_text, str(out), segment_index=99))
    assert ok
    assert out.exists()
    # Ensure audio has some duration
    dur = float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1', str(out)]).decode().strip())
    assert dur > 0.1


def test_fish_retries_then_success(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    gen = ShadowingVideoGenerator(cfg)
    ref = tmp_path / 'ref.wav'
    open(ref, 'wb').write(b"RIFF")
    gen.config['fish_speech_ref_audio'] = str(ref)

    text = 'Short text.'
    calls = {'n': 0}

    async def fake_fish_post(payload):
        class R:
            def __init__(self, code, content, text=''):
                self.status_code = code
                self.content = content
                self.text = text
        calls['n'] += 1
        if calls['n'] < 3:
            return R(500, b'', 'temporarily overloaded')
        else:
            content = make_sample_mp3(tmp_path / f'ok_{calls["n"]}.mp3')
            return R(200, content, '')

    monkeypatch.setattr(gen, '_fish_post', fake_fish_post)

    out = tmp_path / 'out_seg2.mp3'
    import asyncio
    ok = asyncio.run(gen._generate_audio_fish_speech(text, str(out), segment_index=100))
    assert ok
    assert out.exists()
    dur = float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1', str(out)]).decode().strip())
    assert dur > 0.1
    assert calls['n'] >= 3
