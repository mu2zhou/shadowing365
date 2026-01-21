import os
import wave
import struct
from pathlib import Path
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_to_video import ShadowingVideoGenerator


def make_cfg(tmp_path):
    cfg = {'output_dir': str(tmp_path / 'out')}
    p = tmp_path / 'cfg.yaml'
    with open(p, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
    return str(p)


def write_tone(path, duration=0.1, amplitude=0.5, rate=22050):
    n = int(duration * rate)
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for i in range(n):
            t = i / rate
            val = int(amplitude * 16000.0 * __import__('math').sin(2 * __import__('math').pi * 440.0 * t))
            wf.writeframes(struct.pack('<h', val))


def test_short_too_short_audio_is_rejected(tmp_path):
    cfg_path = make_cfg(tmp_path)
    gen = ShadowingVideoGenerator(cfg_path)

    # Text long enough to expect >0.5s
    text = 'word ' * 50
    est = gen._estimate_duration_from_text(text)
    assert est > 1.0

    # Create a 0.1s tone
    short = tmp_path / 'short.wav'
    write_tone(short, duration=0.1, amplitude=0.8)

    stats = gen._is_audio_valid(str(short), estimated_duration=est)
    assert not stats['valid'], f"Short audio incorrectly accepted: {stats}"


def test_low_energy_rejected(tmp_path):
    cfg_path = make_cfg(tmp_path)
    gen = ShadowingVideoGenerator(cfg_path)

    # Create 1s very low amplitude audio
    low = tmp_path / 'low.wav'
    write_tone(low, duration=1.0, amplitude=0.001)

    stats = gen._is_audio_valid(str(low), estimated_duration=1.0)
    assert not stats['valid'], f"Low energy audio incorrectly accepted: {stats}"


def test_compress_internal_silences_reduces_long_pauses(tmp_path):
    cfg_path = make_cfg(tmp_path)
    gen = ShadowingVideoGenerator(cfg_path)

    # Build a test MP3 with tone (0.4s) + long silence (3.0s) + tone (0.4s)
    a1 = tmp_path / 'a1.wav'
    a2 = tmp_path / 'a2.wav'
    write_tone(a1, duration=0.4, amplitude=0.8)
    write_tone(a2, duration=0.4, amplitude=0.8)

    sil = tmp_path / 'sil.wav'
    # Create silent WAV using ffmpeg lavfi
    import subprocess
    subprocess.check_call(['ffmpeg', '-y', '-hide_banner', '-nostats', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', '3.0', str(sil)])

    # Concatenate into a single WAV
    concat_txt = tmp_path / 'concat.txt'
    with open(concat_txt, 'w', encoding='utf-8') as cf:
        cf.write(f"file '{a1}'\nfile '{sil}'\nfile '{a2}'\n")

    bigwav = tmp_path / 'big.wav'
    subprocess.check_call(['ffmpeg', '-y', '-hide_banner', '-nostats', '-f', 'concat', '-safe', '0', '-i', str(concat_txt), '-c', 'copy', str(bigwav)])

    # Convert to MP3 (as our compressor outputs MP3)
    bigmp3 = tmp_path / 'big.mp3'
    subprocess.check_call(['ffmpeg', '-y', '-hide_banner', '-nostats', '-i', str(bigwav), '-codec:a', 'libmp3lame', '-q:a', '4', str(bigmp3)])

    # Confirm the input has a long silence (>1.5s)
    out = subprocess.check_output(['ffmpeg', '-hide_banner', '-nostats', '-i', str(bigmp3), '-af', 'silencedetect=noise=-35dB:d=0.5', '-f', 'null', '-'], stderr=subprocess.STDOUT).decode()
    assert 'silence_duration' in out

    # Run compression
    changed = gen._compress_internal_silences(str(bigmp3), max_silence=1.0, replacement=0.2)
    assert changed, 'Expected compressor to modify file with long silence'

    # Re-run silencedetect and ensure no silence >= 1.0 remains
    out2 = subprocess.check_output(['ffmpeg', '-hide_banner', '-nostats', '-i', str(bigmp3), '-af', 'silencedetect=noise=-35dB:d=0.5', '-f', 'null', '-'], stderr=subprocess.STDOUT).decode()
    import re
    durations = [float(m.group(1)) for m in re.finditer(r'silence_duration: ([0-9\.]+)', out2)]
    assert all(d < 1.0 for d in durations), f"Found long silence durations after compression: {durations}"