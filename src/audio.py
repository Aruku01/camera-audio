import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import numpy as np
from pathlib import Path
pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
pygame.mixer.init()

_SOUNDS_DIR = Path(__file__).parent.parent / "assets" / "sounds"

# 期待するファイル名
_FILES = {
    'L': "footstep_left.wav",
    'R': "footstep_right.wav",
}


def _load_or_beep(filename, fallback_freq):
    """WAVファイルがあればロード、なければビープ音で代替する"""
    path = _SOUNDS_DIR / filename
    if path.exists():
        print(f"[audio] 音声ファイルを読み込みました: {path}")
        return pygame.mixer.Sound(str(path))
    print(f"[audio] {path} が見つかりません。ビープ音で代替します。")
    sr = 44100
    duration = 0.08
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * fallback_freq * t) * 0.6 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(wave)


_SOUNDS = {
    'L': _load_or_beep(_FILES['L'], fallback_freq=380),
    'R': _load_or_beep(_FILES['R'], fallback_freq=520),
}


def play(side):
    """指定した足（'L' or 'R'）の足音を再生する"""
    _SOUNDS[side].play()
