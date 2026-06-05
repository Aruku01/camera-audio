import pygame
import numpy as np

pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
pygame.mixer.init()


def _beep(freq, duration=0.08):
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * 0.6 * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(wave)


# 左右で音程を変えて聞き分けやすくする
# 実際の足音WAVファイルに置き換える場合: pygame.mixer.Sound("footstep.wav")
_SOUNDS = {
    'L': _beep(freq=380),
    'R': _beep(freq=520),
}


def play(side):
    _SOUNDS[side].play()
