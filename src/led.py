from gpiozero import LED
from threading import Thread
import time

_LED_PIN = 17
_FLASH_DURATION = 0.1  # 点灯時間（秒）

_led = LED(_LED_PIN)


def flash():
    """着地検出時にLEDを短く点滅させる（ノンブロッキング）"""
    def _blink():
        _led.on()
        time.sleep(_FLASH_DURATION)
        _led.off()
    Thread(target=_blink, daemon=True).start()
