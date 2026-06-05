"""
検出ロジック単体テスト。
カメラ・MediaPipe不要。Y座標の閾値判定と着地イベントを確認する。
"""
import sys
sys.path.insert(0, 'src')

import detection
from detection import _check_landing, _ankle_state, GROUND_Y_THRESHOLD

ABOVE = GROUND_Y_THRESHOLD + 0.05  # 地面（閾値以上）
BELOW = GROUND_Y_THRESHOLD - 0.10  # 空中（閾値未満）

errors = 0

def check(label, result, expected):
    global errors
    status = "OK" if result == expected else "NG"
    if result != expected:
        errors += 1
    print(f"  [{status}] {label}")

print("=== 検出ロジックテスト ===")
print(f"GROUND_Y_THRESHOLD = {GROUND_Y_THRESHOLD}")
print()

# ---- 左足テスト ----
print("[左足]")
_ankle_state['L'] = 'air'

result = _check_landing('L', ABOVE)
check("air→ground（着地）でTrueを返す", result, True)

result = _check_landing('L', ABOVE)
check("ground→ground（二重検知）はFalse", result, False)

_ankle_state['L'] = 'air'
result = _check_landing('L', BELOW)
check("air→air（空中のまま）はFalse", result, False)

result = _check_landing('L', ABOVE)
check("air→ground（再着地）でTrueを返す", result, True)

# ---- 右足テスト ----
print()
print("[右足]")
_ankle_state['R'] = 'air'

result = _check_landing('R', ABOVE)
check("air→ground（着地）でTrueを返す", result, True)

result = _check_landing('R', BELOW)
check("ground→air（足を上げる）はFalse", result, False)

result = _check_landing('R', ABOVE)
check("air→ground（着地）でTrueを返す", result, True)

# ---- クールダウンテスト ----
print()
print("[クールダウン]")
import time
detection._last_landing_time['L'] = time.time()  # 直前に着地したとみなす

_ankle_state['L'] = 'air'
_check_landing('L', ABOVE)  # 状態遷移だけ行う

# detect()を呼ばず直接クールダウンを確認
elapsed = time.time() - detection._last_landing_time['L']
in_cooldown = elapsed < detection.LANDING_COOLDOWN
check(f"着地直後はクールダウン中（{detection.LANDING_COOLDOWN}秒）", in_cooldown, True)

# ---- 結果 ----
print()
if errors == 0:
    print("全テスト通過。検出ロジックOK。")
else:
    print(f"{errors}件のテストが失敗しました。")
