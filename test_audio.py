"""
音声モジュール単体テスト。
カメラ・MediaPipe不要。スピーカーから音が出るかを確認する。
"""
import sys
import time
sys.path.insert(0, 'src')

from audio import play

print("=== 音声テスト ===")
print("左足音を再生...")
play('L')
time.sleep(1.0)

print("右足音を再生...")
play('R')
time.sleep(1.0)

print("交互に4回再生（歩行シミュレーション）...")
for i in range(4):
    side = 'L' if i % 2 == 0 else 'R'
    label = '左' if side == 'L' else '右'
    print(f"  {label}足")
    play(side)
    time.sleep(0.5)

print("完了。音が聞こえた場合は音声モジュールOK。")
