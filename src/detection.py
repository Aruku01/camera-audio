import mediapipe as mp
from threading import Lock
import time

mp_pose = mp.solutions.pose

# 足首Y座標がこの値以上になったら「地面についた」と判定する（正規化座標 0〜1）
# カメラの高さ・距離・人物の大きさに応じて 0.70〜0.85 の範囲で調整する
GROUND_Y_THRESHOLD = 0.75
LANDING_COOLDOWN = 0.35  # 同じ足の連続着地を防ぐ間隔（秒）

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

current_status = {"message": "検出待機中..."}
_lock = Lock()

# 各足が「空中」か「地面」かの状態を保持する
_ankle_state = {'L': 'ground', 'R': 'ground'}
_last_landing_time = {'L': 0.0, 'R': 0.0}


def _check_landing(side, y):
    """Y座標が閾値を超えたとき（air→ground）を着地と判定する"""
    prev = _ankle_state[side]
    current = 'ground' if y >= GROUND_Y_THRESHOLD else 'air'
    _ankle_state[side] = current
    return prev == 'air' and current == 'ground'


def get_status():
    with _lock:
        return dict(current_status)


def detect(frame):
    """
    戻り値: (results, message, landed)
      landed: このフレームで着地した足のリスト。例: ['L'], ['R'], []
    """
    results = pose.process(frame)
    landed = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        now = time.time()

        for side, ankle in [('L', left_ankle), ('R', right_ankle)]:
            if _check_landing(side, ankle.y) and now - _last_landing_time[side] > LANDING_COOLDOWN:
                _last_landing_time[side] = now
                landed.append(side)

        if landed:
            label = '・'.join('左足' if s == 'L' else '右足' for s in landed)
            message = f"{label}が着地！"
        else:
            message = "人物検出中"
    else:
        message = "人物未検出"

    with _lock:
        current_status["message"] = message

    return results, message, landed
