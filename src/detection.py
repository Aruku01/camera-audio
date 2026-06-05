import mediapipe as mp
from collections import deque
from threading import Lock
import time

mp_pose = mp.solutions.pose

# 着地判定の閾値（正規化座標/フレーム）。カメラ距離・フレームレートに応じて調整する
_LANDING_VEL_MIN = 0.005   # 着地前に必要な最小下降速度
_LANDING_STOP_MAX = 0.004  # 着地とみなす最大速度（ほぼ停止）
LANDING_COOLDOWN = 0.35    # 同じ足の連続着地を防ぐ間隔（秒）

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

current_status = {"message": "検出待機中..."}
_lock = Lock()
_ankle_y = {'L': deque(maxlen=8), 'R': deque(maxlen=8)}
_last_landing_time = {'L': 0.0, 'R': 0.0}


def _check_landing(side):
    """Y座標が下降→停止に転換した瞬間（着地）を検出する"""
    h = list(_ankle_y[side])
    if len(h) < 4:
        return False
    vel_prev = h[-2] - h[-3]  # 1フレーム前の下降速度（正 = 画面下方向）
    vel_curr = h[-1] - h[-2]  # 最新フレームの速度
    return vel_prev > _LANDING_VEL_MIN and vel_curr <= _LANDING_STOP_MAX


def get_status():
    with _lock:
        return dict(current_status)


def detect(frame):
    """
    戻り値: (results, message, landed)
      landed: このフレームで着地した足のリスト。例: ['L'], ['R'], ['L','R'], []
    """
    results = pose.process(frame)
    landed = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        now = time.time()

        for side, ankle in [('L', left_ankle), ('R', right_ankle)]:
            _ankle_y[side].append(ankle.y)
            if _check_landing(side) and now - _last_landing_time[side] > LANDING_COOLDOWN:
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
