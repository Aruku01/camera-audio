import mediapipe as mp
from threading import Lock
from collections import deque
import time

mp_pose = mp.solutions.pose

LANDING_COOLDOWN = 0.4   # 同じ足の連続着地防止（秒）
MIN_LIFT = 0.05          # 着地前にこの量以上足首が上がらないと無効（正規化座標）
VISIBILITY_MIN = 0.5     # ランドマーク可視性の最低値（これ未満は無視）
SMOOTH_N = 5             # Y値の平均化フレーム数

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

current_status = {"message": "検出待機中..."}
_lock = Lock()

_y_buf = {'L': deque(maxlen=SMOOTH_N), 'R': deque(maxlen=SMOOTH_N)}
_state = {'L': 'ground', 'R': 'ground'}  # 'ground' | 'lifted'
_ground_y = {'L': None, 'R': None}       # 着地時のY値（動的に記録）
_air_min_y = {'L': 1.0, 'R': 1.0}        # 空中中の最高到達点（最低Y値）
_last_land_time = {'L': 0.0, 'R': 0.0}


def _smooth(side, y):
    _y_buf[side].append(y)
    return sum(_y_buf[side]) / len(_y_buf[side])


def get_status():
    with _lock:
        return dict(current_status)


def detect(frame):
    """
    「持ち上げてから着地」の状態機械で着地を検出する。
    カメラブレ（全ランドマークが同方向に微量移動）では
    MIN_LIFT 分の上昇→下降サイクルが成立しないため誤検知しない。

    戻り値: (results, message, landed)
      landed: このフレームで着地した足のリスト ['L'], ['R'], []
    """
    results = pose.process(frame)
    landed = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        ankles = {
            'L': lm[mp_pose.PoseLandmark.LEFT_ANKLE],
            'R': lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
        }
        now = time.time()

        for side, ankle in ankles.items():
            if ankle.visibility < VISIBILITY_MIN:
                continue

            y = _smooth(side, ankle.y)

            if _state[side] == 'ground':
                if _ground_y[side] is None:
                    _ground_y[side] = y
                # MIN_LIFT 以上上がったら「空中」へ移行
                if y < _ground_y[side] - MIN_LIFT:
                    _state[side] = 'lifted'
                    _air_min_y[side] = y

            else:  # 'lifted'
                if y < _air_min_y[side]:
                    _air_min_y[side] = y
                # 最高到達点から MIN_LIFT 以上下がったら着地
                if y > _air_min_y[side] + MIN_LIFT:
                    _state[side] = 'ground'
                    _ground_y[side] = y
                    if now - _last_land_time[side] > LANDING_COOLDOWN:
                        _last_land_time[side] = now
                        landed.append(side)

        if landed:
            label = '・'.join('左足' if s == 'L' else '右足' for s in landed)
            message = f"{label}が着地！"
        else:
            message = "人物検出中"
    else:
        # 未検出時はベースラインをリセットして次回の再検出に備える
        for side in ('L', 'R'):
            _ground_y[side] = None
            _state[side] = 'ground'
        message = "人物未検出"

    with _lock:
        current_status["message"] = message

    return results, message, landed
