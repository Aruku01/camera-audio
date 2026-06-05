import mediapipe as mp
from threading import Lock
from collections import deque
import time

mp_pose = mp.solutions.pose

LANDING_COOLDOWN = 0.4
MIN_Y_LIFT  = 0.04   # Y方向の最小離地量（上昇方向）
MIN_X_SWING = 0.07   # 股関節基準の足首の最小前後変位
VISIBILITY_MIN = 0.5
SMOOTH_N = 5

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

current_status = {"message": "検出待機中..."}
_lock = Lock()

_y_buf  = {'L': deque(maxlen=SMOOTH_N), 'R': deque(maxlen=SMOOTH_N)}
_rx_buf = {'L': deque(maxlen=SMOOTH_N), 'R': deque(maxlen=SMOOTH_N)}

_state       = {'L': 'ground', 'R': 'ground'}  # 'ground' | 'moving'
_ground_y    = {'L': None,     'R': None}
_ground_rx   = {'L': None,     'R': None}   # 地面時の股関節基準X
_air_min_y   = {'L': 1.0,      'R': 1.0}
_max_x_swing = {'L': 0.0,      'R': 0.0}   # 動作中の最大前後変位
_last_land_time = {'L': 0.0,   'R': 0.0}

_ANKLE_LM = {'L': mp_pose.PoseLandmark.LEFT_ANKLE,  'R': mp_pose.PoseLandmark.RIGHT_ANKLE}
_HIP_LM   = {'L': mp_pose.PoseLandmark.LEFT_HIP,    'R': mp_pose.PoseLandmark.RIGHT_HIP}


def _smooth_y(side, y):
    _y_buf[side].append(y)
    return sum(_y_buf[side]) / len(_y_buf[side])


def _smooth_rx(side, rx):
    _rx_buf[side].append(rx)
    return sum(_rx_buf[side]) / len(_rx_buf[side])


def get_status():
    with _lock:
        return dict(current_status)


def detect(frame):
    """
    足の離地を「Y上昇」または「股関節基準X前後変位」のいずれかで検出し、
    Y座標が戻った時点を着地と判定する。

    戻り値: (results, message, landed)
      landed: 着地した足のリスト ['L'], ['R'], []
    """
    results = pose.process(frame)
    landed = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        now = time.time()

        for side in ('L', 'R'):
            ankle = lm[_ANKLE_LM[side]]
            hip   = lm[_HIP_LM[side]]

            if ankle.visibility < VISIBILITY_MIN:
                continue

            y = _smooth_y(side, ankle.y)

            # 股関節が見えていれば前後変位を使用、見えなければY軸のみ
            hip_ok = hip.visibility >= VISIBILITY_MIN
            rx = _smooth_rx(side, ankle.x - hip.x) if hip_ok else (_ground_rx[side] or 0.0)

            if _state[side] == 'ground':
                if _ground_y[side] is None:
                    _ground_y[side] = y
                    _ground_rx[side] = rx

                y_lifted = y < _ground_y[side] - MIN_Y_LIFT
                x_swung  = abs(rx - (_ground_rx[side] or rx)) > MIN_X_SWING

                if y_lifted or x_swung:
                    _state[side]       = 'moving'
                    _air_min_y[side]   = y
                    _max_x_swing[side] = abs(rx - (_ground_rx[side] or rx))

            else:  # 'moving'
                if y < _air_min_y[side]:
                    _air_min_y[side] = y
                swing = abs(rx - (_ground_rx[side] or rx))
                if swing > _max_x_swing[side]:
                    _max_x_swing[side] = swing

                # 着地判定: Yが最高点から戻りつつ、有意な動きがあった
                y_returning = y > _air_min_y[side] + MIN_Y_LIFT * 0.5
                had_y_motion = (_ground_y[side] or 0) - _air_min_y[side] >= MIN_Y_LIFT
                had_x_motion = _max_x_swing[side] >= MIN_X_SWING

                if y_returning and (had_y_motion or had_x_motion):
                    _state[side]       = 'ground'
                    _ground_y[side]    = y
                    _ground_rx[side]   = rx
                    _max_x_swing[side] = 0.0
                    if now - _last_land_time[side] > LANDING_COOLDOWN:
                        _last_land_time[side] = now
                        landed.append(side)

        if landed:
            label = '・'.join('左足' if s == 'L' else '右足' for s in landed)
            message = f"{label}が着地！"
        else:
            message = "人物検出中"
    else:
        for side in ('L', 'R'):
            _ground_y[side]  = None
            _ground_rx[side] = None
            _state[side]     = 'ground'
        message = "人物未検出"

    with _lock:
        current_status["message"] = message

    return results, message, landed
