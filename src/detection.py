import mediapipe as mp
from threading import Lock
from collections import deque
import time

mp_pose = mp.solutions.pose

# ── チューニング定数 ────────────────────────────────────────────────────
SMOOTH_N       = 5     # 足首Y座標の移動平均フレーム数
MIN_AMP        = 0.03  # 谷と認める最小振幅（正規化座標）
                       #   立ち止まり時の微小揺れを除外する
                       #   すり足で反応しない場合は 0.01 まで下げる
REFRACTORY_SEC = 0.5   # 接地イベント後の不応期（秒）
VISIBILITY_MIN = 0.5   # ランドマーク可視性の最低値
# ────────────────────────────────────────────────────────────────────────

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

current_status = {"message": "検出待機中..."}
_lock = Lock()

_ANKLE_LM = {
    'L': mp_pose.PoseLandmark.LEFT_ANKLE,   # 27
    'R': mp_pose.PoseLandmark.RIGHT_ANKLE,  # 28
}

_y_buf        = {'L': deque(maxlen=SMOOTH_N), 'R': deque(maxlen=SMOOTH_N)}
# 状態: 'AWAIT_DESCENT'（下降待ち） | 'IN_DESCENT'（下降中）
_state        = {'L': 'AWAIT_DESCENT', 'R': 'AWAIT_DESCENT'}
_prev_y       = {'L': None, 'R': None}
_local_max_y  = {'L': 0.0, 'R': 0.0}  # 今回の下降が始まった時点のY
_min_y        = {'L': 1.0, 'R': 1.0}  # 今回の下降中の最小Y
_last_contact = {'L': 0.0, 'R': 0.0}

# server.py からインポートしてデバッグ表示に使う
debug = {
    'y':     {'L': 0.0, 'R': 0.0},
    'amp':   {'L': 0.0, 'R': 0.0},
    'state': {'L': 'AWAIT_DESCENT', 'R': 'AWAIT_DESCENT'},
}


def _smooth_y(side, raw_y):
    _y_buf[side].append(raw_y)
    return sum(_y_buf[side]) / len(_y_buf[side])


def get_status():
    with _lock:
        return dict(current_status)


def reset_state():
    """人物消失・動画ループ時に呼んで検出状態をクリアする"""
    for side in ('L', 'R'):
        _y_buf[side].clear()
        _state[side]        = 'AWAIT_DESCENT'
        _prev_y[side]       = None
        _local_max_y[side]  = 0.0
        _min_y[side]        = 1.0
        debug['y'][side]    = 0.0
        debug['amp'][side]  = 0.0
        debug['state'][side] = 'AWAIT_DESCENT'


def detect(frame):
    """
    足首Y座標の谷（極小点）を接地イベントとして発火する。

    MediaPipe 正規化座標は Y=0 が上端、Y=1 が下端。
    足が浮くと Y が減少し、地面へ戻ると Y が増加する。

      AWAIT_DESCENT ─[Y が下降開始]──────────────────────────▶ IN_DESCENT
      IN_DESCENT    ─[Y が上昇に転じ amp >= MIN_AMP]──────────▶ 接地イベント + AWAIT_DESCENT

    戻り値: (results, message, landed)
      landed: 今フレームで接地した足のリスト ['L'], ['R'], ['L','R'], []
    """
    results = pose.process(frame)
    landed = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        now = time.time()

        for side in ('L', 'R'):
            ankle = lm[_ANKLE_LM[side]]
            if ankle.visibility < VISIBILITY_MIN:
                continue

            y = _smooth_y(side, ankle.y)
            prev = _prev_y[side]
            _prev_y[side] = y

            if prev is None:
                continue

            if _state[side] == 'AWAIT_DESCENT':
                if y < prev:  # Y が減少 = 足が上昇
                    _local_max_y[side] = prev
                    _min_y[side] = y
                    _state[side] = 'IN_DESCENT'

            else:  # IN_DESCENT
                if y < _min_y[side]:
                    _min_y[side] = y
                if y >= prev:  # 上昇に転じた = 谷を通過
                    amp = _local_max_y[side] - _min_y[side]
                    if amp >= MIN_AMP and now - _last_contact[side] >= REFRACTORY_SEC:
                        _last_contact[side] = now
                        landed.append(side)
                    _state[side]       = 'AWAIT_DESCENT'
                    _local_max_y[side] = 0.0
                    _min_y[side]       = 1.0

            debug['y'][side]    = y
            debug['amp'][side]  = (_local_max_y[side] - _min_y[side]
                                   if _state[side] == 'IN_DESCENT' else 0.0)
            debug['state'][side] = _state[side]

        if landed:
            label = '・'.join('左足' if s == 'L' else '右足' for s in landed)
            message = f"{label}が着地！"
        else:
            message = "人物検出中"
    else:
        reset_state()
        message = "人物未検出"

    with _lock:
        current_status["message"] = message

    return results, message, landed
