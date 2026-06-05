import mediapipe as mp
from threading import Lock
from collections import deque
import math
import time

mp_pose = mp.solutions.pose

# ── チューニング用定数 ───────────────────────────────────────────────────
MOVE_THRESH         = 0.006  # 移動開始とみなす速度（正規化座標/フレーム）
                             # すり足で反応しない場合は 0.004 まで下げる
STILL_THRESH        = 0.003  # 静止とみなす速度（MOVE_THRESH より低くヒステリシス）
MIN_MOVE_FRAMES     = 3      # 有効な一歩の最低移動フレーム数（ノイズ除去）
MIN_MOVE_FRAMES_AUX = 2      # 補助指標（足首間距離変化）あり時の最低フレーム数
LANDING_COOLDOWN    = 0.35   # 同一足の連続着地を防ぐ間隔（秒）
VISIBILITY_MIN      = 0.5    # ランドマーク可視性の最低値
SMOOTH_N            = 5      # 速度移動平均のフレーム数
ANKLE_DIST_DELTA    = 0.012  # 足首間距離変化量の補助閾値（これ以上変化で歩行確認）
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

# 足ごとのランドマークセット（足首・かかと・つま先）
_FOOT_LM = {
    'L': (mp_pose.PoseLandmark.LEFT_ANKLE,
          mp_pose.PoseLandmark.LEFT_HEEL,
          mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    'R': (mp_pose.PoseLandmark.RIGHT_ANKLE,
          mp_pose.PoseLandmark.RIGHT_HEEL,
          mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
}

_prev_pos = {
    side: {lm_id: None for lm_id in lms}
    for side, lms in _FOOT_LM.items()
}
_speed_buf      = {'L': deque(maxlen=SMOOTH_N), 'R': deque(maxlen=SMOOTH_N)}
_state          = {'L': 'still', 'R': 'still'}  # 'still' | 'moving'
_move_frames    = {'L': 0, 'R': 0}
_last_land_time = {'L': 0.0, 'R': 0.0}

_ankle_dist_buf = deque(maxlen=SMOOTH_N + 1)

# デバッグ値（dictにすることでserver.pyのインポート先からも変化が見える）
debug = {
    'speed':       {'L': 0.0, 'R': 0.0},
    'move_frames': {'L': 0,   'R': 0},
    'dist_delta':  0.0,
    'walking_aux': False,
}


def _foot_speed(side, lm_data):
    """足首・かかと・つま先の中で最も大きい移動量（速度）を返す"""
    max_dist = 0.0
    for lm_id in _FOOT_LM[side]:
        lm = lm_data[lm_id]
        if lm.visibility < VISIBILITY_MIN:
            continue
        prev = _prev_pos[side][lm_id]
        pos = (lm.x, lm.y)
        if prev is not None:
            dx = pos[0] - prev[0]
            dy = pos[1] - prev[1]
            d = math.sqrt(dx * dx + dy * dy)
            if d > max_dist:
                max_dist = d
        _prev_pos[side][lm_id] = pos
    return max_dist


def _smooth_speed(side, raw):
    _speed_buf[side].append(raw)
    return sum(_speed_buf[side]) / len(_speed_buf[side])


def get_status():
    with _lock:
        return dict(current_status)


def detect(frame):
    """
    「静止→移動→静止」の1サイクルを1歩として着地を検出する。

    足首・かかと・つま先のうち最も動いたランドマークの速度を基準にし、
    直近 SMOOTH_N フレームの移動平均でノイズを除去する。
    足首間距離の周期的な変化を補助指標として用いる。

    戻り値: (results, message, landed)
      landed: このフレームで着地した足のリスト ['L'], ['R'], []
    """
    results = pose.process(frame)
    landed = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        now = time.time()

        # 補助指標: 足首間距離の変化量
        la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        dist_delta = 0.0
        if la.visibility >= VISIBILITY_MIN and ra.visibility >= VISIBILITY_MIN:
            dist = math.sqrt((la.x - ra.x) ** 2 + (la.y - ra.y) ** 2)
            _ankle_dist_buf.append(dist)
            if len(_ankle_dist_buf) >= 2:
                dist_delta = abs(_ankle_dist_buf[-1] - _ankle_dist_buf[-2])

        walking_aux = dist_delta >= ANKLE_DIST_DELTA
        debug['dist_delta']  = dist_delta
        debug['walking_aux'] = walking_aux

        for side in ('L', 'R'):
            raw_spd = _foot_speed(side, lm)
            spd = _smooth_speed(side, raw_spd)
            debug['speed'][side]       = spd
            debug['move_frames'][side] = _move_frames[side]

            if _state[side] == 'still':
                if spd > MOVE_THRESH:
                    _state[side]       = 'moving'
                    _move_frames[side] = 1

            else:  # 'moving'
                _move_frames[side] += 1

                if spd < STILL_THRESH:
                    # 補助指標ありなら必要フレーム数を緩和
                    required = MIN_MOVE_FRAMES_AUX if walking_aux else MIN_MOVE_FRAMES
                    if _move_frames[side] >= required:
                        _state[side] = 'still'
                        if now - _last_land_time[side] > LANDING_COOLDOWN:
                            _last_land_time[side] = now
                            landed.append(side)
                    else:
                        # 最低フレーム数未満はノイズとして棄却
                        _state[side] = 'still'
                    _move_frames[side] = 0

        if landed:
            label = '・'.join('左足' if s == 'L' else '右足' for s in landed)
            message = f"{label}が着地！"
        else:
            message = "人物検出中"
    else:
        # 未検出時は全状態をリセット（次回再検出で誤速度が出ないよう前位置もクリア）
        for side in ('L', 'R'):
            _prev_pos[side] = {lm_id: None for lm_id in _FOOT_LM[side]}
            _state[side]     = 'still'
            _move_frames[side] = 0
            debug['speed'][side]       = 0.0
            debug['move_frames'][side] = 0
        _ankle_dist_buf.clear()
        debug['dist_delta']  = 0.0
        debug['walking_aux'] = False
        message = "人物未検出"

    with _lock:
        current_status["message"] = message

    return results, message, landed
