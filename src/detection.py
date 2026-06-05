import mediapipe as mp
from threading import Lock
from collections import deque
import math
import time

mp_pose = mp.solutions.pose

# ── チューニング定数 ────────────────────────────────────────────────────
SWING_ENTER_THRESH = 0.006  # STANCE→SWING: この速度を超えたらスイング開始
                            #   すり足で検知しない場合は 0.004 まで下げる
SWING_EXIT_THRESH  = 0.003  # SWING→STANCE: この速度を下回ったら接地と判定
                            #   SWING_ENTER_THRESH より低くしてヒステリシスを作る
MIN_SWING_DIST     = 0.02   # 1歩とみなす最小累積移動量（正規化座標）
                            #   ノイズ除去用。すり足で弾かれる場合は 0.01 に下げる
REFRACTORY_SEC     = 0.35   # 接地イベント後の不応期（秒）
VISIBILITY_MIN     = 0.5    # ランドマーク可視性の最低値
SMOOTH_N           = 5      # 速度移動平均フレーム数
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

# 足ごとのランドマーク (足首・かかと・つま先)
# すり足ではつま先が最も動くため3点の最大値を使う
_FOOT_LM = {
    'L': (mp_pose.PoseLandmark.LEFT_ANKLE,
          mp_pose.PoseLandmark.LEFT_HEEL,
          mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
    'R': (mp_pose.PoseLandmark.RIGHT_ANKLE,
          mp_pose.PoseLandmark.RIGHT_HEEL,
          mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
}

_prev_pos     = {side: {lm: None for lm in lms} for side, lms in _FOOT_LM.items()}
_speed_buf    = {'L': deque(maxlen=SMOOTH_N), 'R': deque(maxlen=SMOOTH_N)}
_state        = {'L': 'STANCE', 'R': 'STANCE'}   # 'STANCE' | 'SWING'
_swing_dist   = {'L': 0.0, 'R': 0.0}             # SWING中の累積移動量
_last_contact = {'L': 0.0, 'R': 0.0}             # 最後の接地イベント時刻

# server.py 側でデバッグ表示に使う（dictはインポート先でも変化が見える）
debug = {
    'speed':      {'L': 0.0, 'R': 0.0},
    'swing_dist': {'L': 0.0, 'R': 0.0},
}


def _foot_speed(side, lm_data):
    """足首・かかと・つま先のうち最も大きい1フレーム移動量を返す"""
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


def _smooth(side, raw):
    _speed_buf[side].append(raw)
    return sum(_speed_buf[side]) / len(_speed_buf[side])


def get_status():
    with _lock:
        return dict(current_status)


def detect(frame):
    """
    各足を STANCE / SWING の2状態で管理する。
    SWING→STANCE の遷移時に「接地イベント」を発火する（足音タイミング）。

      STANCE ─[spd > SWING_ENTER_THRESH]→ SWING
      SWING  ─[spd < SWING_EXIT_THRESH]──→ STANCE + 接地イベント
                                            （累積移動量 >= MIN_SWING_DIST の場合のみ）

    戻り値: (results, message, landed)
      landed: 今フレームで接地した足のリスト ['L'], ['R'], ['L','R'], []
    """
    results = pose.process(frame)
    landed = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        now = time.time()

        for side in ('L', 'R'):
            raw = _foot_speed(side, lm)   # 今フレームの生移動量（累積距離用）
            spd = _smooth(side, raw)      # 平滑化速度（状態遷移の判定用）

            debug['speed'][side]      = spd
            debug['swing_dist'][side] = _swing_dist[side]

            if _state[side] == 'STANCE':
                if spd > SWING_ENTER_THRESH:
                    _state[side]      = 'SWING'
                    _swing_dist[side] = raw   # 遷移フレームの移動量を含める

            else:  # SWING
                _swing_dist[side] += raw      # 累積移動量を積算

                if spd < SWING_EXIT_THRESH:
                    # SWING→STANCE: 接地イベント判定
                    # 累積移動量が最小距離以上 かつ 不応期を過ぎていれば発火
                    if (_swing_dist[side] >= MIN_SWING_DIST and
                            now - _last_contact[side] >= REFRACTORY_SEC):
                        _last_contact[side] = now
                        landed.append(side)
                    _state[side]      = 'STANCE'
                    _swing_dist[side] = 0.0

        if landed:
            label = '・'.join('左足' if s == 'L' else '右足' for s in landed)
            message = f"{label}が着地！"
        else:
            message = "人物検出中"
    else:
        # 人物未検出: 前フレーム位置をリセット（再出現時の誤速度計算を防ぐ）
        for side in ('L', 'R'):
            _prev_pos[side]           = {lm_id: None for lm_id in _FOOT_LM[side]}
            _state[side]              = 'STANCE'
            _swing_dist[side]         = 0.0
            debug['speed'][side]      = 0.0
            debug['swing_dist'][side] = 0.0
        message = "人物未検出"

    with _lock:
        current_status["message"] = message

    return results, message, landed
