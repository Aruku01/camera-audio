import mediapipe as mp
from threading import Lock
from collections import deque
import math
import time

mp_pose = mp.solutions.pose

# ── チューニング定数 ────────────────────────────────────────────────────
PEAK_MIN         = 0.006  # 「山あり」とみなす最低速度（正規化座標/フレーム）
                          #   すり足で反応しない場合は 0.004 まで下げる
VALLEY_THRESH    = 0.003  # この速度を下回ったら谷=接地と即判定
                          #   PEAK_MIN より低くすること
VALLEY_DROP_RATIO = 0.5   # 変曲点による谷判定の条件：
                          #   peak_spd * VALLEY_DROP_RATIO まで下がってから
                          #   上昇に転じた場合のみ発火（微小なギザギザを無視）
REFRACTORY_SEC   = 0.5    # 接地イベント後の不応期（秒）
VISIBILITY_MIN   = 0.5    # ランドマーク可視性の最低値
SMOOTH_N         = 5      # 速度移動平均フレーム数
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

# 足ごとのランドマーク（足首・かかと・つま先）
# すり足はつま先が最も動くため3点の最大値を使う
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

# 状態: 'SEEK_PEAK'(山待ち) | 'SEEK_VALLEY'(谷待ち)
_state        = {'L': 'SEEK_PEAK', 'R': 'SEEK_PEAK'}
_prev_spd     = {'L': 0.0, 'R': 0.0}     # SEEK_VALLEY 中の前フレーム速度
_peak_spd     = {'L': 0.0, 'R': 0.0}     # SEEK_VALLEY 中に観測した最大速度
_descending   = {'L': False, 'R': False}  # 速度が下降中か
_last_contact = {'L': 0.0, 'R': 0.0}

# server.py からインポートしてデバッグ表示に使う
debug = {
    'speed':      {'L': 0.0, 'R': 0.0},
    'peak_spd':   {'L': 0.0, 'R': 0.0},
    'descending': {'L': False, 'R': False},
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
    各足の速度が「山→谷」を刻むたびに接地イベントを発火する。

      SEEK_PEAK  ─[spd >= PEAK_MIN]────────────────────▶ SEEK_VALLEY
      SEEK_VALLEY ─[spd < VALLEY_THRESH                 ▶ 接地イベント + SEEK_PEAK
                    OR (下降後に上昇へ転じた変曲点)]

    連続歩行では速度が周期的に山谷を繰り返すため、1歩ごとに発火できる。
    累積移動量ではなく瞬時速度の変化を見るためリセット漏れが起きない。

    戻り値: (results, message, landed)
      landed: 今フレームで接地した足のリスト ['L'], ['R'], ['L','R'], []
    """
    results = pose.process(frame)
    landed = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        now = time.time()

        for side in ('L', 'R'):
            raw = _foot_speed(side, lm)
            spd = _smooth(side, raw)

            if _state[side] == 'SEEK_PEAK':
                if spd >= PEAK_MIN:
                    _state[side]      = 'SEEK_VALLEY'
                    _prev_spd[side]   = spd
                    _peak_spd[side]   = spd
                    _descending[side] = False

            else:  # SEEK_VALLEY
                # SEEK_VALLEY 中の最大速度を更新
                if spd > _peak_spd[side]:
                    _peak_spd[side] = spd

                # 速度が前フレームより下がっていたら「下降中」フラグを立てる
                if spd < _prev_spd[side]:
                    _descending[side] = True

                # 谷の判定（どちらか一方で発火）
                valley_by_thresh  = spd < VALLEY_THRESH
                # 変曲点による谷判定：peak_spd の VALLEY_DROP_RATIO 以下まで
                # 下がってから上昇に転じた場合のみ（微小なギザギザを無視）
                dropped_enough    = spd <= _peak_spd[side] * VALLEY_DROP_RATIO
                valley_by_inflect = _descending[side] and dropped_enough and spd > _prev_spd[side]

                if valley_by_thresh or valley_by_inflect:
                    if now - _last_contact[side] >= REFRACTORY_SEC:
                        _last_contact[side] = now
                        landed.append(side)
                    # 次の山を待つ状態へリセット
                    _state[side]      = 'SEEK_PEAK'
                    _prev_spd[side]   = 0.0
                    _peak_spd[side]   = 0.0
                    _descending[side] = False
                else:
                    _prev_spd[side] = spd

            debug['speed'][side]      = spd
            debug['peak_spd'][side]   = _peak_spd[side]
            debug['descending'][side] = _descending[side]

        if landed:
            label = '・'.join('左足' if s == 'L' else '右足' for s in landed)
            message = f"{label}が着地！"
        else:
            message = "人物検出中"
    else:
        # 再出現時の誤速度を防ぐため前フレーム位置をクリア
        for side in ('L', 'R'):
            _prev_pos[side]             = {lm_id: None for lm_id in _FOOT_LM[side]}
            _state[side]                = 'SEEK_PEAK'
            _prev_spd[side]             = 0.0
            _peak_spd[side]             = 0.0
            _descending[side]           = False
            debug['speed'][side]        = 0.0
            debug['peak_spd'][side]     = 0.0
            debug['descending'][side]   = False
        message = "人物未検出"

    with _lock:
        current_status["message"] = message

    return results, message, landed
