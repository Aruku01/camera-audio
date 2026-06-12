import mediapipe as mp
from flask import Flask, Response, jsonify
import cv2
from detection import detect, get_status, mp_pose, debug, reset_state
from audio import play
from led import flash
from threading import Thread, Lock
import time

app = Flask(__name__)

# ── 入力ソース設定 ─────────────────────────────────────────────────────
# str  → 動画ファイルパス（テスト用）; プロジェクトルートからの相対パス
# 0    → カメラ（本番: picamera2）
VIDEO_SOURCE = "walk01.mp4"
# ────────────────────────────────────────────────────────────────────────

if not isinstance(VIDEO_SOURCE, str):
    from picamera2 import Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

DETECT_SIZE  = (320, 240)
DISPLAY_SIZE = (640, 480)
JPEG_QUALITY = 70

_latest_jpeg = None
_jpeg_lock   = Lock()


def _detection_loop():
    global _latest_jpeg

    if isinstance(VIDEO_SOURCE, str):
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"[ERROR] 動画を開けませんでした: {VIDEO_SOURCE}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 1.0 / fps
        print(f"[video] {VIDEO_SOURCE}  {fps:.1f}fps")
    else:
        frame_interval = 0.0

    while True:
        t_start = time.time()

        if isinstance(VIDEO_SOURCE, str):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                reset_state()
                continue
            frame = cv2.resize(frame, DISPLAY_SIZE)
        else:
            frame = picam2.capture_array()  # BGR 640x480

        small = cv2.resize(frame, DETECT_SIZE)
        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results, text, landed = detect(frame_rgb)

        for side in landed:
            play(side)
        if landed:
            flash()

        display = frame.copy()
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(display, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for i, (side, label) in enumerate([('L', 'Left'), ('R', 'Right')]):
            y   = debug['y'][side]
            amp = debug['amp'][side]
            st  = debug['state'][side]
            fire = ' ★' if side in landed else ''
            dbg = f"{label}: {st}  Y={y:.3f}  amp={amp:.3f}{fire}"
            cv2.putText(display, dbg, (10, 60 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with _jpeg_lock:
            _latest_jpeg = buf.tobytes()

        if frame_interval > 0:
            elapsed = time.time() - t_start
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


Thread(target=_detection_loop, daemon=True).start()


def generate_frames():
    last_sent = None
    while True:
        with _jpeg_lock:
            jpeg = _latest_jpeg
        if jpeg is None or jpeg is last_sent:
            time.sleep(0.01)
            continue
        last_sent = jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')


@app.route('/')
def index():
    return '''
    <html>
    <head>
        <style>
            body { background:#000; color:#fff; font-family:Arial; margin:0; }
            .container { display:flex; flex-direction:column; align-items:center; padding:20px; }
            img { width:640px; border:2px solid #fff; }
            #status { margin-top:20px; font-size:24px; padding:10px 20px;
                     background:#222; border-radius:8px; min-width:300px; text-align:center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>歩行検出システム</h2>
            <img src="/video">
            <div id="status">検出待機中...</div>
        </div>
        <script>
            setInterval(() => {
                fetch('/status')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('status').innerText = data.message;
                    });
            }, 200);
        </script>
    </body>
    </html>
    '''


@app.route('/video')
def video():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(get_status())


if __name__ == '__main__':
    src = f"動画: {VIDEO_SOURCE}" if isinstance(VIDEO_SOURCE, str) else "カメラ"
    print(f"サーバー起動: http://localhost:5000  入力={src}")
    app.run(host='0.0.0.0', port=5000, debug=False)
