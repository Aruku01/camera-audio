import mediapipe as mp
from flask import Flask, Response, jsonify
import cv2
from picamera2 import Picamera2
from detection import detect, get_status, mp_pose, _state, _ground_y, _air_min_y
from audio import play
from led import flash
import time

app = Flask(__name__)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)


def generate_frames():
    while True:
        # picamera2のRGB888はメモリ上BGR順で渡されるため、
        # MediaPipe用にBGR→RGBへ変換し、表示用はそのまま使う
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results, text, landed = detect(frame_rgb)

        for side in landed:
            play(side)
        if landed:
            flash()

        display = frame.copy()
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                display,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(display, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # デバッグ: 各足首のY値と状態を表示
        for i, (side, label) in enumerate([('L', 'Left'), ('R', 'Right')]):
            gy = _ground_y[side]
            ay = _air_min_y[side]
            st = _state[side]
            gy_str = f"{gy:.2f}" if gy is not None else "--"
            dbg = f"{label}: state={st} gnd={gy_str} air={ay:.2f}"
            cv2.putText(display, dbg, (10, 60 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        _, buffer = cv2.imencode('.jpg', display)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


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
    print("サーバー起動: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
