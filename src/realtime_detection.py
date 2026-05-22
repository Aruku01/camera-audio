import cv2
import mediapipe as mp
from picamera2 import Picamera2
import time

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    print("歩行検出開始 (Ctrl+Cで終了)")

    STEP_THRESHOLD = 0.1  # 踏み出しと判断するx座標の差

    try:
        while True:
            frame = picam2.capture_array()
            results = pose.process(frame)

            if results.pose_landmarks:
                left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

                diff = abs(left_ankle.x - right_ankle.x)

                if diff > STEP_THRESHOLD:
                    if left_ankle.x < right_ankle.x:
                        print(f"左足が前に出ています (差: {diff:.2f})")
                    else:
                        print(f"右足が前に出ています (差: {diff:.2f})")
                else:
                    print(f"両足が揃っています (差: {diff:.2f})")
            else:
                print("人物未検出")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("終了します")

    picam2.stop()
    pose.close()

if __name__ == "__main__":
    main()