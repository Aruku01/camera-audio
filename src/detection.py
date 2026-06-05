import mediapipe as mp

mp_pose = mp.solutions.pose
STEP_THRESHOLD = 0.1

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

current_status = {"message": "検出待機中..."}

def detect(frame):
    results = pose.process(frame)

    if results.pose_landmarks:
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        diff = abs(left_ankle.x - right_ankle.x)

        if diff > STEP_THRESHOLD:
            if left_ankle.x < right_ankle.x:
                current_status["message"] = f"左足が前 (差: {diff:.2f})"
            else:
                current_status["message"] = f"右足が前 (差: {diff:.2f})"
        else:
            current_status["message"] = f"両足揃い (差: {diff:.2f})"
    else:
        current_status["message"] = "人物未検出"

    return results, current_status["message"]