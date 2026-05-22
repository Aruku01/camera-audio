import cv2
import mediapipe as mp

def main():
    # MediaPipeの手検出の準備
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,      # 静止画モード
        max_num_hands=2,
        min_detection_confidence=0.3  # 感度を下げる（デフォルトは0.5）
    )
    # 先ほど撮影した画像を読み込む
    image = cv2.imread("test_capture.jpg")
    
    if image is None:
        print("画像が読み込めませんでした")
        return

    print(f"画像読み込み成功: {image.shape}")

    # BGRからRGBに変換（MediaPipeはRGBを使用）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    # 手の検出
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        print(f"手を検出しました！検出数: {len(results.multi_hand_landmarks)}")
    
        # ランドマークの座標を表示
        for hand_landmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand_landmarks.landmark):
                print(f"ランドマーク {id}: x={landmark.x:.2f}, y={landmark.y:.2f}")
    else:
        print("手が検出されませんでした")

    hands.close()

if __name__ == "__main__":
    main()

