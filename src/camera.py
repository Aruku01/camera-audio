import cv2
from picamera2 import Picamera2

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()

    print("カメラ起動成功")

    frame = picam2.capture_array()
    print(f"フレーム取得成功: {frame.shape}")

    # 画像を保存
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    cv2.imwrite("test_capture.jpg", frame_rgb)
    print("画像を保存しました: test_capture.jpg")
    picam2.stop()
    print("カメラを閉じました")

if __name__ == "__main__":
    main()
