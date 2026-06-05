# CLAUDE.md — camera-audio

## プロジェクト概要

Raspberry Pi のカメラを使い、MediaPipe の姿勢推定モデルでリアルタイムに人間の歩行を検出するシステム。
検出結果（どちらの足が前か、両足が揃っているか）を Flask の HTTP ストリーミングサーバー経由でブラウザに配信する。

## ファイル構成

```
camera-audio/
├── src/
│   ├── server.py            # Flask サーバー。カメラ映像を MJPEG でストリーミングし /status で歩行状態を返す
│   ├── detection.py         # MediaPipe Pose による姿勢推定と歩行状態判定ロジック
│   ├── realtime_detection.py# スタンドアロンのリアルタイム検出スクリプト（サーバーなし、ターミナル出力）
│   └── camera.py            # カメラ動作確認用スクリプト。1フレーム撮影して test_capture.jpg に保存
├── test_capture.jpg         # camera.py で生成されるテスト画像
├── requirements.txt         # 依存パッケージ（現在空）
├── README.md
└── CLAUDE.md                # このファイル
```

## アーキテクチャ

```
Picamera2 → detect() → MediaPipe Pose
                ↓
        current_status (グローバル辞書)
                ↓
Flask /video  → MJPEG ストリーム（ランドマーク描画済み）
Flask /status → JSON {"message": "左足が前 (差: 0.15)" など}
Flask /       → ブラウザ UI（img + status を 200ms ポーリング）
```

## 歩行判定ロジック

`detection.py:STEP_THRESHOLD = 0.1` を閾値として、左右足首の X 座標差で判定する。

| 条件 | 出力 |
|---|---|
| `diff > 0.1` かつ左足首が左側 | 左足が前 |
| `diff > 0.1` かつ右足首が左側 | 右足が前 |
| `diff <= 0.1` | 両足揃い |
| ランドマーク未検出 | 人物未検出 |

## 使用技術

| ライブラリ | バージョン | 用途 |
|---|---|---|
| picamera2 | 0.3.31 | Raspberry Pi カメラ制御 |
| MediaPipe | 0.10.18 | 姿勢推定（33 ランドマーク） |
| OpenCV | 4.13.0 | 画像処理・JPEG エンコード・描画 |
| Flask | 2.2.2 | HTTP サーバー・MJPEG ストリーミング |
| NumPy | 1.26.4 | 配列処理（OpenCV 依存） |

## 開発環境

- **ハードウェア**: Raspberry Pi（ARM64）
- **OS**: Linux 6.12 (Raspberry Pi OS)
- **Python**: 3.11.2
- **仮想環境**: `.venv/`（プロジェクトルートの venv）
- **サーバー起動ポート**: 5000（`http://0.0.0.0:5000`）

## 起動方法

```bash
cd /home/pi/camera-audio
source .venv/bin/activate

# Webサーバー起動（推奨）
python src/server.py

# スタンドアロン検出（ターミナルのみ）
python src/realtime_detection.py

# カメラ動作確認
python src/camera.py
```

サーバー起動後、同一ネットワーク内のブラウザから `http://<RaspberryPiのIP>:5000` にアクセスする。

## 注意事項

- `server.py` はモジュール検索パスの都合上、`src/` ディレクトリ内から実行するか、`PYTHONPATH=src` を設定して実行する必要がある（`from detection import ...` の相対インポートのため）。
- `picamera2` は Raspberry Pi 専用ライブラリのため、他の環境では動作しない。
- MediaPipe のモデルは初回起動時にロードされるため、最初のフレーム処理に数秒かかる場合がある。
