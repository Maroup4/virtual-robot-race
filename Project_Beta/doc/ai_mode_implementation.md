# AI Mode (MODE_NUM=4) 実装ドキュメント

## 概要

2024年11月30日に実装されたEnd-to-End（E2E）AIモードの技術ドキュメント。
イミテーションラーニング（模倣学習）により、キーボード操作のデータから自律走行を学習する。

## アーキテクチャ

### ファイル構成

```
Robot1/
├── model.py              # CNNモデル定義 (DrivingNetwork)
├── train_model.py        # トレーニングスクリプト
├── inference_input.py    # 推論エンジン（フレームワーク）
├── ai_control_strategy.py # 戦略設定（ユーザーカスタマイズ用）
├── models/
│   └── model.pth         # 学習済みモデル
└── training_data/        # トレーニングデータ
    └── run_YYYYMMDD_HHMMSS/
        ├── metadata.csv
        └── images/
```

### モデルアーキテクチャ (DrivingNetwork)

```
入力:
  - RGB画像: [B, 3, 224, 224] (正規化済み)
  - SOC: [B, 1] (バッテリー残量 0.0〜1.0)

CNN Feature Extractor:
  - Block 1: Conv2d(3→32, k=5, s=2) + BN + ReLU  → 112x112
  - Block 2: Conv2d(32→64, k=5, s=2) + BN + ReLU → 56x56
  - Block 3: Conv2d(64→128, k=5, s=2) + BN + ReLU → 28x28
  - Block 4: Conv2d(128→256, k=5, s=2) + BN + ReLU → 14x14
  - Global Average Pooling → [B, 256]

MLP Head:
  - Linear(257→128) + ReLU + Dropout(0.3)
  - Linear(128→64) + ReLU + Dropout(0.2)
  - Linear(64→2)  # [drive_torque, steer_angle]

出力:
  - drive_torque: -1.0〜1.0
  - steer_angle: -0.785〜0.785 rad (±45度)

パラメータ数: 1,120,450
```

## 戦略設定 (ai_control_strategy.py)

### モード選択

| モード | 説明 |
|--------|------|
| `hybrid` | ルールベースのスタート検出 + AI走行（推奨） |
| `pure_e2e` | 完全AIモード（スタート検出もAI） |

### 主要設定

```python
STRATEGY = "hybrid"           # 戦略選択
HYBRID_START_DETECTION = True # スタート信号検出にルールベースを使用
```

### カスタマイズ可能な関数

1. **should_wait_for_start(pil_img, race_started)**
   - スタート待機判定
   - Trueを返すと出力ゼロ

2. **adjust_output(drive, steer, pil_img, soc)**
   - AI出力の後処理
   - スムージング、エネルギー節約などに使用可能

3. **on_race_start()**
   - レース開始時のコールバック

## トレーニング

### データ収集

1. `robot_config.txt`で`MODE_NUM=1`（キーボード）、`DATA_SAVE=1`に設定
2. `python main.py`でキーボード操作でコースを走行
3. データは`training_data/run_YYYYMMDD_HHMMSS/`に保存

### トレーニング実行

```bash
cd Robot1
python train_model.py --epochs 100
```

### オプション

| オプション | デフォルト | 説明 |
|------------|------------|------|
| `--data` | training_data | データディレクトリ |
| `--epochs` | 100 | エポック数 |
| `--batch-size` | 32 | バッチサイズ |
| `--lr` | 1e-4 | 学習率 |
| `--val-split` | 0.2 | 検証データ比率 |
| `--output` | models/model.pth | 出力パス |
| `--device` | auto | cuda/cpu/auto |

### データフィルタリング

`train_model.py`は自動的に以下を除外:
- `status == "StartSequence"` のデータ（レース開始前の静止状態）

これにより、モデルが「走行中」のデータのみから学習する。

## 問題解決履歴

### 1. False Start問題

**症状**: 赤信号中にモデルが小さな正のトルクを出力

**原因**:
- トレーニングデータにStartSequenceが含まれていた
- モデルが完全なゼロ出力を学習できていなかった

**解決策**:
- Hybrid戦略を実装（ルールベースのスタート検出）
- `ai_control_strategy.py`で戦略を切り替え可能に

### 2. 低トルク出力問題

**症状**: Drive出力が0.03〜0.05程度で車がほぼ動かない

**原因**:
- AI失敗走行データ（drive≈0）がトレーニングデータに混入
- データの77%がdrive < 0.1だった

**解決策**:
- 不良データを`training_data_bad/`に移動
- キーボード操作データのみで再トレーニング
- トレーニングデータの品質管理が重要

### データ品質チェック

```python
# データ分布確認スクリプト例
import pandas as pd
from pathlib import Path

data_base = Path('training_data')
for d in data_base.iterdir():
    if d.name.startswith('run_'):
        df = pd.read_csv(d / 'metadata.csv')
        df = df[df['status'] != 'StartSequence']
        print(f"{d.name}: drive mean={df['drive_torque'].mean():.3f}")
```

**良いデータの目安**: drive_torque平均 > 0.3

## GPU トレーニング

### CUDA セットアップ

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 確認

```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # GPU名
```

## モデルアーカイブ

保存されたモデルファイル:

| ファイル名 | 説明 |
|-----------|------|
| `model.pth` | 現在使用中 |
| `model_YYYYMMDD_HHMMSS_valloss0.XXXX.pth` | バックアップ |

## 使用方法

### 推論実行

1. `robot_config.txt`で`MODE_NUM=4`に設定
2. `python main.py`を実行

### 設定例

```ini
# Robot1/robot_config.txt
ROBOT_ID=R1
NAME=Player0000
MODE_NUM=4        # AIモード
RACE_FLAG=1
DATA_SAVE=1       # データ保存（さらなる学習用）
AUTO_MAKE_VIDEO=0
```

## 今後の改善案

1. **データ拡張**: より多様なキーボードデータの収集
2. **モデル改良**: ResNetなどの事前学習済みバックボーンの使用
3. **戦略拡張**: ピットストップ検出、コーナリング最適化
4. **リアルタイム調整**: `adjust_output()`でのトルクブースト実装
