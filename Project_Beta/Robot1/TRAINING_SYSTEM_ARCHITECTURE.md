# VRR Beta AI学習システム - 完全アーキテクチャ解説

**日時:** 2025-12-28
**目的:** AIモデルの学習プロセスとシステム全体の理解

---

## 📋 目次

1. [システム全体像](#システム全体像)
2. [学習プロセスに関わるスクリプト](#学習プロセスに関わるスクリプト)
3. [推論(実行時)に関わるスクリプト](#推論実行時に関わるスクリプト)
4. [データ処理に関わるスクリプト](#データ処理に関わるスクリプト)
5. [詳細解説: 各スクリプトの役割](#詳細解説-各スクリプトの役割)
6. [データフロー図](#データフロー図)
7. [Q&A: よくある質問](#qa-よくある質問)

---

## システム全体像

VRR Beta AI学習システムは、以下の3つの主要フェーズで構成されています:

```
┌─────────────────────────────────────────────────────────────────┐
│                    VRR Beta AI System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: データ収集 (Data Collection)                          │
│    └─> マニュアル走行 → training_data/ 保存                    │
│                                                                 │
│  Phase 2: データ処理 & 学習 (Data Processing & Training)        │
│    ├─> データ分析 (analyze_steering_bias.py)                   │
│    ├─> データオーグメンテーション (augment_training_data.py)   │
│    ├─> データ結合 (combine_training_data.py)                   │
│    └─> モデル学習 (train_model.py + model.py)                  │
│                                                                 │
│  Phase 3: 推論 & 走行 (Inference & Racing)                      │
│    ├─> モデル読み込み (inference_input.py)                     │
│    ├─> 後処理 (ai_control_strategy.py)                         │
│    └─> Unity実行 (main.py経由)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 学習プロセスに関わるスクリプト

### 🎯 コアスクリプト（必須）

| ファイル名 | 役割 | 依存関係 |
|-----------|------|---------|
| **train_model.py** | メイン学習スクリプト | model.py |
| **model.py** | ニューラルネットワーク定義 | なし |

### 🔧 データ処理スクリプト（学習前準備）

| ファイル名 | 役割 | 使用タイミング |
|-----------|------|--------------|
| **analyze_steering_bias.py** | トレーニングデータの左右バイアス分析 | 学習前・後 |
| **augment_training_data.py** | 画像を水平反転してデータ拡張 | 学習前 |
| **combine_training_data.py** | 元データ + 反転データを結合 | 学習前 |

---

## 推論(実行時)に関わるスクリプト

### 🚗 AIモード実行時に使用されるスクリプト

| ファイル名 | 役割 | 実行タイミング |
|-----------|------|--------------|
| **inference_input.py** | AI推論エンジン（モデルロード & 推論実行） | 毎フレーム |
| **ai_control_strategy.py** | 後処理レイヤー（堅牢性改善） | 毎フレーム |
| **model.py** | ニューラルネットワーク定義（学習時と同じ） | ロード時 |

### 🎮 その他の入力モード

| ファイル名 | 役割 |
|-----------|------|
| **keyboard_input.py** | キーボード手動操作 |
| **rule_based_input.py** | ルールベースアルゴリズム |
| **table_input.py** | テーブルデータ入力 |

---

## データ処理に関わるスクリプト

### 📊 分析 & デバッグツール

| ファイル名 | 役割 |
|-----------|------|
| **verify_model_input.py** | モデルへの入力データ検証 |
| **test_robustness.py** | ai_control_strategy.pyの単体テスト |

### 🎬 その他

| ファイル名 | 役割 |
|-----------|------|
| **make_promotion_video.py** | プロモーションビデオ作成 |

---

## 詳細解説: 各スクリプトの役割

### 1. train_model.py（メイン学習スクリプト）

**目的**: トレーニングデータからニューラルネットワークを学習

#### 主要クラス・関数

```python
class DrivingDataset(Dataset):
    """
    PyTorchのDatasetクラス
    - training_data/内のrun_*/metadata.csvを読み込み
    - 各フレームの画像 + SOC → drive_torque, steer_angle のペアを作成
    - StartSequenceフレームは除外（レース中のデータのみ使用）
    """

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    1エポック分の学習を実行
    - 全トレーニングデータを1回ずつ学習
    - Loss計算 → 勾配計算 → パラメータ更新
    """

def validate(model, dataloader, criterion, device):
    """
    検証データでモデル性能を評価
    - Lossを計算（パラメータ更新なし）
    - 過学習チェックに使用
    """

def main():
    """
    メイン学習ループ
    1. データセット作成
    2. モデル初期化
    3. 各エポックで学習 & 検証
    4. ベストモデルを保存
    """
```

#### 学習フロー

```
┌──────────────────────────────────────────────────┐
│ 1. データロード                                   │
│    - training_data_combined/内の全run読み込み    │
│    - 画像変換（224x224リサイズ、正規化）         │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ 2. データセット分割                               │
│    - トレーニング: 80%                            │
│    - 検証: 20%                                    │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ 3. モデル初期化                                   │
│    - DrivingNetwork（CNN + MLP）                 │
│    - Adam optimizer                              │
│    - MSE Loss（平均二乗誤差）                     │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ 4. エポックループ（50回繰り返し）                 │
│    For each epoch:                               │
│      a) トレーニング                              │
│         - 全データをバッチ処理                    │
│         - Loss計算                                │
│         - 勾配降下法でパラメータ更新              │
│      b) 検証                                      │
│         - 検証データでLoss計算                    │
│      c) ベストモデル保存                          │
│         - 検証Lossが最小なら保存                  │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ 5. 最終モデル保存                                 │
│    - models/model.pth に保存                     │
└──────────────────────────────────────────────────┘
```

#### 使用するデータ

**入力 (X)**:
- 画像: RGB 224x224ピクセル
- SOC: バッテリー残量 (0.0 ~ 1.0)

**出力 (Y)**:
- drive_torque: 駆動トルク (-1.0 ~ +1.0)
- steer_angle: ステアリング角度 (radians)

#### 学習パラメータ

```python
--data training_data_combined  # データディレクトリ
--epochs 50                     # エポック数
--batch-size 32                 # バッチサイズ（デフォルト）
--lr 0.001                      # 学習率（デフォルト）
--val-split 0.2                 # 検証データ割合（デフォルト）
--device cuda                   # GPU使用
```

---

### 2. model.py（ニューラルネットワーク定義）

**目的**: CNNベースのEnd-to-Endドライビングモデル定義

#### ネットワーク構造

```
┌─────────────────────────────────────────────────────────┐
│                   DrivingNetwork                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: Image [B, 3, 224, 224] + SOC [B, 1]            │
│                                                         │
│  ┌─────────────────────────────────────────────┐       │
│  │ CNN Feature Extractor                       │       │
│  ├─────────────────────────────────────────────┤       │
│  │ Conv2d(3→32)  + BN + ReLU  [224→112]        │       │
│  │ Conv2d(32→64) + BN + ReLU  [112→56]         │       │
│  │ Conv2d(64→128) + BN + ReLU [56→28]          │       │
│  │ Conv2d(128→256) + BN + ReLU [28→14]         │       │
│  │ AdaptiveAvgPool2d           [14→1]          │       │
│  └─────────────────────────────────────────────┘       │
│                    ↓                                    │
│               Features [B, 256]                         │
│                    ↓                                    │
│  ┌─────────────────────────────────────────────┐       │
│  │ Concatenate with SOC                        │       │
│  │ [B, 256] + [B, 1] → [B, 257]                │       │
│  └─────────────────────────────────────────────┘       │
│                    ↓                                    │
│  ┌─────────────────────────────────────────────┐       │
│  │ MLP Head                                    │       │
│  ├─────────────────────────────────────────────┤       │
│  │ Linear(257→128) + ReLU + Dropout(0.3)       │       │
│  │ Linear(128→64)  + ReLU + Dropout(0.2)       │       │
│  │ Linear(64→2)                                │       │
│  └─────────────────────────────────────────────┘       │
│                    ↓                                    │
│  Output: [drive_torque, steer_angle] [B, 2]            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### パラメータ数

- **CNNパラメータ**: 約1.5M（150万）
- **MLPパラメータ**: 約40K（4万）
- **合計**: 約1.54M（154万パラメータ）

#### 設計思想

1. **CNN部分**: 画像から視覚的特徴を抽出
   - コース形状、車の位置、障害物などを認識
   - BatchNormで学習安定化
   - Global Average Poolingで過学習抑制

2. **MLP部分**: 特徴 + SOC から操作を予測
   - Dropoutで汎化性能向上
   - 2つの出力（drive, steer）を同時に予測

---

### 3. inference_input.py（AI推論エンジン）

**目的**: 学習済みモデルを読み込み、リアルタイムで推論実行

#### 主要処理フロー

```python
# 1. モデルロード（初回のみ）
def preload_model():
    """
    学習済みモデル(model.pth)をロード
    - CPUまたはGPUに配置
    - 評価モード(eval)に設定
    """

# 2. 画像前処理
def preprocess_image(pil_image):
    """
    Unity画像を224x224にリサイズ & 正規化
    - トレーニング時と同じ変換を適用
    """

# 3. 推論実行
def inference(robot_id, race_started):
    """
    毎フレーム呼ばれる推論関数
    1. 最新画像とSOCを取得
    2. モデルに入力
    3. drive, steerを予測
    4. ai_control_strategy.pyで後処理
    5. Unityへ送信
    """
```

#### データフロー（実行時）

```
Unity Server
    ↓
Camera Image (480x270) + SOC
    ↓
inference_input.py
    ├─> 画像リサイズ (224x224)
    ├─> 正規化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ├─> Tensor化
    └─> モデル推論
            ↓
    Raw Output: [drive, steer]
            ↓
ai_control_strategy.py
    ├─> スムージング（Low-pass filter）
    ├─> レート制限（急激な変化を抑制）
    ├─> スタートブースト（条件付き）
    └─> コーナー対応トルク制限
            ↓
    Final Output: [drive, steer]
            ↓
Unity Server → 車両制御
```

---

### 4. ai_control_strategy.py（後処理レイヤー）

**目的**: モデルの生出力を実際のレース環境に適した制御に変換

#### 実装されている堅牢性改善（3つ）

##### (A) 条件付きスタートブースト

```python
# スタート直後22フレーム（約1.1秒）のみ
# かつ、ステアリング要求が小さい時（abs(steer) <= 0.10）
# → MIN_DRIVE_TORQUE (0.32)を保証

if race_frame_count <= 22 and abs(steer) <= 0.10:
    drive = max(drive, 0.32)
```

**理由**: スタート直後の加速不足を防ぐが、コーナーでは無効化

##### (B) ステアリングレート制限（Hard Safety）

```python
# フレーム間のステア変化を0.03 rad/frameに制限
delta_steer = steer - prev_steer
if abs(delta_steer) > 0.03:
    delta_steer = clamp(delta_steer, -0.03, 0.03)
    steer = prev_steer + delta_steer
```

**理由**: 急激なステアリング変化でコースアウトを防ぐ

##### (C) コーナー対応ドライブトルク制限

```python
# ステアリング角が大きい時、トルクを制限
if abs(steer) >= 0.20:
    # 線形補間で0.20→0.50の範囲でトルク減少
    # 最大0.55 → 最小0.30
    t = (abs(steer) - 0.20) / (0.50 - 0.20)
    drive_cap = 0.55 + t * (0.30 - 0.55)
    drive = min(drive, drive_cap)
```

**理由**: コーナーでの速度超過による転倒・コースアウト防止

#### ハイブリッドモード vs Pure E2Eモード

```python
STRATEGY = "hybrid"  # または "pure_e2e"

if STRATEGY == "hybrid":
    # スタート検出はルールベース（perception_Startsignal.py）
    # 走行はAIモデル
    race_started = detect_start_signal(image)

elif STRATEGY == "pure_e2e":
    # すべてAIモデルに任せる
    # （スタート検出もモデルが学習）
    race_started = True  # 常にレース状態
```

---

### 5. augment_training_data.py（データオーグメンテーション）

**目的**: 画像を水平反転してステアリング角を反転し、左右バランスを改善

#### 処理内容

```python
for run in training_data/:
    for frame in metadata.csv:
        # 1. 画像を水平反転
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. ステアリング角を反転
        flipped_steer = -original_steer

        # 3. yaw（車両方向）も反転
        flipped_yaw = -original_yaw

        # 4. 保存
        save_to(training_data_augmented/)
```

#### 効果

**Before（元データ）**:
- 左ステア: 64.4%
- 右ステア: 23.1%
- 平均ステア: -0.2106 rad（左バイアス）

**After（反転データ）**:
- 左ステア: 23.1%（反転）
- 右ステア: 64.4%（反転）
- 平均ステア: +0.2106 rad（右バイアス）

**Combined（元 + 反転）**:
- 左ステア: 43.4%
- 右ステア: 44.0%
- 平均ステア: **0.0034 rad（完璧なバランス！）**

---

### 6. combine_training_data.py（データ結合）

**目的**: 元データと反転データを1つのディレクトリに統合

#### 処理内容

```python
# 元データをそのままコピー
training_data/ → training_data_combined/

# 反転データを "_flipped" サフィックス付きでコピー
training_data_augmented/ → training_data_combined/*_flipped/
```

#### 結果

```
training_data_combined/
├── run_20251214_065243/          # 元データ
├── run_20251214_065243_flipped/  # 反転データ
├── run_20251214_065331/
├── run_20251214_065331_flipped/
...
```

- **総run数**: 94 runs（元52 + 反転42）
- **総フレーム数**: 65,202フレーム（約2倍）
- **左右バランス**: 完璧（50/50）

---

### 7. analyze_steering_bias.py（バイアス分析）

**目的**: トレーニングデータの左右ステアリング分布を分析

#### 出力情報

1. **全体統計**
   - 総フレーム数
   - 左/右/ニュートラル比率
   - 平均ステア
   - 最小/最大ステア

2. **run別統計**
   - 各runの平均ステア
   - 左右比率
   - バイアス判定

3. **推奨アクション**
   - バイアスが大きい場合の対処法

#### 使用例

```bash
# 元データ分析
python scripts/analyze_steering_bias.py --input training_data

# 反転データ分析
python scripts/analyze_steering_bias.py --input training_data_augmented

# 結合データ分析
python scripts/analyze_steering_bias.py --input training_data_combined
```

---

## データフロー図

### 全体データフロー

```
┌───────────────────────────────────────────────────────────────┐
│ Phase 1: データ収集                                            │
└───────────────────────────────────────────────────────────────┘
                              ↓
         Keyboard入力でマニュアル走行
                              ↓
      training_data/run_YYYYMMDD_HHMMSS/
          ├── metadata.csv
          ├── images/frame_*.jpg
          └── unity_log.txt
                              ↓
┌───────────────────────────────────────────────────────────────┐
│ Phase 2: データ処理                                            │
└───────────────────────────────────────────────────────────────┘
                              ↓
      1. analyze_steering_bias.py
         └─> 左右バイアスを確認（-0.2106 rad）
                              ↓
      2. augment_training_data.py
         └─> 画像反転 & ステア反転
         └─> training_data_augmented/ 作成
                              ↓
      3. combine_training_data.py
         └─> 元データ + 反転データを結合
         └─> training_data_combined/ 作成（94 runs）
                              ↓
┌───────────────────────────────────────────────────────────────┐
│ Phase 3: モデル学習                                            │
└───────────────────────────────────────────────────────────────┘
                              ↓
      train_model.py --data training_data_combined --epochs 50
                              ↓
      ┌─────────────────────────────────────┐
      │ DrivingDataset                      │
      │ - 65,202フレーム読み込み            │
      │ - 画像224x224リサイズ & 正規化       │
      │ - train/val split (80/20)          │
      └─────────────────────────────────────┘
                              ↓
      ┌─────────────────────────────────────┐
      │ DrivingNetwork                      │
      │ - CNN (4 blocks) + MLP              │
      │ - 1.54M parameters                  │
      └─────────────────────────────────────┘
                              ↓
      ┌─────────────────────────────────────┐
      │ Training Loop (50 epochs)           │
      │ - Batch size: 32                    │
      │ - Optimizer: Adam (lr=0.001)        │
      │ - Loss: MSE                         │
      │ - GPU: NVIDIA RTX 3060              │
      └─────────────────────────────────────┘
                              ↓
      models/model.pth（学習済みモデル）
                              ↓
┌───────────────────────────────────────────────────────────────┐
│ Phase 4: 推論 & 走行                                           │
└───────────────────────────────────────────────────────────────┘
                              ↓
      main.py (MODE=ai)
         └─> inference_input.py
                ├─> モデルロード (model.pth)
                ├─> 画像取得 (Unity)
                ├─> 推論実行
                └─> 後処理 (ai_control_strategy.py)
                       ├─> スムージング
                       ├─> レート制限
                       ├─> スタートブースト
                       └─> トルク制限
                              ↓
      Unity Server → 車両制御
```

---

## Q&A: よくある質問

### Q1: train_model.pyはどのスクリプトを呼び出していますか？

**A**: `train_model.py`は**model.pyのみ**をインポートします。

```python
from model import DrivingNetwork
```

その他の処理（データロード、学習ループ、保存）はすべて`train_model.py`内で完結しています。

---

### Q2: 学習中に他のスクリプトは動いていますか？

**A**: いいえ。学習中は**train_model.pyのプロセス1つだけ**が動作します。

他のスクリプト（augment, combine, analyzeなど）は**学習前の準備段階**で実行され、学習開始後は使われません。

---

### Q3: inference_input.pyとai_control_strategy.pyの違いは？

**A**: 役割が異なります:

| スクリプト | 役割 | タイミング |
|-----------|------|-----------|
| **inference_input.py** | モデルロード & 推論実行 | 毎フレーム |
| **ai_control_strategy.py** | モデル出力の後処理（堅牢性改善） | 毎フレーム |

**フロー**:
```
Unity画像 → inference_input.py (推論) → ai_control_strategy.py (後処理) → Unity
```

---

### Q4: モデルはどこに保存されますか？

**A**: `Robot1/models/model.pth`

学習中は各エポック後に検証Lossが最小のモデルが保存されます。

---

### Q5: 学習にどれくらい時間がかかりますか？

**A**: 環境により異なりますが:

| 環境 | 時間（50エポック） |
|------|------------------|
| RTX 3060 Laptop GPU | 約3-4時間 |
| RTX 3090 Desktop | 約1-2時間 |
| CPU only | 約10-20時間 |

---

### Q6: オーグメンテーションは必須ですか？

**A**: 左右バイアスがある場合は**強く推奨**します。

**効果**:
- バイアス解消（-0.21 rad → 0.003 rad）
- データ量2倍（32,994 → 65,202フレーム）
- 汎化性能向上

---

### Q7: ハイブリッドモードとPure E2Eの違いは？

**A**: スタート検出の方法が異なります:

| モード | スタート検出 | メリット | デメリット |
|--------|------------|---------|----------|
| **Hybrid** | ルールベース | 確実、フライング無し | 一部手動実装 |
| **Pure E2E** | AI学習 | 完全自動 | フライングリスク |

**推奨**: ほとんどの場合、**Hybrid**が安全です。

---

### Q8: 学習中のLossはどれくらいが正常ですか？

**A**: 目安（MSE Loss）:

| エポック | Train Loss | Val Loss | 状態 |
|---------|-----------|---------|------|
| 1 | 0.5~1.0 | 0.5~1.0 | 初期 |
| 10 | 0.1~0.3 | 0.1~0.3 | 学習中 |
| 30 | 0.05~0.15 | 0.06~0.18 | 収束中 |
| 50 | 0.03~0.10 | 0.05~0.15 | 完了 |

**注意**: Val LossがTrain Lossより大幅に高い場合は過学習の可能性。

---

### Q9: 学習済みモデルの性能をテストするには？

**A**: 2つの方法があります:

1. **実走行テスト**:
   ```bash
   # robot_config.txt で MODE=ai に設定
   # main.py 実行
   ```

2. **検証スクリプト**:
   ```bash
   python scripts/verify_model_input.py
   ```

---

### Q10: モデルが左に落ち続ける場合の対処法は？

**A**: 以下の順番で確認:

1. **データバイアス確認**:
   ```bash
   python scripts/analyze_steering_bias.py --input training_data_combined
   ```
   → 平均ステアが0に近いか？

2. **後処理パラメータ調整**:
   - `ai_control_strategy.py`のパラメータ確認
   - MAX_STEER_DELTA, START_BOOST_FRAMES など

3. **モデル再学習**:
   - バランスの取れたデータで再学習

---

## まとめ

### 学習プロセスに関わるコアスクリプト

1. **train_model.py** - メイン学習スクリプト（唯一実行中のプロセス）
2. **model.py** - ニューラルネットワーク定義（train_model.pyから使用）

### 学習前の準備スクリプト

3. **analyze_steering_bias.py** - バイアス分析
4. **augment_training_data.py** - データ拡張
5. **combine_training_data.py** - データ結合

### 推論時のスクリプト

6. **inference_input.py** - 推論エンジン
7. **ai_control_strategy.py** - 後処理レイヤー

### その他のツール

8. **verify_model_input.py** - 入力検証
9. **test_robustness.py** - 単体テスト

---

**重要ポイント**:

✅ 学習中は**train_model.py 1つだけ**が動作
✅ model.pyは**定義のみ**（処理は実行しない）
✅ データ処理スクリプトは**学習前に実行**
✅ 推論スクリプトは**学習後の実行時に使用**

---

**作成日**: 2025-12-28
**対象**: VRR Beta 1.x AI学習システム
**次のステップ**: NotebookLMで理解を深める 📚
