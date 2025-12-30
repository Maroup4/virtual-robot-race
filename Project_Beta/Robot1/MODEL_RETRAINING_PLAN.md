# モデル再学習プラン
## 左バイアス問題の根本解決

**日時:** 2025-12-28
**目標:** 左に落ちる問題を解決し、安定した2周完走を実現

---

## 📊 現状分析

### トレーニングデータの問題

| 指標 | 値 | 評価 |
|-----|-----|------|
| 総フレーム数 | 32,994 | 十分 |
| 右ステア比率 | 64.5% | 多すぎる |
| 左ステア比率 | 22.8% | 少なすぎる |
| 平均ステア | +0.1849 rad | 右バイアス |

### 問題のあるラン

**極端な左バイアス:**
- `run_20251214_091114`: -0.3334 rad (1,576フレーム) ← **要削除**
- `run_20251214_090634`: -0.2581 rad (962フレーム) ← **要削除**

これらのランはクラッシュしたり、コース外を走行した可能性が高い。

**極端な右バイアス:**
- `run_20251214_093746`: +0.3808 rad (1,576フレーム)
- `run_20251214_095849`: +0.3650 rad (1,577フレーム)

これらは直線主体の走行の可能性。

---

## 🔧 ステップ1: データクリーニング

### 1.1 問題のあるランを除外

```bash
cd Robot1/training_data

# 極端なバイアスのランを移動(削除ではなくバックアップ)
mkdir -p ../excluded_runs
move run_20251214_091114 ../excluded_runs/
move run_20251214_090634 ../excluded_runs/
```

### 1.2 クリーニング後のバランスを確認

```bash
cd Robot1
../.venv/Scripts/python scripts/analyze_steering_bias.py
```

期待される結果:
- 左バイアスの極端なランが削除される
- 全体の平均ステアが改善される

---

## 🔧 ステップ2: データオーグメンテーション

左右のバランスを完全に整えるため、画像の水平反転を行います。

### 2.1 オーグメンテーションスクリプトの作成

[scripts/augment_training_data.py](scripts/augment_training_data.py) を作成済み(次のセクションで実装)

### 2.2 実行

```bash
cd Robot1
../.venv/Scripts/python scripts/augment_training_data.py
```

これにより:
- すべての画像を水平反転
- ステアリング角を反転(negate)
- 左右完全バランスのデータセットを生成

**効果:**
- データセット: 32,994 → 約65,000フレーム(2倍)
- 左右バランス: 強制的に50/50

---

## 🔧 ステップ3: モデル再学習

### 3.1 学習設定の確認

[train_model.py](train_model.py) の設定を確認:

```python
# 推奨設定
EPOCHS = 50  # より多くのエポック
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # 小さめに
```

### 3.2 再学習の実行

```bash
cd Robot1
../.venv/Scripts/python train_model.py --epochs 50 --batch-size 32
```

**学習時間見込み:**
- GPU使用時: 約30-60分
- CPU使用時: 数時間

### 3.3 学習中の監視

ターミナルでloss低下を確認:
```
Epoch 10/50: loss=0.0234
Epoch 20/50: loss=0.0156
Epoch 30/50: loss=0.0098  <- 良好
```

---

## 🔧 ステップ4: モデル検証

### 4.1 推論テスト

```bash
cd Robot1
../.venv/Scripts/python test_inference.py
```

ランダムな画像で推論を実行し、ステアリング出力を確認。

### 4.2 実走行テスト

```bash
cd ..
../.venv/Scripts/python main.py
```

**確認事項:**
- スタート成功
- 左に落ちないか
- 2周完走できるか

### 4.3 ログ分析

```bash
cd Robot1/training_data
# 最新ランのメタデータを確認
python -c "
import pandas as pd
import glob
latest = max(glob.glob('run_*/metadata.csv'))
df = pd.read_csv(latest)
racing = df[df['status'] != 'StartSequence']
print(f'Mean steer: {racing[\"steer_angle\"].mean():.4f}')
print(f'Min steer: {racing[\"steer_angle\"].min():.4f}')
print(f'Max steer: {racing[\"steer_angle\"].max():.4f}')
"
```

**期待値:**
- 平均ステア: -0.01 ~ +0.01 (ほぼ中立)
- 左右の最大値がバランス

---

## 📋 チェックリスト

### データクリーニング
- [ ] 極端なバイアスランを特定
- [ ] 問題のあるランを除外
- [ ] クリーニング後のバランスを確認

### データオーグメンテーション
- [ ] augment_training_data.py を実装
- [ ] オーグメンテーション実行
- [ ] augmented/ ディレクトリ作成確認
- [ ] 左右バランス50/50確認

### モデル再学習
- [ ] train_model.py の設定確認
- [ ] 再学習実行(50 epochs)
- [ ] loss収束確認
- [ ] models/model.pth 更新確認

### 検証
- [ ] 推論テスト実行
- [ ] 実走行テスト(2周完走)
- [ ] ログ分析(ステアリングバランス確認)
- [ ] 改善が見られれば完了!

---

## 🚨 もし改善しない場合

### プランB: モデルアーキテクチャの変更

現在のモデル(ResNet18ベース)が不十分な可能性:

1. **より深いモデルを試す**:
   - ResNet34, ResNet50
   - EfficientNet

2. **入力画像サイズを大きくする**:
   - 現在: 64x64 → 128x128

3. **時系列情報を追加**:
   - LSTM/GRUレイヤーを追加
   - 過去3-5フレームを入力

### プランC: ハイブリッドアプローチ

ルールベースとAIの組み合わせ:

```python
# ai_control_strategy.py
# コース端検出(簡易版)
if pos_x < -0.85:
    adjusted_steer = max(adjusted_steer, 0.10)  # 強制右ステア
elif pos_x > 0.85:
    adjusted_steer = min(adjusted_steer, -0.10)  # 強制左ステア
```

---

## 💡 期待される成果

### データクリーニング後
- 極端なバイアスランの影響を排除
- より一貫したトレーニングデータ

### オーグメンテーション後
- 完璧な左右バランス(50/50)
- データ量2倍(汎化性能向上)

### 再学習後
- 左に落ちる問題の解消
- 安定した2周完走
- よりスムーズなステアリング

---

## 📚 参考コマンド

### クイックリファレンス

```bash
# データ分析
cd Robot1
../.venv/Scripts/python scripts/analyze_steering_bias.py

# データオーグメンテーション
../.venv/Scripts/python scripts/augment_training_data.py

# モデル再学習
../.venv/Scripts/python train_model.py --epochs 50

# テスト走行
cd ..
../.venv/Scripts/python main.py
```

---

## ✅ 次のステップ

1. まず、[scripts/augment_training_data.py](scripts/augment_training_data.py) を実装
2. データオーグメンテーションを実行
3. モデルを再学習
4. テスト走行

準備ができたら、次のコマンドを実行してください!

Good luck, Engineer! 🏁
