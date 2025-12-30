# クラッシュ分析レポート
## Run: run_20251228_092313

**日時:** 2025-12-28 09:23
**結果:** 左コースアウト (pos_x=-1.024, pos_y=-0.102)
**レース時間:** 44.012秒 (Lap 0 - 未完走)

---

## 📊 データ分析

### ステアリング統計
- **平均ステア:** -0.0027 rad (左寄り)
- **最小ステア:** -0.1500 rad (大きく左)
- **最大ステア:** +0.0600 rad (右ステアが弱い)
- **標準偏差:** 0.0193 rad

### ステアリングレート制限の動作
- **最大変化量:** 0.024 rad/frame ✅ (制限値0.03以内)
- **平均変化量:** 0.0011 rad/frame
- **レート制限は正常に動作**

### クラッシュ直前の動き

| フレーム | 時間(ms) | Drive | Steer | pos_z | pos_x | yaw |
|---------|---------|-------|-------|-------|-------|-----|
| 917 | 42975 | 0.343 | -0.107 | 1.208 | -0.821 | -15.6° |
| 920 | 43139 | 0.395 | -0.076 | 1.285 | -0.847 | -17.9° |
| 925 | 43412 | 0.550 | -0.001 | 1.451 | -0.905 | -19.2° |
| **929** | **43630** | **0.550** | **+0.017** | **1.612** | **-0.961** | **-17.5°** |
| 932 | 43794 | 0.550 | +0.034 | 1.745 | -0.996 | -10.6° |
| 935 | 43957 | 0.399 | +0.050 | 1.870 | -1.019 | -6.2° |
| 936 | 44012 | 0.000 | 0.000 | 1.928 | -1.025 | -2.9° |

**観察:**
1. フレーム917-925: 左ステアを継続、徐々に左に寄る (pos_x: -0.82 → -0.90)
2. フレーム929: 右ステアに切り替え(+0.017) **← この時点でpos_x=-0.961 (手遅れ)**
3. フレーム932-935: 右ステアを強めるが、すでにコース外
4. フレーム936: 地面に落下 (pos_y=-0.102)

---

## 🔍 根本原因

### 1. **モデルの左バイアス (主要因)**

現在のニューラルネットワークは**左ステアリングに偏っています**:

- 平均ステア: -0.0027 rad (中立が0.0なので左寄り)
- 最大左ステア: -0.15 rad vs 最大右ステア: +0.06 rad
  - 左の方が **2.5倍強い**

**原因:**
- トレーニングデータが右回り楕円コースに偏っている可能性
- または、キーボード操作での左ステアが多かった

**解決策:**
- モデルの再学習が必要(左右バランスの取れたデータで)
- **または**、後処理で右バイアスを追加(応急処置)

### 2. **修正反応が遅い (副要因)**

モデルがコースアウトに気づくのが遅い:
- pos_x=-0.96になってから右ステアを開始
- トラック幅の限界を超えている

**解決策:**
- ルールベースの境界検出を追加
- または、モデルに「コース端」の認識を強化

### 3. **堅牢性改善の効果**

今回実装した3つの改善は**正常に動作しています**:

✅ **(A) 条件付きスタートブースト**
- ログに "Start boost SUPPRESSED" が記録されている
- スタート直後の問題は回避できた

✅ **(B) ステアリングレート制限**
- すべてのフレームで0.03 rad/frame以内
- 急激なステアリング変化は防止できている

✅ **(C) コーナー対応トルクキャップ**
- 大きなステアリング時にトルク削減が機能
- ただし、モデルの方向判断が間違っているため効果限定的

**問題は「モデル自体の方向判断」にあり、後処理の範囲外**

---

## 💡 次の対策オプション

### オプション1: モデル再学習 (推奨)

**目的:** 左バイアスを修正し、バランスの取れたステアリングを学習

**手順:**
1. 既存のトレーニングデータを分析:
   ```bash
   python analyze_training_data.py
   ```
2. 右ステアリングのデータを増やす:
   - 左回りコースでのデータ収集
   - データオーグメンテーション(左右反転)
3. モデルを再学習:
   ```bash
   python train_model.py --epochs 50
   ```

**効果:** 根本的な解決

---

### オプション2: 後処理で右バイアス補正 (応急処置)

`ai_control_strategy.py`に右バイアスを追加:

```python
# === EXPERIMENTAL: Right bias correction ===
# Compensate for model's left bias
RIGHT_BIAS_CORRECTION = 0.05  # rad - Add right steering bias

if adjust_output._race_frame_count > START_BOOST_FRAMES:
    # Only apply after start boost period
    adjusted_steer += RIGHT_BIAS_CORRECTION
```

**メリット:** 即座に適用可能、再学習不要
**デメリット:** すべての状況に一律適用されるため、本来右に切るべき場所でオーバーシュート
**リスク:** 今度は右に落ちる可能性

---

### オプション3: ルールベースの境界検出 (ハイブリッド強化)

pos_xを監視して、コース端に近づいたら強制的に反対方向にステア:

```python
# === EXPERIMENTAL: Boundary detection ===
TRACK_LEFT_LIMIT = -0.85   # Left boundary (conservative)
TRACK_RIGHT_LIMIT = 0.85   # Right boundary
BOUNDARY_STEER_CORRECTION = 0.15  # Strong correction

# Get position from metadata (requires integration with data manager)
# This is a conceptual example - actual implementation needs position data
if pos_x < TRACK_LEFT_LIMIT:
    # Too far left - force right steering
    adjusted_steer = max(adjusted_steer, BOUNDARY_STEER_CORRECTION)
    print(f"[Strategy] BOUNDARY WARNING: pos_x={pos_x:.2f}, forcing right steer")
elif pos_x > TRACK_RIGHT_LIMIT:
    # Too far right - force left steering
    adjusted_steer = min(adjusted_steer, -BOUNDARY_STEER_CORRECTION)
    print(f"[Strategy] BOUNDARY WARNING: pos_x={pos_x:.2f}, forcing left steer")
```

**課題:** 現在の`adjust_output()`は位置情報にアクセスできない(画像とSOCのみ)
**必要な変更:** インターフェースを拡張してpos_x, pos_zを渡す

---

## 🎯 推奨アクション

### 短期(今すぐ):
1. **オプション2を試す** - 右バイアス補正(0.03~0.05 rad)を追加
2. 数回テスト走行して効果を確認
3. もし右に落ちるようなら、バイアス量を調整

### 中期(今日~明日):
1. **トレーニングデータを分析**:
   ```bash
   python Robot1/scripts/analyze_steering_bias.py
   ```
2. 左右のステアリングバランスを確認
3. データオーグメンテーション(左右反転)を検討

### 長期(今週):
1. **モデル再学習** - バランスの取れたデータで
2. または、より高度なアーキテクチャ(attention, transformerなど)を検討

---

## 📝 まとめ

- ✅ **堅牢性改善は成功** - レート制限、スタートブースト、トルクキャップは正常動作
- ❌ **モデルの左バイアスが主要因** - これは後処理の範囲外
- 🔧 **応急処置:** 右バイアス補正を追加可能
- 🎓 **根本解決:** モデル再学習が必要

次のテスト走行の前に、オプション2の右バイアス補正を試すことをお勧めします。

---

**Engineer's Note:**
"The car is doing what the model tells it to do. The model just happens to be telling it to turn left more than it should. It's not a safety issue - it's a training data issue."

Good luck! 🏁
