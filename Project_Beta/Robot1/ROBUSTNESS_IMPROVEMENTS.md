# AI Control Strategy Robustness Improvements
## VRR Beta 1.x - Crash Avoidance Update

**Date:** 2025-12-28
**Target:** `ai_control_strategy.py`
**Goal:** Increase 2-lap completion rate by stabilizing cornering without retraining the neural network

---

## 概要 (Summary)

ニューラルネットワークの再学習なしで、`ai_control_strategy.py`内の制御ロジックのみを改善し、コーナーでのクラッシュを防ぎ、2周完走率を向上させます。

---

## 実装した3つの改善

### (A) Conditional Start Boost (コーナー対応スタートブースト)

**問題:**
- 従来のスタートブーストは、ステアリング角に関係なく一律に最小トルク(0.4)を強制
- スタート直後にコーナーがある場合、高トルクで突入してクラッシュ

**解決策:**
- スタートブースト適用条件を追加:
  - `abs(adjusted_steer) <= 0.10 rad` の場合のみ最小トルクを適用
  - それ以外(急ハンドル時)は、モデル出力またはコーナーキャップに任せる
- パラメータ調整:
  - `START_BOOST_FRAMES`: 40 → **22** (約1.1秒)
  - `MIN_DRIVE_TORQUE`: 0.4 → **0.32** (コーナリング余地を確保)
  - `START_BOOST_STEER_THRESHOLD`: **0.10 rad** (判定閾値)

**実装箇所:** [ai_control_strategy.py:221-239](ai_control_strategy.py#L221-L239)

```python
if steer_abs <= START_BOOST_STEER_THRESHOLD:
    # 直線または緩いカーブ: ブースト適用
    if adjusted_drive < MIN_DRIVE_TORQUE:
        adjusted_drive = MIN_DRIVE_TORQUE
else:
    # 急カーブ検出: ブーストを抑制
    # (ログ出力のみ、トルク強制なし)
```

---

### (B) Steering Rate Limiter (ステアリングレート制限)

**問題:**
- ローパスフィルタ(平滑化)だけでは、急激なステアリング変化を防げない
- 1フレームで大きく舵角が変わると、車両が不安定化

**解決策:**
- フレーム間のステアリング変化量を **0.03 rad/frame** に制限
- 平滑化の**後**、最終クランプの**前**に適用(ハードセーフティ)

**実装箇所:** [ai_control_strategy.py:210-219](ai_control_strategy.py#L210-L219)

```python
delta_steer = adjusted_steer - adjust_output._prev_steer
if abs(delta_steer) > MAX_STEER_DELTA_PER_FRAME:
    delta_steer = max(-MAX_STEER_DELTA_PER_FRAME,
                     min(MAX_STEER_DELTA_PER_FRAME, delta_steer))
    adjusted_steer = adjust_output._prev_steer + delta_steer
```

**パラメータ:**
- `MAX_STEER_DELTA_PER_FRAME`: **0.03 rad/frame**

---

### (C) Corner-Aware Drive Torque Cap (コーナー対応トルクキャップ)

**問題:**
- 全体的な最大トルク(0.55)は設定されているが、コーナー時も同じ上限
- 急カーブ中に高トルクが出力され、オーバースピードでクラッシュ

**解決策:**
- ステアリング角の大きさに応じて、ドライブトルクの上限を動的に下げる
- 線形補間(lerp)を使用:
  - `abs(steer) < 0.20 rad`: 通常の最大トルク(0.55)
  - `0.20 <= abs(steer) < 0.50 rad`: 線形に減少
  - `abs(steer) >= 0.50 rad`: 最小トルク(0.30)

**実装箇所:** [ai_control_strategy.py:245-269](ai_control_strategy.py#L245-L269)

```python
if steer_abs >= CORNER_STEER_THRESHOLD_LOW:
    # 補間係数 t を計算 (0.0 at LOW, 1.0 at HIGH)
    t = (steer_abs - CORNER_STEER_THRESHOLD_LOW) / \
        (CORNER_STEER_THRESHOLD_HIGH - CORNER_STEER_THRESHOLD_LOW)
    t = max(0.0, min(1.0, t))

    # 線形補間: drive_cap = MAX + t * (MIN - MAX)
    drive_cap = MAX_DRIVE_TORQUE + t * (CORNER_MIN_DRIVE_TORQUE - MAX_DRIVE_TORQUE)

    if adjusted_drive > drive_cap:
        adjusted_drive = drive_cap
```

**パラメータ:**
- `CORNER_STEER_THRESHOLD_LOW`: **0.20 rad**
- `CORNER_STEER_THRESHOLD_HIGH`: **0.50 rad**
- `CORNER_MIN_DRIVE_TORQUE`: **0.30**

---

## デフォルトパラメータ値 (推奨設定)

すべてのパラメータはファイル上部で定義されています([ai_control_strategy.py:86-104](ai_control_strategy.py#L86-L104)):

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `START_BOOST_FRAMES` | **22** | スタートブースト適用フレーム数(約1.1秒) |
| `MIN_DRIVE_TORQUE` | **0.32** | スタートブースト時の最小トルク |
| `START_BOOST_STEER_THRESHOLD` | **0.10 rad** | ブースト抑制のステアリング閾値 |
| `MAX_STEER_DELTA_PER_FRAME` | **0.03 rad** | 1フレームあたりの最大ステアリング変化 |
| `MAX_DRIVE_TORQUE` | **0.55** | 全体的な最大トルク(従来通り) |
| `MAX_STEER_RAD` | **0.30 rad** | 最大ステアリング角(従来通り) |
| `CORNER_STEER_THRESHOLD_LOW` | **0.20 rad** | コーナーキャップ開始閾値 |
| `CORNER_STEER_THRESHOLD_HIGH` | **0.50 rad** | コーナーキャップ最大閾値 |
| `CORNER_MIN_DRIVE_TORQUE` | **0.30** | 急カーブ時の最小トルク上限 |
| `STEER_SMOOTHING_ALPHA` | **0.7** | ステアリング平滑化係数(従来通り) |

---

## テスト方法

### Unit-Level Test (ユニットレベル)

以下のスニペットで、ステアリングレート制限とトルクキャップの動作を確認できます:

```python
# test_robustness.py
import sys
sys.path.insert(0, 'Robot1')
from ai_control_strategy import adjust_output

# Reset state
if hasattr(adjust_output, '_race_frame_count'):
    delattr(adjust_output, '_race_frame_count')

# Simulate race start
test_cases = [
    # (drive, steer, expected_behavior)
    (0.5, 0.0, "Straight: start boost should activate"),
    (0.5, 0.05, "Gentle turn: start boost should activate"),
    (0.5, 0.15, "Sharp turn: start boost SUPPRESSED"),
    (0.5, 0.25, "Sharp corner: drive torque should be capped"),
    (0.5, 0.40, "Very sharp: drive torque heavily capped"),
]

print("=== Robustness Test ===\n")
for i, (drive, steer, desc) in enumerate(test_cases):
    adj_drive, adj_steer = adjust_output(
        drive, steer,
        pil_img=None,
        soc=1.0,
        race_started=True
    )
    print(f"Frame {i+1}: {desc}")
    print(f"  Input:  drive={drive:.2f}, steer={steer:.2f}")
    print(f"  Output: drive={adj_drive:.2f}, steer={adj_steer:.2f}")
    print()

# Test steering rate limit
print("=== Steering Rate Limit Test ===\n")
adjust_output._prev_steer = 0.0
adj_drive, adj_steer = adjust_output(0.5, 0.20, None, 1.0, True)
print(f"Step 1: steer 0.00 -> 0.20 request, actual: {adj_steer:.3f} (should be limited to ~0.03)")

adj_drive, adj_steer = adjust_output(0.5, 0.20, None, 1.0, True)
print(f"Step 2: continue to 0.20, actual: {adj_steer:.3f} (gradual approach)")
```

**実行方法:**
```bash
cd d:\AARACE\GitProjects\virtual-robot-race\Project_Beta
.venv\Scripts\python test_robustness.py
```

---

### Runtime-Level Test (実行時検証)

実際のレース中にログから以下を確認:

#### 1. スタートブーストの抑制
ログで以下のメッセージを探す:
```
[Strategy] Start boost SUPPRESSED (10/22): steer=0.152 > 0.10
```
→ 急ハンドル時にブーストが無効化されていることを確認

#### 2. コーナー時のトルクキャップ
```
[Strategy] Corner cap: steer=0.284, drive capped 0.50 -> 0.38
```
→ 大きなステアリング角でトルクが削減されていることを確認

#### 3. ステアリングレート制限
コンソールログまたはテレメトリで、連続フレーム間のステアリング変化量が0.03 rad以下であることを確認:
```python
# ai_control.py 内に追加して検証
prev = 0.0
for frame in telemetry:
    delta = abs(frame['steer'] - prev)
    assert delta <= 0.031, f"Rate limit violated: {delta}"
    prev = frame['steer']
```

---

## 変更箇所のまとめ

1. **パラメータ定義セクション追加** ([L80-105](ai_control_strategy.py#L80-L105))
   - すべての調整可能パラメータを上部に集約

2. **`adjust_output()` 関数の全面改修** ([L161-275](ai_control_strategy.py#L161-L275))
   - 処理順序の最適化(ステアリング処理 → ブースト → トルクキャップ)
   - 3つの新機能を統合
   - 詳細なログ出力を追加

---

## なぜこの改善が効果的か

1. **スタート直後のコーナークラッシュを防止**
   - 条件付きブーストにより、コーナー中に無理な加速をしない

2. **ステアリングの安定性向上**
   - レート制限により、モデルの急な出力変化を物理的に抑制
   - 平滑化と併用することで、より安定した挙動

3. **コーナーでのオーバースピード防止**
   - ステアリング角に応じて自動的にスピードダウン
   - モデルが高トルクを出力しても、物理的に安全な範囲に制限

4. **モデル再学習不要**
   - すべて後処理(post-processing)レイヤーでの改善
   - 既存のモデル資産を活用しながら安全性を向上

---

## 今後の調整

もしクラッシュが続く場合、以下のパラメータを調整:

- **より保守的にする場合:**
  - `MAX_STEER_DELTA_PER_FRAME` を 0.025 に減少
  - `CORNER_MIN_DRIVE_TORQUE` を 0.25 に減少
  - `CORNER_STEER_THRESHOLD_LOW` を 0.15 に減少

- **よりアグレッシブにする場合:**
  - `START_BOOST_FRAMES` を 30 に増加
  - `MIN_DRIVE_TORQUE` を 0.35 に増加
  - `CORNER_STEER_THRESHOLD_LOW` を 0.25 に増加

---

## まとめ

この改善により、以下が達成されます:

✅ ニューラルネットワークの再学習なし
✅ コーナーでの安定性向上
✅ スタート直後のクラッシュ防止
✅ パラメータ調整が容易(すべて上部に集約)
✅ 既存インターフェースとの完全な互換性

**Expected Result:** 2周完走率の大幅な向上

Good luck, Race Engineer! 🏁
