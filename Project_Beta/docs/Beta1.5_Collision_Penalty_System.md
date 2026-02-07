# Beta 1.5: SOC-based Collision Penalty System

## 概要

Virtual Robot Raceにおいて、衝突時にバッテリー残量（SOC: State of Charge）を減少させるペナルティシステム。
機械学習モデルが「衝突を避ける」行動を学習するためのフレームワーク。

---

## 1. 設計原理

### 1.1 なぜ衝突ペナルティが必要か？

```
問題: AIモデルは「速く走る」ことだけを学習すると、壁や他ロボットに衝突しても気にしない
解決: 衝突 → バッテリー減少 → レース完走不可 → ペナルティとして学習される
```

### 1.2 設計思想

| 原則 | 説明 |
|------|------|
| **エネルギー比例** | 高速衝突 = 大ペナルティ、低速衝突 = 小ペナルティ |
| **責任分担** | ロボット同士の衝突では、突っ込んだ側がより多く責任を負う |
| **クールダウン** | 同一対象への連続衝突を1回としてカウント（物理的な跳ね返り対策） |
| **学習可能性** | フレームごとにCSVに記録し、オフライン強化学習で活用 |

### 1.3 衝突タイプ

```
┌─────────────────────────────────────────────────────────┐
│                    衝突タイプ                            │
├─────────────────────────────────────────────────────────┤
│  1. 壁衝突 (Wall Collision)                              │
│     - 責任: 100% 自己責任                                │
│     - エネルギー: E = |V|² (自分の速度)                   │
│                                                         │
│  2. ロボット間衝突 (Robot-to-Robot Collision)            │
│     - 責任: 速度方向に基づく分担 (0%～100%)               │
│     - エネルギー: E = |V_rel|² (相対速度)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 数学的モデル

### 2.1 基本ペナルティ計算式

$$
\text{Penalty} = k \times E \times R
$$

| 変数 | 意味 | 単位 |
|------|------|------|
| $k$ | 正規化係数 | 1/(m/s)² |
| $E$ | 衝突エネルギー | (m/s)² |
| $R$ | 責任比率 | 0.0 ~ 1.0 |

### 2.2 正規化係数 k の計算

```
k = basePenaltyRate / (maxSpeed)²
```

**デフォルト値:**
- `basePenaltyRate = 0.20` (20% SOC損失 @ 最大速度正面衝突)
- `maxSpeed = 5.0 m/s`

$$
k = \frac{0.20}{5.0^2} = \frac{0.20}{25} = 0.008
$$

### 2.3 壁衝突のエネルギー計算

```
E = |V|² = Vx² + Vy² + Vz²
```

**例:** 速度 3.0 m/s で壁に衝突
$$
E = 3.0^2 = 9.0
$$
$$
\text{Penalty} = 0.008 \times 9.0 \times 1.0 = 0.072 = 7.2\%
$$

### 2.4 ロボット間衝突のエネルギー計算

```
V_rel = V_self - V_other
E = |V_rel|²
```

**例:** Robot1が4.0m/sで前進、Robot2が1.0m/sで同方向に移動
$$
V_{rel} = 4.0 - 1.0 = 3.0 \text{ m/s}
$$
$$
E = 3.0^2 = 9.0
$$

### 2.5 責任比率 R の計算

```
R = 0.5 + 0.5 × dot(V_normalized, -collisionNormal)
```

**図解:**
```
            衝突法線 (collision normal)
                    ↑
                    │
    Robot1 ──→ ○ ← Robot2
              衝突点

dot = +1.0 → R = 1.0 (100%責任: 正面から突っ込んだ)
dot =  0.0 → R = 0.5 (50%責任: 横から接触)
dot = -1.0 → R = 0.0 (0%責任: 逃げていた側)
```

**責任分担の例:**
```
シナリオ1: Robot1が停止中のRobot2に突っ込む
  Robot1: R = 1.0 (100%)
  Robot2: R = 0.0 (0%)

シナリオ2: 両者が正面衝突
  Robot1: R ≈ 1.0 (≈100%)
  Robot2: R ≈ 1.0 (≈100%)
  → 両者とも高いペナルティ

シナリオ3: 追突 (同方向に移動中)
  追突した側: R ≈ 0.8~1.0
  追突された側: R ≈ 0.0~0.2
```

---

## 3. 実装コード

### 3.1 設定パラメータ (BodyCollisionHandler.cs)

```csharp
[Header("Collision Penalty Settings (Beta 1.5)")]
public float basePenaltyRate = 0.20f;   // 最大速度衝突時のSOC損失率
public float maxSpeed = 5.0f;            // 正規化用の基準速度 (m/s)
public float cooldownDuration = 1.0f;    // 同一対象への衝突クールダウン (秒)
```

### 3.2 壁衝突ペナルティ

```csharp
private void ApplyWallCollisionPenalty()
{
    // クールダウン中ならスキップ
    if (IsOnCooldown("Wall")) return;
    RecordCollision("Wall");

    // 衝突直前の速度を取得 (FixedUpdateで記録済み)
    Vector3 velocity = robotStatus.lastVelocity;

    // エネルギー計算: E = |V|²
    float E = velocity.sqrMagnitude;

    // 壁衝突は100%自己責任
    float R = 1.0f;

    // ペナルティ計算: Penalty = k × E × R
    float k = basePenaltyRate / (maxSpeed * maxSpeed);
    float penalty = k * E * R;

    // バッテリーにペナルティ適用
    batteryManager.ApplyCollisionPenaltyFraction(penalty);

    // データロガーに記録
    robotStatus.RecordCollisionForFrame("wall", penalty, "Wall");
}
```

### 3.3 ロボット間衝突ペナルティ

```csharp
private void ApplyRobotCollisionPenalty(BodyCollisionHandler otherHandler, Vector3 collisionNormal)
{
    string otherId = otherHandler.name;  // 例: "Robot2"

    // クールダウン中ならスキップ
    if (IsOnCooldown(otherId)) return;
    RecordCollision(otherId);

    // 両者の速度を取得
    Vector3 myVelocity = robotStatus.lastVelocity;
    Vector3 otherVelocity = otherHandler.robotStatus.lastVelocity;

    // 相対速度からエネルギー計算: E = |V_rel|²
    Vector3 relativeVelocity = myVelocity - otherVelocity;
    float E = relativeVelocity.sqrMagnitude;

    // 責任比率計算: R = 0.5 + 0.5 × dot(V_norm, -collisionNormal)
    float dotProduct = Vector3.Dot(myVelocity.normalized, -collisionNormal);
    float R = 0.5f + 0.5f * Mathf.Clamp(dotProduct, -1f, 1f);

    // ペナルティ計算: Penalty = k × E × R
    float k = basePenaltyRate / (maxSpeed * maxSpeed);
    float penalty = k * E * R;

    // バッテリーにペナルティ適用
    batteryManager.ApplyCollisionPenaltyFraction(penalty);

    // データロガーに記録 (ターゲットID付き)
    robotStatus.RecordCollisionForFrame("robot", penalty, otherId);
}
```

### 3.4 クールダウン管理

```csharp
private Dictionary<string, float> lastCollisionTime = new Dictionary<string, float>();

private bool IsOnCooldown(string targetId)
{
    if (!lastCollisionTime.ContainsKey(targetId)) return false;
    return Time.time - lastCollisionTime[targetId] < cooldownDuration;
}

private void RecordCollision(string targetId)
{
    lastCollisionTime[targetId] = Time.time;
}
```

### 3.5 バッテリーへのペナルティ適用 (BatteryManager.cs)

```csharp
public void ApplyCollisionPenaltyFraction(float fraction)
{
    if (isDepleted) return;  // 既に空なら何もしない

    // fraction: 0.0~1.0 (例: 0.05 = 5%のSOC損失)
    float penalty = maxCapacity * fraction;
    currentCapacity -= penalty;
    currentCapacity = Mathf.Clamp(currentCapacity, 0f, maxCapacity);
    soc = currentCapacity / maxCapacity;

    if (currentCapacity <= 0f)
    {
        isDepleted = true;  // バッテリー切れフラグ
    }
}
```

---

## 4. データ記録フォーマット

### 4.1 CSV出力形式

```csv
id,session_time_ms,race_time_ms,filename,soc,...,collision_type,collision_penalty,collision_target
267,8032,4062,frame_000267.jpg,0.996,...,robot,0.0035,Robot2
380,13932,9961,frame_000380.jpg,0.978,...,wall,0.0146,Wall
421,16052,12082,frame_000421.jpg,0.966,...,robot,0.0094,Robot2
```

### 4.2 フィールド説明

| フィールド | 型 | 説明 |
|------------|------|------|
| `collision_type` | string | "wall", "robot", "both", "" (空) |
| `collision_penalty` | float | 0.0~1.0+ (SOC減少率) |
| `collision_target` | string | "Wall", "Robot1", "Robot2", etc. |

### 4.3 データフロー

```
BodyCollisionHandler    →    RobotStatus    →    DataLogger    →    CSV
 (衝突検知・計算)         (フレーム蓄積)      (Tick毎に消費)      (ファイル出力)
```

---

## 5. 機械学習での活用

### 5.1 報酬関数への組み込み

```python
def calculate_reward(frame_data):
    reward = 0.0

    # 基本報酬: 前進距離
    reward += frame_data['distance_traveled'] * 1.0

    # 衝突ペナルティ (Beta 1.5)
    if frame_data['collision_penalty'] > 0:
        reward -= frame_data['collision_penalty'] * 100  # 大きな負の報酬

    return reward
```

### 5.2 学習目標

```
目標: 衝突ペナルティを最小化しながら、レースを完走する

      maximize: Σ (前進報酬) - Σ (衝突ペナルティ × 重み)
```

---

## 6. パラメータチューニングガイド

| パラメータ | 効果 | 推奨調整 |
|------------|------|----------|
| `basePenaltyRate` ↑ | 衝突1回の影響が大きくなる | 衝突回避を重視したい場合 |
| `maxSpeed` ↓ | 低速でも大ペナルティ | 慎重な走行を促したい場合 |
| `cooldownDuration` ↑ | 連続衝突が1回扱いになりやすい | 物理シミュ跳ね返り対策 |

---

## 7. 実際のログ例

```
[Collision] Robot1 hit Wall: E=1.87, penalty=1.5%
[Collision] Robot1 hit Robot2: E=3.56, R=0.93, penalty=2.7%
[Collision] Robot2 hit Robot1: E=3.56, R=1.00, penalty=2.8%
```

**解釈:**
- Robot1が壁に衝突: 速度² = 1.87、1.5%のSOC損失
- Robot1とRobot2が衝突: 相対速度² = 3.56
  - Robot1: 責任93%、2.7%損失
  - Robot2: 責任100%、2.8%損失
  - Robot2の方が相手に向かって進んでいた

---

## 8. まとめ

```
┌────────────────────────────────────────────────────────────┐
│                Beta 1.5 Collision Penalty System           │
├────────────────────────────────────────────────────────────┤
│  Penalty = k × E × R                                       │
│                                                            │
│  k = basePenaltyRate / maxSpeed²  (正規化係数)             │
│  E = |V|² or |V_rel|²             (衝突エネルギー)         │
│  R = 0.5 + 0.5 × dot(V, -n)       (責任比率, 壁は1.0固定)  │
│                                                            │
│  出力: collision_type, collision_penalty, collision_target │
└────────────────────────────────────────────────────────────┘
```
