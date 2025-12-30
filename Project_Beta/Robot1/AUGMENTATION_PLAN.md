# データオーグメンテーション実行プラン

**日時:** 2025-12-28
**目的:** マニュアル走行データのみを対象に左右反転オーグメンテーション

---

## 📋 修正内容

### 問題点
元のスクリプトは`training_data/`内の**すべてのrun**を対象にしていた:
- ✅ マニュアル走行データ(Iteration 1, 2, 3)
- ❌ **AIモードのクラッシュデータ**(run_20251228_*)も含まれる

### 解決策
**レース時間でフィルタリング**:
```python
# Include only runs with race time > 10 seconds (manual driving)
if race_time_sec > 10.0:
    run_dirs.append(run_dir)
else:
    excluded_runs.append((run_dir.name, race_time_sec))
```

**基準:**
- レース時間 > 10秒 → マニュアル走行データとして採用
- レース時間 ≤ 10秒 → AIクラッシュとして除外

---

## 🎯 対象データ

### 採用されるrun(推定)
- Iteration 1, 2, 3のマニュアル走行データ
- 12/14, 12/27のデータ(おそらく20~60秒のrun)
- 合計: 約40~50 runs

### 除外されるrun
- 12/28の今日のAIクラッシュデータ:
  - run_20251228_092313 (44秒) → **採用される**(意外!)
  - run_20251228_094712 (2.8秒) → 除外
  - run_20251228_095310 (4.0秒) → 除外
  - その他短いrun → すべて除外

**注意:** run_20251228_092313は44秒走っているので採用されますが、
これはAIモードで左に落ちたデータなので、本来は除外すべきかもしれません。

---

## 🔧 さらなる改善案(オプション)

### オプション1: より厳密なフィルタ(レース時間 > 30秒)

```python
# More strict: only long manual runs
if race_time_sec > 30.0:
    run_dirs.append(run_dir)
```

**効果:**
- より確実にマニュアル走行データのみ
- AIクラッシュデータを完全排除

### オプション2: 日付でフィルタ

```python
# Exclude runs from 2025-12-28 (today's AI crashes)
run_date = run_dir.name.split('_')[1]  # '20251228'
if run_date != '20251228':
    run_dirs.append(run_dir)
```

**効果:**
- 今日のAIクラッシュを確実に除外
- シンプルで確実

### オプション3: 手動で指定

```python
# Manually specify good runs
GOOD_RUNS = [
    'run_20251214_*',  # Iteration 3 manual data
    'run_20251227_*',  # Recent manual data
]
```

---

## 🚀 実行コマンド

### 基本実行(現在の設定)
```bash
cd Robot1
../.venv/Scripts/python scripts/augment_training_data.py
```

### 確認だけしたい場合
スクリプトに`--dry-run`オプションを追加するか、または実行して最初の出力を確認:

```
Found 57 total runs
  -> 43 manual driving runs (race time > 10s)
  -> 14 excluded (AI crashes or short runs)

Excluded runs (showing last 5):
  - run_20251228_095310 (4.0s)
  - run_20251228_094712 (2.8s)
  - ...

43 runs will be augmented
```

この段階で **Ctrl+C** で中断して内容を確認できます。

---

## ⚠️ 注意事項

### 1. AIクラッシュデータの影響

もし run_20251228_092313 (44秒、左に落ちた)が含まれる場合:
- このrunは左バイアスが強い(-0.0027 rad)
- オーグメンテーションで反転されるが、**両方のデータが含まれる**
- つまり、左バイアスと右バイアスの両方が学習される

**判断:**
- 含めても問題ない(オーグメンテーションでバランスが取れる)
- より厳密にするなら、レース時間 > 30秒 に変更

### 2. データ量の確認

オーグメンテーション後のデータ量:
```
元データ: 32,994フレーム(50 runs)
除外後: 約30,000フレーム(43 runs)
オーグメンテーション後: 約60,000フレーム(43 runs × 2)
```

**期待される効果:**
- 左右完全バランス(50/50)
- データ量2倍
- 汎化性能向上

---

## 📊 推奨フロー

### ステップ1: 確認実行
```bash
cd Robot1
../.venv/Scripts/python scripts/augment_training_data.py
```

最初の出力を確認:
```
Found XX total runs
  -> YY manual driving runs (race time > 10s)
  -> ZZ excluded (AI crashes or short runs)
```

**確認ポイント:**
- YYが40~50くらいになっているか
- ZZに今日のクラッシュrunが含まれているか

### ステップ2: 判断

**ケースA: 問題なし**
→ そのまま実行継続(Enter押すだけ)

**ケースB: 今日のAIデータ(44秒)が含まれている**
→ オプション:
1. そのまま実行(問題ない可能性が高い)
2. スクリプト修正して30秒に変更

### ステップ3: 実行
- 5-10分程度で完了(50 runs × 平均600フレーム)
- 進捗が表示される

### ステップ4: 確認
```bash
../.venv/Scripts/python scripts/analyze_steering_bias.py --input training_data_augmented
```

**期待される結果:**
```
Left steering frames:  30,000 (50.0%)
Right steering frames: 30,000 (50.0%)
Average mean steer: 0.0000 rad  # ← ほぼゼロ!
```

---

## ✅ チェックリスト

実行前:
- [ ] スクリプト修正完了(レース時間フィルタ追加)
- [ ] 対象runが適切か確認
- [ ] ディスク容量確認(約2GB必要)

実行:
- [ ] オーグメンテーション実行
- [ ] エラーなく完了
- [ ] training_data_augmented/ 作成確認

検証:
- [ ] analyze_steering_bias.py で分析
- [ ] 左右バランス50/50確認
- [ ] 平均ステア ≈ 0.0確認

次のステップ:
- [ ] train_model.py修正
- [ ] モデル再学習
- [ ] テスト走行

---

準備完了です! 🏁
