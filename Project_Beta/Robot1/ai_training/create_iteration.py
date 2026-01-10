#!/usr/bin/env python3
"""
create_iteration.py
===================
新しいiterationフォルダを作成し、データソースをコピーして学習準備を行う

Usage:
    python scripts/create_iteration.py --data training_data_selected
    python scripts/create_iteration.py --data training_data --runs run_20260104_140000 run_20260104_140300

Features:
    - iteration_YYMMDD_HHMMSS フォルダ作成
    - データソースrun_*を全コピー
    - データ統計を自動分析
    - training_config.yaml自動生成
    - フォルダ構造の自動セットアップ
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import yaml


class IterationCreator:
    """新しいiterationフォルダを作成・管理するクラス"""

    def __init__(self, robot_dir: Path):
        """
        Initialize iteration creator.

        Args:
            robot_dir: Robot1ディレクトリのパス
        """
        self.robot_dir = Path(robot_dir)
        self.experiments_dir = self.robot_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)

    def create(
        self,
        data_source_dir: Path,
        specific_runs: Optional[List[str]] = None,
        description: str = ""
    ) -> Path:
        """
        新しいiterationフォルダを作成

        Args:
            data_source_dir: トレーニングデータのソースディレクトリ
            specific_runs: 特定のrun名リスト（Noneの場合は全run）
            description: iterationの説明

        Returns:
            作成されたiterationディレクトリのパス
        """
        # タイムスタンプGenerate # TODO: Manual translation needed
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        iteration_dir = self.experiments_dir / f"iteration_{timestamp}"

        print(f"\n{'='*70}")
        print(f"Creating Iteration: iteration_{timestamp}")
        print(f"{'='*70}\n")

        # Folder構造Create # TODO: Manual translation needed
        print("[1/5] Creating folder structure...")
        self._create_folder_structure(iteration_dir)

        # Dataソースをコピー # TODO: Manual translation needed
        print("\n[2/5] Copying data sources...")
        run_dirs = self._copy_data_sources(
            data_source_dir,
            iteration_dir / "data_sources",
            specific_runs
        )

        if not run_dirs:
            print("\n[Error] No valid training data found!")
            shutil.rmtree(iteration_dir)
            sys.exit(1)

        # Data統計を分析 # TODO: Manual translation needed
        print("\n[3/5] Analyzing data statistics...")
        data_stats = self._analyze_data_sources(iteration_dir / "data_sources")

        # training_config.yaml作成
        print("\n[4/5] Creating training_config.yaml...")
        self._create_training_config(
            iteration_dir,
            timestamp,
            data_source_dir,
            run_dirs,
            data_stats,
            description
        )

        # README作成
        print("\n[5/5] Creating README.md...")
        self._create_readme(iteration_dir, timestamp, data_stats)

        print(f"\n{'='*70}")
        print(f"[SUCCESS] Iteration created successfully!")
        print(f"{'='*70}")
        print(f"\nIteration directory: {iteration_dir}")
        print(f"\nNext steps:")
        print(f"  1. Review data_sources/ ({data_stats['total_runs']} runs)")
        print(f"  2. Run training:")
        print(f"     python train_model.py \\")
        print(f"       --data {iteration_dir / 'data_sources'} \\")
        print(f"       --output {iteration_dir} \\")
        print(f"       --epochs 50")
        print(f"  3. Evaluate with 3 test runs")
        print()

        return iteration_dir

    def _create_folder_structure(self, iteration_dir: Path):
        """フォルダ構造を作成"""
        folders = [
            iteration_dir,
            iteration_dir / "data_sources",
            iteration_dir / "evaluation",
            iteration_dir / "logs",
        ]

        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"  [OK] {folder.relative_to(self.robot_dir)}")

    def _copy_data_sources(
        self,
        source_dir: Path,
        dest_dir: Path,
        specific_runs: Optional[List[str]] = None
    ) -> List[Path]:
        """
        データソースをコピー

        Args:
            source_dir: コピー元ディレクトリ
            dest_dir: コピー先ディレクトリ
            specific_runs: 特定のrun名リスト

        Returns:
            コピーされたrunディレクトリのリスト
        """
        source_dir = Path(source_dir)
        run_dirs = []

        if specific_runs:
            # 特定のrunのみコピー # TODO: Manual translation needed
            for run_name in specific_runs:
                run_path = source_dir / run_name
                if run_path.exists() and run_path.is_dir():
                    dest_path = dest_dir / run_name
                    print(f"  Copying {run_name}...", end=" ")
                    shutil.copytree(run_path, dest_path)
                    run_dirs.append(dest_path)
                    print("[OK]")
                else:
                    print(f"  [Warning] Run not found: {run_name}")
        else:
            # 全runをコピー # TODO: Manual translation needed
            all_runs = sorted(source_dir.glob("run_*"))
            if not all_runs:
                print(f"  [Warning] No run_* directories found in {source_dir}")
                return []

            for run_path in all_runs:
                if run_path.is_dir():
                    dest_path = dest_dir / run_path.name
                    print(f"  Copying {run_path.name}...", end=" ")
                    shutil.copytree(run_path, dest_path)
                    run_dirs.append(dest_path)
                    print("[OK]")

        print(f"\n  Total runs copied: {len(run_dirs)}")
        return run_dirs

    def _analyze_data_sources(self, data_sources_dir: Path) -> Dict:
        """
        データソースの統計を分析

        Args:
            data_sources_dir: data_sourcesディレクトリ

        Returns:
            統計情報の辞書
        """
        stats = {
            "total_runs": 0,
            "total_frames": 0,
            "total_racing_frames": 0,
            "avg_steer": 0.0,
            "left_steer_ratio": 0.0,
            "right_steer_ratio": 0.0,
            "neutral_ratio": 0.0,
            "runs_detail": []
        }

        all_steers = []
        left_count = 0
        right_count = 0
        neutral_count = 0

        for run_dir in sorted(data_sources_dir.glob("run_*")):
            metadata_path = run_dir / "metadata.csv"
            if not metadata_path.exists():
                print(f"  [Warning] metadata.csv not found in {run_dir.name}")
                continue

            try:
                df = pd.read_csv(metadata_path)

                # Racing frames のみ抽出（StartSequenceを除外）
                # statusカラムをCheck（final_statusまたはstatus） # TODO: Manual translation needed
                status_col = 'final_status' if 'final_status' in df.columns else 'status'
                racing_df = df[df[status_col] != 'StartSequence']

                if len(racing_df) == 0:
                    continue

                # タイムスタンプカラムをCheck # TODO: Manual translation needed
                time_col = 'timestamp' if 'timestamp' in racing_df.columns else 'race_time_ms'
                duration = racing_df[time_col].max() / 1000.0  # ms -> s
                avg_steer = racing_df['steer_angle'].mean()

                # 左右カウント（閾値: 0.05 rad） # TODO: Manual translation needed
                left = (racing_df['steer_angle'] < -0.05).sum()
                right = (racing_df['steer_angle'] > 0.05).sum()
                neutral = len(racing_df) - left - right

                run_info = {
                    "name": run_dir.name,
                    "frames": len(df),
                    "racing_frames": len(racing_df),
                    "duration_sec": round(duration, 2),
                    "avg_steer": round(avg_steer, 4),
                    "left_count": int(left),
                    "right_count": int(right),
                    "neutral_count": int(neutral),
                }

                stats["runs_detail"].append(run_info)
                stats["total_runs"] += 1
                stats["total_frames"] += len(df)
                stats["total_racing_frames"] += len(racing_df)

                all_steers.extend(racing_df['steer_angle'].tolist())
                left_count += left
                right_count += right
                neutral_count += neutral

                print(f"  [OK] {run_dir.name}: {len(racing_df)} frames, "
                      f"{duration:.1f}s, avg_steer={avg_steer:.3f}")

            except Exception as e:
                print(f"  [Error] Failed to analyze {run_dir.name}: {e}")

        # 全体統計Calculate # TODO: Manual translation needed
        if all_steers:
            stats["avg_steer"] = round(sum(all_steers) / len(all_steers), 4)

            total_steers = left_count + right_count + neutral_count
            stats["left_steer_ratio"] = round(left_count / total_steers * 100, 2)
            stats["right_steer_ratio"] = round(right_count / total_steers * 100, 2)
            stats["neutral_ratio"] = round(neutral_count / total_steers * 100, 2)

        print(f"\n  Summary:")
        print(f"    Total runs: {stats['total_runs']}")
        print(f"    Total racing frames: {stats['total_racing_frames']}")
        print(f"    Average steer: {stats['avg_steer']:.4f} rad")
        print(f"    Left/Right/Neutral: {stats['left_steer_ratio']:.1f}% / "
              f"{stats['right_steer_ratio']:.1f}% / {stats['neutral_ratio']:.1f}%")

        # バランスチェック # TODO: Manual translation needed
        if abs(stats['avg_steer']) > 0.05:
            print(f"\n  [WARNING] Steering bias detected! avg_steer = {stats['avg_steer']:.4f}")
        elif abs(stats['left_steer_ratio'] - stats['right_steer_ratio']) > 20:
            print(f"\n  [WARNING] Left/Right imbalance! "
                  f"({stats['left_steer_ratio']:.1f}% vs {stats['right_steer_ratio']:.1f}%)")
        else:
            print(f"\n  [OK] Good balance!")

        return stats

    def _create_training_config(
        self,
        iteration_dir: Path,
        timestamp: str,
        source_dir: Path,
        run_dirs: List[Path],
        data_stats: Dict,
        description: str
    ):
        """training_config.yamlを作成"""
        config = {
            "iteration": {
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "description": description or f"Iteration {timestamp}",
            },
            "data_source": {
                "original_dir": str(source_dir),
                "total_runs": len(run_dirs),
                "run_names": [run_dir.name for run_dir in run_dirs],
            },
            "data_statistics": {
                "total_frames": data_stats["total_frames"],
                "total_racing_frames": data_stats["total_racing_frames"],
                "avg_steer": data_stats["avg_steer"],
                "left_steer_ratio": data_stats["left_steer_ratio"],
                "right_steer_ratio": data_stats["right_steer_ratio"],
                "neutral_ratio": data_stats["neutral_ratio"],
            },
            "training": {
                "status": "not_started",
                "model_path": None,
                "best_val_loss": None,
                "best_epoch": None,
            },
            "evaluation": {
                "test_runs": [],
                "completion_rate": None,
            }
        }

        config_path = iteration_dir / "training_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        print(f"  [OK] {config_path.relative_to(self.robot_dir)}")

        # JSONでもSave（Loadやすい） # TODO: Manual translation needed
        json_path = iteration_dir / "training_config.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _create_readme(self, iteration_dir: Path, timestamp: str, data_stats: Dict):
        """README.mdを作成"""
        readme_content = f"""# Iteration {timestamp}

**Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# # Dataソース # TODO: Manual translation needed

- **総run数:** {data_stats['total_runs']}
- **総フレーム数:** {data_stats['total_frames']:,}
- **レーシングフレーム数:** {data_stats['total_racing_frames']:,}
- **平均ステア:** {data_stats['avg_steer']:.4f} rad
- **左/右/ニュートラル:** {data_stats['left_steer_ratio']:.1f}% / {data_stats['right_steer_ratio']:.1f}% / {data_stats['neutral_ratio']:.1f}%

### Run詳細

| Run名 | フレーム数 | レーシング | 時間(s) | 平均ステア |
|-------|-----------|----------|---------|-----------|
"""
        for run in data_stats['runs_detail']:
            readme_content += f"| {run['name']} | {run['frames']} | {run['racing_frames']} | {run['duration_sec']} | {run['avg_steer']:.4f} |\n"

        readme_content += f"""
# # Folder構造 # TODO: Manual translation needed

```
iteration_{timestamp}/
├── README.md                  # このFile # TODO: Manual translation needed
├── training_config.yaml       # TrainingSettings・Dataソース情報 # TODO: Manual translation needed
├── training_config.json       # 同上（JSON形式） # TODO: Manual translation needed
├── data_sources/              # トレーニングData（全コピー） # TODO: Manual translation needed
│   ├── run_XXXXXX_XXXXXX/
│   └── ...
├── evaluation/                # Test走行結果 # TODO: Manual translation needed
│   ├── test_run_001.json
│   ├── test_run_002.json
│   ├── test_run_003.json
│   └── evaluation_summary.md
└── logs/                      # TrainingLog
    ├── training_log.txt
    └── loss_curve.png
```

# # 次のステップ # TODO: Manual translation needed

# ## 1. TrainingExecute

```bash
python train_model.py \\
  --data {iteration_dir.name}/data_sources \\
  --output {iteration_dir.name} \\
  --epochs 50 \\
  --device cuda
```

# ## 2. Test走行（3回） # TODO: Manual translation needed

手動またはスクリプトで3回走行し、評価結果を記録

# ## 3. 結果分析 # TODO: Manual translation needed

- 完走率を確認
- クラッシュ地点を分析
- 次のiterationの方針を決定

# # Training結果 # TODO: Manual translation needed

（学習後に自動更新されます）

# # 評価結果 # TODO: Manual translation needed

（テスト走行後に記録されます）
"""

        readme_path = iteration_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"  [OK] {readme_path.relative_to(self.robot_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create new iteration folder with data sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 全runをコピー # TODO: Manual translation needed
  python scripts/create_iteration.py --data training_data_selected

  # 特定のrunのみコピー # TODO: Manual translation needed
  python scripts/create_iteration.py --data training_data \\
    --runs run_20260104_140000 run_20260104_140300 run_20260104_140500

  # 説明をAdd # TODO: Manual translation needed
  python scripts/create_iteration.py --data training_data_selected \\
    --description "Manual keyboard runs only, balanced steering"
        """
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Data source directory (e.g., training_data_selected)"
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        help="Specific run names to copy (optional, default: all runs)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of this iteration"
    )

    args = parser.parse_args()

    # Determine Robot1 directory
    script_dir = Path(__file__).parent
    robot_dir = script_dir.parent

    # DataソースDirectoryのValidation # TODO: Manual translation needed
    data_source_dir = robot_dir / args.data
    if not data_source_dir.exists():
        print(f"[Error] Data source directory not found: {data_source_dir}")
        sys.exit(1)

    # Iteration作成
    creator = IterationCreator(robot_dir)
    iteration_dir = creator.create(
        data_source_dir=data_source_dir,
        specific_runs=args.runs,
        description=args.description
    )

    return iteration_dir


if __name__ == "__main__":
    main()
