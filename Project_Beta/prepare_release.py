"""
Beta 1.2 Release Preparation Script
====================================
This script prepares the repository for Beta 1.2 release by:
1. Deleting development documentation and personal data
2. Reorganizing folder structure
3. Creating clean directories

WARNING: This script will DELETE files. Make sure you have a backup!

Usage:
    python prepare_release.py --dry-run    # Preview changes without applying
    python prepare_release.py --execute    # Apply changes
"""

import os
import shutil
from pathlib import Path
import argparse


class ReleasePreparation:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent
        self.actions = []

    def log(self, action, path, extra=""):
        """Log an action."""
        status = "[DRY-RUN]" if self.dry_run else "[EXECUTE]"
        message = f"{status} {action}: {path} {extra}"
        print(message)
        self.actions.append(message)

    def delete_file(self, file_path):
        """Delete a file."""
        if file_path.exists():
            self.log("DELETE FILE", file_path)
            if not self.dry_run:
                file_path.unlink()

    def delete_directory(self, dir_path):
        """Delete a directory and all its contents."""
        if dir_path.exists():
            self.log("DELETE DIR", dir_path)
            if not self.dry_run:
                shutil.rmtree(dir_path)

    def move_file(self, src, dst):
        """Move a file from src to dst."""
        if src.exists():
            self.log("MOVE", src, f"→ {dst}")
            if not self.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    def create_directory(self, dir_path):
        """Create an empty directory."""
        self.log("CREATE DIR", dir_path)
        if not self.dry_run:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_gitkeep(self, dir_path):
        """Create a .gitkeep file in a directory."""
        gitkeep = dir_path / ".gitkeep"
        self.log("CREATE", gitkeep)
        if not self.dry_run:
            dir_path.mkdir(parents=True, exist_ok=True)
            gitkeep.touch()

    def run(self):
        """Execute the release preparation."""
        print("=" * 80)
        print("Beta 1.2 Release Preparation")
        print("=" * 80)
        print(f"Mode: {'DRY-RUN (no changes will be made)' if self.dry_run else 'EXECUTE (changes will be applied)'}")
        print("=" * 80)
        print()

        # ========================================
        # Step 1: Delete development documentation
        # ========================================
        print("\n### Step 1: Deleting development documentation ###\n")

        dev_docs = [
            "AgentAI-Context.md",
            "Robot1/AUGMENTATION_PLAN.md",
            "Robot1/CRASH_ANALYSIS_20251228.md",
            "Robot1/CRASH_ANALYSIS_20251228_2.md",
            "Robot1/DATA_PIPELINE_VERIFICATION.md",
            "Robot1/ITERATION_WORKFLOW_GUIDE.md",
            "Robot1/MODEL_COMPARISON_ANALYSIS.md",
            "Robot1/MODEL_RETRAINING_PLAN.md",
            "Robot1/PARAMETER_ADJUSTMENTS_20251228.md",
            "Robot1/ROBUSTNESS_IMPROVEMENTS.md",
            "Robot1/TRAINING_SYSTEM_ARCHITECTURE.md",
            "Robot1/experiments/progress_report.md",
        ]

        for doc in dev_docs:
            self.delete_file(self.project_root / doc)

        # ========================================
        # Step 2: Delete personal data folders
        # ========================================
        print("\n### Step 2: Deleting personal data folders ###\n")

        # Delete all iteration folders
        experiments_dir = self.project_root / "Robot1" / "experiments"
        if experiments_dir.exists():
            for item in experiments_dir.iterdir():
                if item.is_dir() and item.name.startswith("iteration_"):
                    self.delete_directory(item)

        # Delete all training data
        training_data_dir = self.project_root / "Robot1" / "training_data"
        if training_data_dir.exists():
            for item in training_data_dir.iterdir():
                if item.is_dir() and item.name.startswith("run_"):
                    self.delete_directory(item)

        # Delete augmented training data
        self.delete_directory(self.project_root / "Robot1" / "training_data_augmented")

        # Delete Beta2_track_design
        self.delete_directory(self.project_root / "Beta2_track_design")

        # ========================================
        # Step 3: Delete old scripts
        # ========================================
        print("\n### Step 3: Deleting old/test scripts ###\n")

        old_scripts = [
            "Robot1/train_model.py",
            "Robot1/train_model_extended.py",
            "Robot1/test_robustness.py",
            "Robot1/scripts/analyze_steering_bias.py",
            "Robot1/scripts/auto_collect.py",
            "Robot1/scripts/combine_training_data.py",
            "Robot1/scripts/data_filter.py",
            "Robot1/scripts/verify_model_input.py",
            "Robot1/scripts/augment_training_data.py",
        ]

        for script in old_scripts:
            self.delete_file(self.project_root / script)

        # ========================================
        # Step 4: Reorganize folder structure
        # ========================================
        print("\n### Step 4: Reorganizing folder structure ###\n")

        # Move colab/ to top level
        src_colab = self.project_root / "Robot1" / "colab"
        dst_colab = self.project_root / "colab"
        if src_colab.exists() and not dst_colab.exists():
            self.log("MOVE DIR", src_colab, f"→ {dst_colab}")
            if not self.dry_run:
                shutil.move(str(src_colab), str(dst_colab))

        # Create Robot1/ai_training/
        ai_training_dir = self.project_root / "Robot1" / "ai_training"
        self.create_directory(ai_training_dir)

        # Move scripts to ai_training/
        scripts_to_move = [
            ("Robot1/scripts/train.py", "Robot1/ai_training/train.py"),
            ("Robot1/scripts/create_iteration.py", "Robot1/ai_training/create_iteration.py"),
            ("Robot1/scripts/run_pipeline.py", "Robot1/ai_training/run_pipeline.py"),
            ("Robot1/scripts/analyze.py", "Robot1/ai_training/analyze.py"),
            ("Robot1/scripts/run_iteration.py", "Robot1/ai_training/run_iteration.py"),
            ("Robot1/experiments/config.yaml", "Robot1/ai_training/config.yaml"),
        ]

        for src, dst in scripts_to_move:
            src_path = self.project_root / src
            dst_path = self.project_root / dst
            if src_path.exists():
                self.move_file(src_path, dst_path)

        # Delete now-empty Robot1/scripts/
        scripts_dir = self.project_root / "Robot1" / "scripts"
        if scripts_dir.exists() and not any(scripts_dir.iterdir()):
            self.delete_directory(scripts_dir)

        # ========================================
        # Step 5: Create empty directories with .gitkeep
        # ========================================
        print("\n### Step 5: Creating empty directories ###\n")

        # Robot1
        self.create_gitkeep(self.project_root / "Robot1" / "training_data")
        self.create_gitkeep(self.project_root / "Robot1" / "experiments")
        self.create_gitkeep(self.project_root / "Robot1" / "models")

        # Robot2 - same structure
        self.create_gitkeep(self.project_root / "Robot2" / "training_data")
        self.create_gitkeep(self.project_root / "Robot2" / "experiments")
        self.create_gitkeep(self.project_root / "Robot2" / "models")

        # ========================================
        # Step 6: Clean up Robot2 (same as Robot1)
        # ========================================
        print("\n### Step 6: Cleaning up Robot2 ###\n")

        # Delete Robot2 experiments
        robot2_experiments = self.project_root / "Robot2" / "experiments"
        if robot2_experiments.exists():
            for item in robot2_experiments.iterdir():
                if item.is_dir() and item.name.startswith("iteration_"):
                    self.delete_directory(item)

        # Delete Robot2 training data
        robot2_training = self.project_root / "Robot2" / "training_data"
        if robot2_training.exists():
            for item in robot2_training.iterdir():
                if item.is_dir() and item.name.startswith("run_"):
                    self.delete_directory(item)

        # Delete Robot2 old scripts (if any exist)
        robot2_old_scripts = [
            "Robot2/train_model.py",
            "Robot2/train_model_extended.py",
            "Robot2/test_robustness.py",
        ]

        for script in robot2_old_scripts:
            self.delete_file(self.project_root / script)

        # Create Robot2/ai_training/ and move scripts (if they exist)
        robot2_ai_training = self.project_root / "Robot2" / "ai_training"
        self.create_directory(robot2_ai_training)

        robot2_scripts_to_move = [
            ("Robot2/scripts/train.py", "Robot2/ai_training/train.py"),
            ("Robot2/scripts/create_iteration.py", "Robot2/ai_training/create_iteration.py"),
            ("Robot2/experiments/config.yaml", "Robot2/ai_training/config.yaml"),
        ]

        for src, dst in robot2_scripts_to_move:
            src_path = self.project_root / src
            dst_path = self.project_root / dst
            if src_path.exists():
                self.move_file(src_path, dst_path)

        # ========================================
        # Summary
        # ========================================
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Total actions: {len(self.actions)}")

        if self.dry_run:
            print("\nWARNING: This was a DRY-RUN. No changes were made.")
            print("To apply these changes, run with --execute flag:")
            print("    python prepare_release.py --execute")
        else:
            print("\nSUCCESS: All changes have been applied!")
            print("\nNext steps:")
            print("1. Review the changes with: git status")
            print("2. Translate comments to English")
            print("3. Commit changes")
            print("4. Create release branch")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Prepare Beta 1.2 release")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    group.add_argument("--execute", action="store_true", help="Apply changes")

    args = parser.parse_args()

    prep = ReleasePreparation(dry_run=args.dry_run)
    prep.run()


if __name__ == "__main__":
    main()
