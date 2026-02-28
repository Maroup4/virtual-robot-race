# 0. 事前準備

## はじめに
こんにちは！この研修では、仮想ロボットレースプラットフォーム「aira」で、AIやルールベースの制御プログラムを開発する方法を学びます。
この最初のセッションでは、開発に必要な環境をあなたのPCに構築します。

**作業時間の目安**: 約10分（※Pythonライブラリ等のダウンロード時間を除く）

スムーズな研修開始のため、一つずつ確認しながら進めましょう。

*   **そもそも「aira」とは？** -> [コンセプト紹介](https://www.youtube.com/watch?v=wAaaAODsfrE&t=26s)
*   **動画で見る環境構築ガイド** -> [YouTube版インストールガイド](https://www.youtube.com/watch?v=cvUdITqjpc8)

aira:autonomous intelligence racing arena

---

## 対象となる方

本研修は、以下のような方を対象としています。

*   AI、自動運転、ロボット制御といった分野に興味があり、実際に手を動かして学んでみたい、大学3年生以上の方、および社会人の方。
*   Pythonのプログラミング経験は問いません。ただし、基本的なPC操作（タイピング、ファイルのコピー＆ペースト、ターミナルでのコマンド実行など）はスムーズに行えることを前提とします。
*   研修で使用するGoogle社のAIサービス（Gemini Code Assist等）の利用規約に同意し、ご自身のアカウントで利用可能な方（18歳以上であることが推奨されます）。

---

## 1. 開発環境の要件　Beta1.6
まず、研修に必要なハードウェアとソフトウェアの要件を確認します。

### ハードウェア要件

| 項目 | 必須 | 推奨 | 備考 |
|------|------|------|------|
| **OS** | Windows 10 以上 | Windows 11 | Mac/Linux は未対応 |
| **RAM** | 8 GB | 16 GB | AI(torch) + Unity 同時起動のため |
| **ディスク空き** | 5 GB | 10 GB | .venv 約2.5GB + Unity 約1GB + 走行データ |
| **GPU** | 不要 | NVIDIA GeForce (CUDA対応) | AI訓練時にGPU推奨 |
| **ネットワーク** | **必須** | - | ライブラリインストール(約2.5GB)に必要 |

### ソフトウェア要件

| ソフトウェア | バージョン | 入手先 | インストール時の注意 |
|:---|:---|:---|:---|
| **Python** | 3.12 以上 (64bit) | [公式サイト](https://www.python.org/downloads/) | **「Add Python to PATH」に必ずチェック** |
| **VSCode** | 最新版 | [公式サイト](https://code.visualstudio.com/) | 日本語化パック推奨 |
| **Git** | 最新版 | [公式サイト](https://git-scm.com/) | デフォルト設定でOK |
| **Googleアカウント** | - | [作成ページ](https://accounts.google.com/signup) | NotebookLM, Gemini Code Assistで利用 |
| **GitHubアカウント**| - | [作成ページ](https://github.com/signup) | 開発環境のフォーク(Fork)で必須 |

> **⚠️ 最重要ポイント**
> Pythonインストール時に「**Add Python to PATH**」にチェックを入れ忘れると、`python` コマンドが認識されず、この後の手順がすべて失敗します。もし忘れてしまった場合は、一度アンインストールして再インストールしてください。

---

## 2. セットアップ手順
ここからが実際の手順です。自分のPC上に、自分専用の開発環境を構築していきます。

### Step 1: 自分用のリポジトリを準備 (Fork & Clone)
まず、公式リポジトリの自分用のコピー（フォーク）を作り、それを自分のPCにダウンロード（クローン）します。

1.  **フォーク (Fork)**
    - Webブラウザで [公式リポジトリ](https://github.com/AAgrandprix/virtual-robot-race) を開きます。
    - 画面右上にある **Fork** ボタンを押して、自分のGitHubアカウントにリポジトリのコピーを作成します。
    - これで、`https://github.com/あなたのユーザー名/virtual-robot-race` という、あなた専用のリポジトリができました。

2.  **クローン (Clone)**
    - 次に、**あなたのPC**にソースコードをダウンロードします。
    - 任意の作業フォルダでターミナルを開き、以下のコマンドを実行します。`あなたのユーザー名` の部分は、ご自身のGitHubユーザー名に置き換えてください。
    ```bash
    git clone https://github.com/あなたのユーザー名/virtual-robot-race.git
    ```
    - これで、PC上に `virtual-robot-race` というフォルダが作成されます。

### Step 2: 本家リポジトリとの連携設定 (Upstream)
今後、本家リポジトリがアップデートされた際に、その変更を簡単に取り込めるように「Upstream（上流）」という名前で本家を登録しておきます。

1.  ターミナルで、先ほど作成されたプロジェクトフォルダに移動します。
    ```bash
    cd virtual-robot-race
    ```
2.  以下のコマンドを実行して、本家リポジトリを `upstream` として登録します。
    ```bash
    git remote add upstream https://github.com/AAgrandprix/virtual-robot-race.git
    ```
3.  設定が正しくできたか確認しましょう。
    ```bash
    git remote -v
    ```
    以下のように `origin` (あなた自身のフォーク)と `upstream` (本家)の2種類が表示されれば成功です。
    ```
    origin    https://github.com/あなたのユーザー名/virtual-robot-race.git (fetch)
    origin    https://github.com/あなたのユーザー名/virtual-robot-race.git (push)
    upstream  https://github.com/AAgrandprix/virtual-robot-race.git (fetch)
    upstream  https://github.com/AAgrandprix/virtual-robot-race.git (push)
    ```

### Step 3: Python環境の構築
次に、ロボットを制御するPythonプログラムの開発環境を構築します。

1.  ターミナルで `Project_Beta` フォルダに移動します。
    ```bash
    cd Project_Beta
    ```
2.  環境構築スクリプト `setup_env.bat` を実行します。
    ```bash
    .\setup_env.bat
    ```
    このスクリプトは、仮想環境の作成と、必要なPythonライブラリのインストールを自動で行います。スクリプトが正常に完了すると、仮想環境が有効化された新しいターミナルウィンドウが自動的に開きます。プロンプトの行頭に `(.venv)` と表示されていれば成功です。

### Step 4: AIモデルのダウンロード
AIモードで使用する学習済みモデル（`model.pth`）を手動でダウンロードして配置します。

1.  [こちらから `model.pth` をダウンロード](https://drive.google.com/file/d/1NDL3A2lWDgXdy7OUWctyoR35jtYqthWD/view?usp=sharing)してください。
2.  ダウンロードした `model.pth` ファイルを、以下の**両方**のフォルダにコピーしてください。
    - `Project_Beta/Robot1/models/model.pth`
    - `Project_Beta/Robot2/models/model.pth`

### Step 5: VS Code の設定
最後に、`Visual Studio Code` の設定を行います。

1.  **プロジェクトフォルダを開く**
    - VS Codeを起動し、「ファイル」>「フォルダーを開く」から、`virtual-robot-race/Project_Beta` フォルダを選択します。

2.  **推奨拡張機能のインストール**
    - VS Codeの拡張機能マーケットプレイスで、以下の拡張機能を検索してインストールします。
      - **Python** (Microsoft提供): Python開発の必須ツールです。
      - **Gemini Code Assist** (Google Cloud提供): AIによるコーディング支援です。研修の後半で活用します。（※ログインにはGoogleアカウントを使用します）

3.  **Pythonインタープリターの選択**
    - `Ctrl + Shift + P` でコマンドパレットを開き、`Python: Select Interpreter` と入力します。
    - プロジェクトの仮想環境である `./.venv/Scripts/python.exe` を選択します。
    
    これで、VS Codeがプロジェクト用のPython環境を正しく認識するようになります。

---

## 3. GPU高速化 (オプション)
NVIDIA製のGPUを搭載したPCをお使いの場合、AIの学習や推論を高速化できます。

1.  `setup_env.bat`などで作成済みの仮想環境ターミナル（行頭に `(.venv)` が表示されているもの）を開きます。
2.  以下のコマンドを実行し、CPU版のPyTorchをアンインストール後、GPU(CUDA)版をインストールします。
    ```bash
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ```
3.  以下のコマンドでGPUが認識されているか確認します。
    ```bash
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```
    `CUDA available: True` と表示されれば成功です。

---

## 4. 最終動作確認
最後に、すべてのセットアップが正しく完了したか、テスト実行して確認しましょう。

1.  VS Codeで新しいターミナルを開きます（`Ctrl + Shift + @`）。プロンプトに `(.venv)` が表示されていることを確認してください。
2.  `main.py` を実行します。
    ```bash
    python main.py
    ```
3.  以下の現象が起きることを確認してください。
    - [ ] Unityのシミュレーター画面が**自動で起動**する。
    - [ ] 2台のロボットがコースを自動で走り始める（デフォルトはルールベースモード）。
    - [ ] レースが終了すると、`training_data`フォルダが開き、走行データが保存されている。

4.  保存されたデータを確認します。`Robot1/training_data/` の中に `run_YYYYMMDD_HHMMSS` という形式のフォルダができていれば、データ保存も成功です。

以上で開発環境の準備は完了です！お疲れ様でした。
次のチャプターに進みましょう！
