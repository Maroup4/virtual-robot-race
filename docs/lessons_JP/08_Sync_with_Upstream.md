# 8. Upstreamとの同期

aira公式リポジトリは、新機能の追加やバグ修正のために定期的に更新されます。
あなたが [レッスン00](00_Preparation.md) でフォークしたリポジトリは、フォーク時点のスナップショットです。そのままでは公式の更新は自動的に反映されません。

このレッスンでは、**自分のアルゴリズムを守りながら、公式の最新版を取り込む方法**を学びます。

**学習目標:**
- `upstream` と `origin` の違いを理解する
- `git fetch` + `git merge` で公式の更新を安全に取り込む手順を習得する
- `git stash` で作業中の変更を退避・復元する
- コンフリクトが発生した場合の対処法を知る

---

## Upstreamとは？

Gitでは、リモートリポジトリに名前をつけて管理します。

| 名前 | 指す場所 | 役割 |
|------|---------|------|
| `origin` | あなたのフォーク（GitHub上） | 自分の作業を保存・共有する場所 |
| `upstream` | aira公式リポジトリ | 最新のシミュレーターを取得する場所 |

3者の関係を図で整理すると：

```
  公式 aira リポジトリ (upstream)
        │
        │  git fetch upstream
        │  git merge upstream/main
        ▼
  あなたのPC（ローカル）
        │
        │  git push origin main
        ▼
  あなたの GitHub フォーク (origin)
```

フォーク直後は `origin` しか存在しません。`upstream` を手動で追加することで、公式の更新を取り込めるようになります。

---

## 手順

### Step 1: upstreamを追加する（初回のみ）

```bash
git remote add upstream https://github.com/aira-race/virtual-robot-race.git
```

追加できたか確認します。

```bash
git remote -v
```

以下のように表示されれば成功です。

```
origin    https://github.com/あなたのユーザー名/virtual-robot-race.git (fetch)
origin    https://github.com/あなたのユーザー名/virtual-robot-race.git (push)
upstream  https://github.com/aira-race/virtual-robot-race.git (fetch)
upstream  https://github.com/aira-race/virtual-robot-race.git (push)
```

---

### Step 2: ローカルの変更を退避する

`config.txt` など、自分が編集中のファイルがある場合、そのままでは merge がブロックされます。

```
error: Your local changes to the following files would be overwritten by merge:
        config.txt
```

このエラーが出たら、まず変更を一時退避します。

```bash
git stash
```

> **💡 `git stash` とは**: 作業中の変更を一時的に棚上げするコマンドです。merge 後に `git stash pop` で元に戻せます。**退避した内容は消えません。**
>
> **代替手順**: `git stash` の代わりに `git add` → `git commit` してから merge しても構いません。ただし「まだ未完成のコードをコミットしたくない」場合に `git stash` が便利です。
>
> **万が一の保険**: VS Code には「タイムライン」という機能があり、ファイルの変更履歴をローカルに自動保存しています。ファイルを開いた状態で画面左下の「タイムライン」パネルを開くと、過去の状態に戻せます。`git stash` や `git merge` の操作が不安な場合は、先にタイムラインで現在の状態を確認しておくと安心です。

---

### Step 3: 公式の最新版を取得・マージする

aira公式リポジトリの更新を取り込みます。

```bash
git fetch upstream
git merge upstream/main
```

- `git fetch upstream` — 公式の変更をローカルに取得します（まだ自分のコードには反映されません）
- `git merge upstream/main` — 取得した変更を自分のブランチに統合します

> **💡 `git pull upstream main` ではなく2つに分けている理由**: `pull` は fetch + merge を一気に行います。2つに分けることで、fetch 後に `git log upstream/main` で何が変わったか確認してから merge できるため、より安全です。
>
> **💡 自分のコードは消えません**: `merge` は公式の変更と自分の変更を**統合**します。同じファイルの同じ行を両方が編集していた場合のみ「コンフリクト（競合）」が発生します（後述）。

merge が完了したら、退避した変更を戻します。

```bash
git stash pop
```

---

### Step 4: 自分のoriginに反映する

マージが完了したら、自分のGitHub上のフォークにも反映します。

```bash
git push origin main
```

---

## コンフリクト（競合）が発生した場合

公式の更新が、あなたが編集したファイルの同じ箇所に変更を加えていた場合、コンフリクトが発生します。

> **💡 やり直したい場合**: コンフリクトを解決する前であれば `git merge --abort` でマージ前の状態に戻せます。「よく分からなくなった」と思ったらこれを使いましょう。

コンフリクトが起きたファイルはこのような表示になります。

```
<<<<<<< HEAD
# あなたの変更
=======
# 公式の変更
>>>>>>> upstream/main
```

**対処の流れ：**

1. VSCodeでコンフリクトしているファイルを開く
2. 残したい内容を選ぶ（両方残すことも可能）
   - ファイル上部に表示される **「現在の変更を取り込む」「受信した変更を取り込む」「両方の変更を取り込む」** ボタンをクリックするのが最も簡単です
   - または `<<<<<<`、`=======`、`>>>>>>>` の行を手動で削除して保存
3. `git add` → `git commit` で完了

> **💡 迷ったらAIアシスタントに丸投げする**: コンフリクトが発生したファイルの内容をそのままコピーして、以下のように聞くのが最も確実です。
> ```
> 以下のコンフリクトが発生しました。
> 「自分のアルゴリズムを守りながら公式の更新を取り込む」という目的で、
> どちらを残すべきか、または両方どう統合すべきか教えてください。
>
> （コンフリクトしたファイルの内容をここに貼る）
> ```
> `<<<<<<` などの記号を誤って残したり、消すべき行を間違えたりする心配がなくなります。

---

## まとめ

| やること | コマンド | タイミング |
|---------|---------|-----------|
| upstreamを登録 | `git remote add upstream <URL>` | 初回のみ |
| ローカル変更を退避 | `git stash` | merge前（変更がある場合） |
| 公式の更新を取得 | `git fetch upstream` | 更新があるとき |
| 自分のブランチに統合 | `git merge upstream/main` | 更新があるとき |
| 退避した変更を戻す | `git stash pop` | merge完了後 |
| 自分のフォークに反映 | `git push origin main` | merge完了後 |
| マージをやり直す | `git merge --abort` | コンフリクト解決前 |

aira のバージョンアップ情報は [GitHub Releases](https://github.com/aira-race/virtual-robot-race/releases) や [X (@RaceYourAlgo)](https://x.com/RaceYourAlgo) でお知らせします。

---

> **❓ うまくいかない場合は？**
> [NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e) にエラーメッセージをそのまま貼り付けて質問してください。

---

⬅️ [前のレッスン: 07_How_to_Join_Race.md（レースに参加する）](07_How_to_Join_Race.md) ｜ [用語集](99_Glossary.md)
