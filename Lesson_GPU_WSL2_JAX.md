# WSL2でのGPU加速セットアップ実践ガイド (v0.2.0)

## 概要

このドキュメントは、Windows環境でJAX GPU加速を使用するためのWSL2セットアップの実践記録です。実際のセットアップ過程で遭遇した問題と解決策をまとめています。

**実現したこと:**
- ✅ NVIDIA GeForce RTX 4060 Ti (16GB) をWSL2から使用
- ✅ JAX 0.10.0 + NumPyro 0.21.0 + PyMC 6.0.0のGPU環境構築
- ✅ NUTS推定のGPU高速化（CPU: 60-120分 → GPU: 15-30分）

## 前提条件

### ハードウェア
- NVIDIA GPU（今回: GeForce RTX 4060 Ti 16GB）
- Windows 10/11（今回: Windows 11）

### ソフトウェア
- NVIDIA GPU Driver（WSL2対応版）
- WSL2

## セットアップ手順

### ステップ1: 環境確認

#### 1.1 NVIDIA GPUドライバーの確認

Windows PowerShellで実行：

```powershell
nvidia-smi
```

**期待される出力:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.86                 Driver Version: 591.86         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 4060 Ti   WDDM  |   00000000:45:00.0 Off |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

✅ **確認ポイント:** Driver VersionとCUDA Versionが表示される

#### 1.2 WSL2のインストール状況確認

```powershell
wsl --status
```

**期待される出力:**
```
既定のディストリビューション: docker-desktop
既定のバージョン: 2
```

✅ **確認ポイント:** 既定のバージョンが「2」

```powershell
wsl --list --verbose
```

### ステップ2: Ubuntu on WSL2のインストール

#### 2.1 Ubuntuをインストール

```powershell
wsl --install -d Ubuntu
```

または最新LTS版を指定：

```powershell
wsl --install -d Ubuntu-22.04
```

#### 2.2 初期セットアップ

インストール後、Ubuntuが自動起動します。

**ユーザー名とパスワードの設定:**

```
Create a default Unix user account: yasun
New password: [パスワードを入力（画面には表示されない）]
Retype new password: [同じパスワードを再入力]
```

⚠️ **重要:** パスワード入力時は何も表示されませんが、正常に入力されています。

**Ubuntu改善プログラムへの参加:**

```
Would you like to opt-in to platform metrics collection (Y/n)? [y]
```

どちらでも構いません。`y`を推奨。

### ステップ3: WSL2からGPUにアクセス確認

Ubuntuターミナルで実行：

```bash
nvidia-smi
```

**期待される出力:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.57                 Driver Version: 591.86         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 4060 Ti     On  |   00000000:45:00.0 Off |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

✅ **確認ポイント:** WSL2内でもGPUが認識される

### ステップ4: プロジェクトのセットアップ

#### 4.1 重要: プロジェクトをLinux側にコピー

⚠️ **教訓: Windows側のファイルシステム（`/mnt/`）では仮想環境の作成に失敗する**

**NG例（失敗するパターン）:**
```bash
# Windows側のパスで直接作業（これは失敗する）
cd /mnt/i/ACT2025.5.26-2030/MVP/SURVIV/HeteroSurviv/hazard_effect_lingam
python3 -m venv .venv-cause-wsl2  # ❌ エラー: Operation not permitted
```

**OK例（正しいパターン）:**
```bash
# プロジェクトをLinux側にコピー
cd ~
mkdir -p projects
cp -r /mnt/i/ACT2025.5.26-2030/MVP/SURVIV/HeteroSurviv/hazard_effect_lingam ~/projects/

# Linux側で作業
cd ~/projects/hazard_effect_lingam
python3 -m venv .venv-cause-wsl2  # ✅ 成功
```

**理由:**
- `/mnt/`はWindows側のファイルシステム（NTFS）をマウントしているため、Linuxの一部機能（シンボリックリンク、パーミッション）が制限される
- Python仮想環境はシンボリックリンクを多用するため、NTFS上では正常に動作しない

#### 4.2 システムパッケージのインストール

```bash
# パッケージリストを更新
sudo apt update

# python3-venvをインストール（仮想環境作成に必要）
sudo apt install -y python3-venv
```

#### 4.3 Pythonバージョンの確認

```bash
python3 --version
```

**出力例:**
```
Python 3.14.0
```

Python 3.10以上であれば問題ありません。

#### 4.4 Python仮想環境の作成

```bash
# プロジェクトディレクトリに移動（Linux側）
cd ~/projects/hazard_effect_lingam

# 仮想環境を作成
python3 -m venv .venv-cause-wsl2

# 仮想環境をアクティベート
source .venv-cause-wsl2/bin/activate

# プロンプトが (.venv-cause-wsl2) で始まることを確認
```

#### 4.5 pipのアップグレード

```bash
pip install --upgrade pip
```

### ステップ5: JAXとNumPyroのインストール

#### 5.1 基本的な依存関係のインストール

```bash
pip install -r requirements.txt
```

このコマンドで以下がインストールされます：
- PyMC 6.0.0
- arviz
- pandas, numpy, scipy
- lingam（因果探索）
- その他の依存関係

#### 5.2 JAX CUDA版のインストール

```bash
# JAX CUDA 12.x版をインストール
pip install --upgrade "jax[cuda12]"
```

**出力例:**
```
Successfully installed jax-0.10.0 jaxlib-0.10.0
```

#### 5.3 NumPyroのインストール

```bash
pip install numpyro>=0.13.0
```

**出力例:**
```
Successfully installed numpyro-0.21.0
```

### ステップ6: GPU動作確認

```bash
python check_gpu.py
```

**期待される出力:**

```
================================================================================
GPU加速環境確認スクリプト (v0.2.0)
================================================================================
================================================================================
JAX GPU動作確認
================================================================================

[JAX バージョン]
  JAX: 0.10.0

[利用可能なデバイス]
  0: cuda:0

[デフォルトバックエンド]
  gpu

  ✅ GPU が利用可能です！

[計算テスト]
  行列積計算: (1000, 1000) @ (1000, 1000) = (1000, 1000)
  結果サンプル: 1000.0

================================================================================
NumPyro インストール確認
================================================================================

  ✅ NumPyro 0.21.0 がインストールされています

================================================================================
PyMC NumPyroサンプラー確認
================================================================================

  PyMC バージョン: 6.0.0

  簡単なモデルでNumPyroサンプラーをテスト中...
  ✅ NumPyroサンプラーが正常に動作しました
     サンプル数: 100
     推定平均: 0.794

================================================================================
総合判定
================================================================================

  🎉 GPU加速の準備が完了しました！

  config.yamlで以下の設定を有効化してください:
  ```yaml
  hazard_model:
    use_gpu: true
    jax_platform: 'gpu'
  ```
================================================================================
```

⚠️ **警告メッセージについて:**
```
E0519 17:24:35.966207    2895 cuda_executor.cc:1526] Could not get kernel mode driver version
```
このエラーメッセージは無視して構いません。WSL2環境特有のもので、GPU機能には影響しません。

### ステップ7: config.yamlの設定

Linux側のプロジェクトで設定を変更：

```bash
# エディタでconfig.yamlを開く
nano config.yaml
```

または、Windows側のエディタで編集してWSL2側にコピー。

**config.yamlの設定:**

```yaml
hazard_model:
  # NUTS推定パラメータ
  n_draws: 2000
  n_tune: 1000
  n_chains: 8
  n_cores: 8
  target_accept: 0.95
  random_seed: 42
  
  # GPU設定（v0.2.0: JAX/NumPyroによる高速化）
  use_gpu: true       # GPU使用を有効化
  jax_platform: "gpu"  # GPUモード
```

### ステップ8: パイプライン実行

```bash
# ハザードモデルからu_i抽出（GPU加速）
python main.py --step ui_extraction
```

**期待される出力:**

```
[NUTS推定開始]
  draws: 2000
  tune: 1000
  chains: 8
  cores: 8
  target_accept: 0.95
  🚀 GPU加速: 有効 (JAXバックエンド: gpu)
  ⚠ JAX GPU/TPUモードでは chains=1 に設定（並列化はJAX内部で処理）
  推定中... (GPU使用時: 10-30分程度、CPU: 60-120分程度)
```

## トラブルシューティング

### 問題1: 仮想環境作成時に「Operation not permitted」エラー

**症状:**
```bash
cd /mnt/i/...
python3 -m venv .venv-cause-wsl2
# Error: Operation not permitted
```

**原因:** Windows側のファイルシステム（/mnt/）で仮想環境を作成しようとしている

**解決策:**
```bash
# プロジェクトをLinux側にコピー
cp -r /mnt/i/path/to/project ~/projects/
cd ~/projects/project_name
python3 -m venv .venv-cause-wsl2  # これで成功
```

### 問題2: python3-venvパッケージがない

**症状:**
```bash
python3 -m venv .venv-cause-wsl2
# Error: ensurepip is not available
```

**解決策:**
```bash
sudo apt install -y python3-venv
```

### 問題3: Python 3.12が見つからない

**症状:**
```bash
sudo apt install python3.12
# Error: Unable to locate package python3.12
```

**解決策:**
Ubuntu 25.04（Resolute）ではPython 3.14がデフォルトです。`python3`コマンドを使用してください：
```bash
python3 --version  # Python 3.14.0
python3 -m venv .venv-cause-wsl2
```

### 問題4: GPU動作確認で「Could not get kernel mode driver version」警告

**症状:**
```
E0519 17:24:35.966207    2895 cuda_executor.cc:1526] Could not get kernel mode driver version
```

**解決策:**
これはWSL2環境特有の警告で、**無視して構いません**。GPUは正常に動作します。

確認方法：
```bash
python -c "import jax; print(jax.devices())"
# [cuda:0] と表示されればGPU利用可能
```

## 結果のWindows側へのコピー

GPU処理が完了したら、結果をWindows側にコピーバックできます：

```bash
# 結果をWindows側にコピー
cp -r ~/projects/hazard_effect_lingam/output /mnt/i/ACT2025.5.26-2030/MVP/SURVIV/HeteroSurviv/hazard_effect_lingam/

# または、特定のファイルのみコピー
cp ~/projects/hazard_effect_lingam/output/pump_heterogeneity.csv /mnt/i/ACT2025.5.26-2030/MVP/SURVIV/HeteroSurviv/hazard_effect_lingam/output/
```

## パフォーマンス比較

### 環境
- **GPU**: NVIDIA GeForce RTX 4060 Ti (16GB VRAM)
- **CPU**: Intel/AMD 8コア
- **データ**: 64ポンプ、92,861サンプル

### NUTS推定時間

| 環境 | 設定 | 推定時間 | 高速化率 |
|------|------|---------|---------|
| Windows (CPU) | PyMC標準サンプラー | 60-120分 | 1.0x (基準) |
| WSL2 (GPU) | JAX/NumPyroサンプラー | 15-30分 | **3-4倍高速** |

### リソース使用状況

**CPU版:**
- CPU使用率: 80-100%（8コア）
- メモリ: 8-12GB
- GPU使用率: 0%

**GPU版:**
- CPU使用率: 20-40%
- メモリ: 6-8GB
- **GPU使用率: 60-80%**
- **GPU メモリ: 4-6GB / 16GB**

タスクマネージャーでGPU使用状況を確認できます：
- 「パフォーマンス」→「GPU 1」（NVIDIA）
- 「3D」「Copy」のグラフでGPU利用が確認できる

## まとめ

### 成功のポイント

1. ✅ **WSL2を使用する**: Windows (ネイティブ) ではJAX GPUサポートなし
2. ✅ **プロジェクトをLinux側にコピー**: `/mnt/`経由では仮想環境作成に失敗
3. ✅ **python3-venvをインストール**: Ubuntu最小インストールには含まれない
4. ✅ **JAX CUDA版を正しくインストール**: `pip install "jax[cuda12]"`
5. ✅ **GPU動作確認**: `python check_gpu.py` で事前確認

### 学んだこと

- WSL2はWindows上でLinux環境を実行できる強力なツール
- NVIDIA GPUドライバー（WSL2対応版）があれば、WSL2からGPUに直接アクセス可能
- ファイルシステムの違い（NTFS vs ext4）がPython仮想環境に影響する
- JAXのGPU加速により、NUTS推定が大幅に高速化される

### 次のステップ

1. **因果探索の実行**: `python main.py --step causal_discovery`
2. **結果の分析**: `output/`ディレクトリの結果ファイルを確認
3. **高度な手法の適用**: NonlinearLiNGAM、BayesianMixture（[Lesson_Advanced_Causal_Methods.md](Lesson_Advanced_Causal_Methods.md)参照）

## 参考資料

- [Lesson_GPU_Acceleration.md](Lesson_GPU_Acceleration.md) - GPU加速の詳細ガイド
- [README.md](README.md) - プロジェクト概要
- JAX Documentation: https://jax.readthedocs.io/
- NumPyro Documentation: https://num.pyro.ai/
- WSL2 Documentation: https://learn.microsoft.com/en-us/windows/wsl/

---

**作成日**: 2026年5月19日  
**バージョン**: v0.2.0  
**テスト環境**: Windows 11 + WSL2 Ubuntu 25.04 + NVIDIA GeForce RTX 4060 Ti
