# 高度な因果探索手法ガイド (v0.2.0)

## 概要

v0.2.0では、DirectLiNGAMに加えて、2018年以降の高度な因果探索手法をサポートしています。これにより、線形性の仮定を緩和し、より複雑な因果関係を検出できます。

## サポートする手法

### 1. DirectLiNGAM（標準・推奨）

**特徴:**

- 線形因果関係を仮定
- 大規模データに対応（10万サンプル以上）
- 高速・安定
- ガウス・非ガウスの両方で機能

**適用条件:**

- 線形関係が主である場合
- 大規模データセット
- 高速な結果が必要な場合

**使用方法:**

```yaml
# config.yaml
lingam:
  algorithm: "DirectLiNGAM"
  random_state: 42

advanced_methods:
  enabled: false  # DirectLiNGAMのみ使用
```

**実行:**

```bash
python main.py --step causal_discovery
```

---

### 2. NonlinearLiNGAM（非線形因果探索）

**特徴:**

- 非線形因果関係を検出
- カーネルベースの独立性テスト
- 計算コストが高い（DirectLiNGAMの10-100倍）
- サブセット戦略で高速化

**適用条件:**

- 非線形関係が疑われる場合
- サブセット（1,000-10,000サンプル）で検証
- 時間的余裕がある場合

**使用方法:**

```yaml
# config.yaml
lingam:
  algorithm: "DirectLiNGAM"  # ベースライン用

advanced_methods:
  enabled: true
  method: "NonlinearLiNGAM"
  
  nonlinear:
    subset_size: 1000  # 各グループから1,000サンプル抽出
    method: "NonlinearLiNGAM"
```

**実行:**

```bash
# 標準のDirectLiNGAM
python main.py --step causal_discovery

# NonlinearLiNGAMで追加実験（手動実行）
# 以下のPythonスクリプトを作成:
```

```python
# run_nonlinear.py
from src.causal_discovery_nonlinear import run_nonlinear_causal_discovery
from src.feature_engineering import FeatureEngineer
import pandas as pd
import yaml

# 設定読み込み
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# データ読み込み
df_scaled = pd.read_csv('output/scaled_features.csv')
feature_names = FeatureEngineer.get_feature_names()
kpi_col = config['kpi']['primary_label']

# グループ別に非線形探索を実行
for group in ['positive', 'negative']:
    print(f"\n{'='*80}")
    print(f"NonlinearLiNGAM: {group} グループ")
    print(f"{'='*80}")
  
    df_group = df_scaled[df_scaled['u_group'] == group]
  
    results, effects = run_nonlinear_causal_discovery(
        features_df=df_group,
        feature_cols=feature_names,
        kpi_col=kpi_col,
        method="NonlinearLiNGAM",
        subset_size=1000,
        random_state=42,
        output_path=f"output/causal_results_{group}_nonlinear.pkl",
        verbose=True
    )
  
    # 効果をCSV保存
    effects.to_csv(f"output/kpi_effects_{group}_nonlinear.csv", index=False)
    print(f"\n保存完了: output/kpi_effects_{group}_nonlinear.csv")
```

**実行:**

```bash
python run_nonlinear.py
```

**結果の解釈:**

- DirectLiNGAMとの比較で、新たに検出されたエッジを確認
- エッジ一致率が80%以上なら、線形近似で十分
- 一致率が低い場合は、非線形関係が重要

---

### 3. BayesianMixture（ベイズ混合モデル）

**特徴:**

- 分布を仮定しない（Student-t分布）
- 外れ値に対してロバスト
- 不確実性を定量化（95%信用区間）
- PyMCによるNUTS推定

**適用条件:**

- 外れ値が多い場合
- 不確実性の定量化が必要な場合
- ベイズ統計の解釈が求められる場合

**使用方法:**

```yaml
# config.yaml
lingam:
  algorithm: "DirectLiNGAM"  # ベースライン用

advanced_methods:
  enabled: true
  method: "BayesianMixture"
  
  bayesian:
    n_draws: 1000  # NUTSドロー数
    n_tune: 500    # チューニング期間
    n_chains: 4    # チェイン数
    target_accept: 0.90
    use_student_t: true  # 外れ値対応
```

**実行:**

```python
# run_bayesian.py
from src.causal_discovery_bayesian import run_bayesian_causal_discovery
from src.feature_engineering import FeatureEngineer
import pandas as pd
import yaml

# 設定読み込み
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# データ読み込み
df_scaled = pd.read_csv('output/scaled_features.csv')
feature_names = FeatureEngineer.get_feature_names()
kpi_col = config['kpi']['primary_label']

# グループ別にベイズ探索を実行
for group in ['positive', 'negative']:
    print(f"\n{'='*80}")
    print(f"BayesianMixture: {group} グループ")
    print(f"{'='*80}")
  
    df_group = df_scaled[df_scaled['u_group'] == group]
  
    results, effects = run_bayesian_causal_discovery(
        features_df=df_group,
        feature_cols=feature_names,
        kpi_col=kpi_col,
        n_draws=1000,
        n_tune=500,
        n_chains=4,
        target_accept=0.90,
        random_seed=42,
        use_student_t=True,
        output_path=f"output/causal_results_{group}_bayesian.pkl",
        verbose=True
    )
  
    # 効果をCSV保存
    effects.to_csv(f"output/kpi_effects_{group}_bayesian.csv", index=False)
    print(f"\n保存完了: output/kpi_effects_{group}_bayesian.csv")
```

**実行:**

```bash
python run_bayesian.py
```

**結果の解釈:**

- `effect`: 因果効果の事後平均
- `std`: 事後標準偏差（不確実性）
- `lower_95`, `upper_95`: 95%信用区間
- `significant`: 信用区間が0を含まない場合True

---

## 手法の選択ガイド

### フローチャート

```
データサイズは？
├─ 小規模（<10,000）
│   └─ 非線形関係が疑われる？
│       ├─ YES → NonlinearLiNGAM
│       └─ NO → DirectLiNGAM
│
└─ 大規模（≥10,000）
    └─ 外れ値が多い？
        ├─ YES → BayesianMixture（サブセット）
        └─ NO → DirectLiNGAM（推奨）
```

### 比較表

| 手法            | 線形性   | データ規模 | 実行時間   | 解釈性     | 推奨度     |
| --------------- | -------- | ---------- | ---------- | ---------- | ---------- |
| DirectLiNGAM    | 線形のみ | 大規模OK   | 15分       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| NonlinearLiNGAM | 非線形OK | 中規模     | 1-3時間    | ⭐⭐⭐     | ⭐⭐⭐     |
| BayesianMixture | 線形のみ | 中規模     | 30分-2時間 | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   |

---

## 実践ワークフロー

### ステップ1: ベースライン分析（DirectLiNGAM）

```bash
# まずDirectLiNGAMで全データを分析
python main.py --step causal_discovery
```

結果:

- `output/kpi_effects_positive.csv`
- `output/kpi_effects_negative.csv`

### ステップ2: 非線形検証（オプション）

```bash
# 非線形関係の検証（サブセット）
python run_nonlinear.py
```

結果:

- `output/kpi_effects_positive_nonlinear.csv`
- `output/kpi_effects_negative_nonlinear.csv`

**比較:**

```python
import pandas as pd

# DirectLiNGAM vs NonlinearLiNGAM
linear = pd.read_csv('output/kpi_effects_positive.csv')
nonlinear = pd.read_csv('output/kpi_effects_positive_nonlinear.csv')

# 共通特徴量の効果比較
merged = linear.merge(nonlinear, on='feature', suffixes=('_linear', '_nonlinear'))
merged['diff'] = merged['effect_nonlinear'] - merged['effect_linear']
merged.sort_values('diff', key=abs, ascending=False).head(10)
```

**判定基準:**

- エッジ一致率 > 80%: 線形近似で十分
- エッジ一致率 < 80%: 非線形効果が重要

### ステップ3: ロバスト性検証（オプション）

```bash
# ベイズ混合モデルで外れ値対応
python run_bayesian.py
```

結果:

- `output/kpi_effects_positive_bayesian.csv` (信用区間付き)
- `output/kpi_effects_negative_bayesian.csv`

**比較:**

```python
import pandas as pd

# DirectLiNGAM vs BayesianMixture
linear = pd.read_csv('output/kpi_effects_positive.csv')
bayesian = pd.read_csv('output/kpi_effects_positive_bayesian.csv')

# 有意性の比較
# DirectLiNGAM: 効果の絶対値が大きい特徴量
# BayesianMixture: significant=Trueの特徴量

print(f"DirectLiNGAM 上位10特徴量: {len(linear.head(10))}")
print(f"BayesianMixture 有意な特徴量: {bayesian['significant'].sum()}")
```

---

## トラブルシューティング

### 問題1: NonlinearLiNGAMが遅すぎる

**原因:** サンプル数が多すぎる

**解決策:**

```yaml
# config.yaml
advanced_methods:
  nonlinear:
    subset_size: 500  # サブセットサイズを減らす
```

### 問題2: BayesianMixtureが収束しない（R̂ ≥ 1.1）

**原因:** チェイン数やドロー数が不足

**解決策:**

```yaml
# config.yaml
advanced_methods:
  bayesian:
    n_draws: 2000  # ドロー数を増やす
    n_tune: 1000   # チューニング期間を延長
    n_chains: 6    # チェイン数を増やす
```

### 問題3: NonlinearLiNGAMとDirectLiNGAMの結果が大きく異なる

**原因:** 真の因果関係が非線形である可能性

**対応:**

1. サブセットサイズを増やして再実行
2. 非線形特徴量エンジニアリングを追加
3. 論文では両方の結果を報告

---

## 参考文献

1. Shimizu et al. (2011). "DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model"
2. Hoyer et al. (2009). "Nonlinear causal discovery with additive noise models"
3. Peters et al. (2014). "Causal inference by using invariant prediction: identification and confidence intervals"
4. Gretton et al. (2008). "Kernel-based tests for joint independence"

---

**作成日**: 2026年5月19日
**バージョン**: v0.2.0
