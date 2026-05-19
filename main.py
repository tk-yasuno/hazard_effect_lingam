"""
メインパイプライン

ポンプ設備LiNGAM因果探索MVPの統合パイプライン
v0.1.2: マルコフ劣化ハザードのランダム効果をKPIとして使用
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import warnings

# 自作モジュールのインポート
from src.data_preprocessing import load_and_preprocess_data
from src.feature_engineering import extract_features_from_time_series, FeatureEngineer
from src.causal_discovery import run_causal_discovery
from src.causal_discovery_nonlinear import run_nonlinear_causal_discovery
from src.causal_discovery_bayesian import run_bayesian_causal_discovery
from src.visualization import CausalVisualizer, generate_interpretation_report
from src.validation import run_bootstrap_validation

# v0.1.2: ハザードモデルモジュール
from src.hazard_model import (
    HazardConfig,
    prepare_hazard_data,
    build_hazard_model,
    run_nuts_sampling,
    extract_random_effects,
    assign_ui_groups
)

warnings.filterwarnings('ignore')


class PumpCausalPipeline:
    """
    ポンプ設備因果探索パイプラインクラス
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定の読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.base_dir = Path(__file__).parent
        self.output_dir = self.base_dir / self.config['data']['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ保持用
        self.df_preprocessed = None
        self.df_features = None
        self.df_scaled = None
        self.causal_results = None
        self.effects_df = None
        
        # v0.1.2: ハザードモデル関連
        self.hazard_config = None
        self.hazard_data = None
        self.hazard_trace = None
        self.ui_heterogeneity = None
        
        print("=" * 80)
        print("ポンプ設備LiNGAM因果探索MVP v0.1.2")
        print("=" * 80)
    
    def run_preprocessing(self):
        """
        ステップ1-2: データ前処理
        """
        print("\n" + "=" * 80)
        print("ステップ1-2: データ前処理")
        print("=" * 80)
        
        time_series_path = self.base_dir / self.config['data']['labeled_time_series']
        selected_equipment_path = self.base_dir / self.config['data']['selected_equipment']
        
        self.df_preprocessed = load_and_preprocess_data(
            time_series_path=str(time_series_path),
            selected_equipment_path=str(selected_equipment_path),
            outlier_threshold=self.config['preprocessing']['outlier_threshold'],
            interpolation_method=self.config['preprocessing']['interpolation_method'],
            max_missing_ratio=self.config['preprocessing']['max_missing_ratio'],
            verbose=True
        )
        
        # 保存
        output_path = self.output_dir / "preprocessed_data.csv"
        self.df_preprocessed.to_csv(output_path, index=False)
        print(f"\n[保存] {output_path}")
    
    # ========================================================================
    # v0.1.2: ハザードモデル関連メソッド
    # ========================================================================
    
    def run_hazard_preprocessing(self):
        """
        Phase 1.3: ハザードモデル用データ前処理
        """
        print("\n" + "=" * 80)
        print("Phase 1.3: ハザードモデル用データ前処理")
        print("=" * 80)
        
        # HazardConfigの初期化
        self.hazard_config = HazardConfig(
            base_dir=self.base_dir,
            output_dir=self.output_dir
        )
        
        # config.yamlの設定を反映
        if 'hazard_model' in self.config:
            hm_config = self.config['hazard_model']
            self.hazard_config.n_states = hm_config.get('n_states', 8)
            self.hazard_config.min_data_points = hm_config.get('min_data_points', 30)
            self.hazard_config.min_delta_t = hm_config.get('min_delta_t', 1)
            self.hazard_config.max_delta_t = hm_config.get('max_delta_t', 365)
            self.hazard_config.n_draws = hm_config.get('n_draws', 3000)
            self.hazard_config.n_tune = hm_config.get('n_tune', 2000)
            self.hazard_config.n_chains = hm_config.get('n_chains', 6)
            self.hazard_config.n_cores = hm_config.get('n_cores', 6)
            self.hazard_config.target_accept = hm_config.get('target_accept', 0.95)
            # GPU設定 (v0.2.0)
            self.hazard_config.use_gpu = hm_config.get('use_gpu', False)
            self.hazard_config.jax_platform = hm_config.get('jax_platform', 'cpu')
        
        # データ前処理の実行
        self.hazard_data = prepare_hazard_data(self.hazard_config)
    
    def run_hazard_estimation(self):
        """
        Phase 2.1: NUTS推定の実行
        """
        print("\n" + "=" * 80)
        print("Phase 2.1: NUTS推定の実行")
        print("=" * 80)
        
        if self.hazard_data is None:
            raise ValueError("ハザードモデル用データが準備されていません。run_hazard_preprocessing()を先に実行してください。")
        
        # PyMCモデルの構築
        model = build_hazard_model(self.hazard_data)
        
        # NUTS推定
        self.hazard_trace = run_nuts_sampling(model, self.hazard_config)
        
        # トレースの保存
        trace_path = self.hazard_config.get_absolute_path(self.hazard_config.trace_file)
        self.hazard_trace.to_netcdf(str(trace_path))
        print(f"\n[保存] {trace_path}")
    
    def run_ui_extraction_and_grouping(self):
        """
        Phase 2.2-2.3: u_i抽出とグループ分け
        """
        print("\n" + "=" * 80)
        print("Phase 2.2-2.3: u_i抽出とグループ分け")
        print("=" * 80)
        
        if self.hazard_trace is None:
            raise ValueError("NUTS推定が実行されていません。run_hazard_estimation()を先に実行してください。")
        
        # u_iの抽出
        self.ui_heterogeneity = extract_random_effects(
            self.hazard_trace,
            self.hazard_data,
            self.hazard_config
        )
        
        # グループ分け
        self.ui_heterogeneity = assign_ui_groups(
            self.ui_heterogeneity,
            self.hazard_config
        )
        
        # グループ情報を含めて再保存
        output_path = self.output_dir / "pump_heterogeneity.csv"
        self.ui_heterogeneity.to_csv(output_path, index=False)
        print(f"  グループ情報を含めて再保存: {output_path}")
    
    def run_feature_ui_merge(self):
        """
        Phase 2.3: 特徴量データとu_iの結合
        """
        print("\n" + "=" * 80)
        print("Phase 2.3: 特徴量データとu_iの結合")
        print("=" * 80)
        
        if self.df_features is None:
            raise ValueError("特徴量が生成されていません。run_feature_engineering()を先に実行してください。")
        
        if self.ui_heterogeneity is None:
            raise ValueError("u_iが抽出されていません。run_ui_extraction_and_grouping()を先に実行してください。")
        
        # equipment_idでマージ
        print(f"\n[マージ前]")
        print(f"  特徴量データ: {len(self.df_features)} レコード")
        print(f"  u_i データ: {len(self.ui_heterogeneity)} ポンプ")
        
        # u_i列を追加（equipment_idでjoin）
        df_merged = self.df_features.merge(
            self.ui_heterogeneity[['equipment_id', 'u_mean', 'u_group']],
            on='equipment_id',
            how='inner'
        )
        
        # KPI列の追加
        df_merged['u_i_target'] = df_merged['u_mean']
        
        print(f"\n[マージ後]")
        print(f"  結合後レコード数: {len(df_merged)}")
        print(f"  u_i_target 統計:")
        print(df_merged['u_i_target'].describe())
        print(f"\n  グループ別統計:")
        print(df_merged['u_group'].value_counts())
        
        # 特徴量データフレームを更新
        self.df_features = df_merged
        
        # 保存
        output_path = self.output_dir / "features_with_ui.csv"
        df_merged.to_csv(output_path, index=False)
        print(f"\n[保存] {output_path}")
    
    # ========================================================================
    # 既存メソッド（v0.1.0）
    # ========================================================================
    
    def run_feature_engineering(self):
        """
        ステップ3-5: 特徴量エンジニアリング
        """
        print("\n" + "=" * 80)
        print("ステップ3-5: 特徴量エンジニアリング")
        print("=" * 80)
        
        if self.df_preprocessed is None:
            raise ValueError("前処理が実行されていません。run_preprocessing()を先に実行してください。")
        
        # n_jobsをconfigから取得（デフォルトは-1で全コア使用）
        n_jobs = self.config.get('misc', {}).get('n_jobs', -1)
        
        self.df_features = extract_features_from_time_series(
            time_series_df=self.df_preprocessed,
            lookback_days=self.config['feature_engineering']['lookback_days'],
            min_data_points=self.config['feature_engineering']['min_data_points'],
            n_jobs=n_jobs,
            verbose=True
        )
        
        # 保存
        output_path = self.output_dir / "features_90d.csv"
        self.df_features.to_csv(output_path, index=False)
        print(f"\n[保存] {output_path}")
    
    def run_scaling_and_correlation_check(self):
        """
        ステップ6: スケーリングと相関チェック
        """
        print("\n" + "=" * 80)
        print("ステップ6: スケーリングと相関チェック")
        print("=" * 80)
        
        if self.df_features is None:
            raise ValueError("特徴量生成が実行されていません。")
        
        # 特徴量列を取得
        feature_names = FeatureEngineer.get_feature_names()
        kpi_col = self.config['kpi']['primary_label']
        
        # 必要な列を選択（u_groupがある場合は含める）
        required_cols = feature_names + [kpi_col]
        if 'u_group' in self.df_features.columns:
            required_cols.append('u_group')
        
        df_for_scaling = self.df_features[required_cols].copy()
        
        print(f"\n[スケーリング前]")
        print(f"  レコード数: {len(df_for_scaling)}")
        print(f"  特徴量数: {len(feature_names)}")
        print(f"  欠損率: {df_for_scaling.isnull().sum().sum() / (len(df_for_scaling) * len(required_cols)) * 100:.2f}%")
        
        # NaN除去
        df_for_scaling = df_for_scaling.dropna()
        print(f"  NaN除去後: {len(df_for_scaling)} レコード")
        
        # 標準化
        scaler = StandardScaler()
        X_features = df_for_scaling[feature_names].values
        X_scaled = scaler.fit_transform(X_features)
        
        # データフレーム化
        df_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=df_for_scaling.index)
        df_scaled[kpi_col] = df_for_scaling[kpi_col].values
        
        # u_groupがある場合は追加
        if 'u_group' in df_for_scaling.columns:
            df_scaled['u_group'] = df_for_scaling['u_group'].values
        
        print(f"\n[標準化完了]")
        print(f"  平均（サンプル）: {X_scaled[:, 0].mean():.6f}")
        print(f"  標準偏差（サンプル）: {X_scaled[:, 0].std():.6f}")
        
        # 相関チェック
        corr_threshold = self.config['scaling']['correlation_threshold']
        corr_matrix = df_scaled[feature_names].corr()
        
        # 高相関ペアを検出
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                    high_corr_pairs.append((
                        feature_names[i],
                        feature_names[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        print(f"\n[相関チェック]")
        print(f"  相関閾値: {corr_threshold}")
        print(f"  高相関ペア数: {len(high_corr_pairs)}")
        
        if len(high_corr_pairs) > 0:
            print("\n  高相関ペア（トップ5）:")
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                print(f"    {feat1} <-> {feat2}: {corr:.3f}")
        
        # 分散チェック
        variance_threshold = self.config['scaling']['variance_threshold']
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(df_scaled[feature_names])
        
        low_variance_features = [
            name for name, var in zip(feature_names, selector.variances_)
            if var < variance_threshold
        ]
        
        print(f"\n[分散チェック]")
        print(f"  分散閾値: {variance_threshold}")
        print(f"  低分散特徴量数: {len(low_variance_features)}")
        
        if len(low_variance_features) > 0:
            print(f"  低分散特徴量: {low_variance_features}")
        
        self.df_scaled = df_scaled
        
        # 保存
        output_path = self.output_dir / "scaled_features.csv"
        df_scaled.to_csv(output_path, index=False)
        print(f"\n[保存] {output_path}")
    
    def run_causal_discovery(self):
        """
        ステップ7-8 (Phase 3.2-3.3): LiNGAM因果探索（グループ別対応・v0.2.0: 2グループ）
        """
        print("\n" + "=" * 80)
        print("ステップ7-8: LiNGAM因果探索（v0.2.0: Binary Grouping）")
        print("=" * 80)
        
        if self.df_scaled is None:
            raise ValueError("スケーリングが実行されていません。")
        
        feature_names = FeatureEngineer.get_feature_names()
        kpi_col = self.config['kpi']['primary_label']
        
        # グループ別分析の確認
        u_group_analysis = self.config.get('kpi', {}).get('u_group_analysis', {})
        enabled = u_group_analysis.get('enabled', False)
        groups = u_group_analysis.get('groups', ['positive', 'negative'])  # v0.2.0: 2グループ
        
        if enabled and 'u_group' in self.df_scaled.columns:
            print(f"\n[グループ別分析モード（v0.2.0: Sign-Based Binary Grouping）]")
            print(f"  対象グループ: {groups}")
            
            # グループデータを辞書に格納
            group_data = {}
            for group in groups:
                df_group = self.df_scaled[self.df_scaled['u_group'] == group].copy()
                if len(df_group) >= 30:
                    group_data[group] = df_group
                    print(f"  {group}: {len(df_group)} サンプル")
                else:
                    print(f"  ⚠ 警告: {group} のデータ点数が少なすぎます（{len(df_group)} < 30）。スキップします。")
            
            # v0.2.0: SMOTE無効化（符号ベース2グループで不要）
            # SMOTE拡張をスキップ
            imbalance_enabled = self.config.get('imbalance_correction', {}).get('enabled', False)
            if imbalance_enabled:
                print(f"\n  ⚠ 注意: imbalance_correction.enabled=true ですが、v0.2.0ではSMOTEは使用しません。")
            
            # グループ別に因果探索を実行
            self.causal_results = {}
            self.effects_df = {}
            
            for group in group_data.keys():
                print(f"\n--- グループ: {group} ---")
                
                df_group = group_data[group]
                print(f"  レコード数: {len(df_group)}")
                
                # 因果探索の実行
                results, effects = run_causal_discovery(
                    features_df=df_group,
                    feature_cols=feature_names,
                    kpi_col=kpi_col,
                    algorithm=self.config['lingam']['algorithm'],
                    random_state=self.config['lingam']['random_state'],
                    output_path=str(self.output_dir / f"causal_results_{group}.pkl"),
                    verbose=True
                )
                
                self.causal_results[group] = results
                self.effects_df[group] = effects
                
                # 効果をCSV保存
                output_path = self.output_dir / f"kpi_effects_{group}.csv"
                effects.to_csv(output_path, index=False)
                print(f"  [保存] {output_path}")
        
        else:
            # 従来の方法（全体で実行）
            print(f"\n[全体分析モード]")
            
            self.causal_results, self.effects_df = run_causal_discovery(
                features_df=self.df_scaled,
                feature_cols=feature_names,
                kpi_col=kpi_col,
                algorithm=self.config['lingam']['algorithm'],
                random_state=self.config['lingam']['random_state'],
                output_path=str(self.output_dir / "causal_results.pkl"),
                verbose=True
            )
            
            # 効果をCSV保存
            output_path = self.output_dir / "kpi_effects.csv"
            self.effects_df.to_csv(output_path, index=False)
            print(f"\n[保存] {output_path}")
    
    def run_causal_discovery_nonlinear(self):
        """
        ステップ7-9 (Phase 3.3b): 非線形因果探索（v0.2.0: Positive群重点）
        """
        print("\n" + "=" * 80)
        print("ステップ7-9: 非線形因果探索（v0.2.0: NonlinearLiNGAM）")
        print("=" * 80)
        
        # 非線形設定の確認
        nonlinear_config = self.config.get('nonlinear_causal', {})
        if not nonlinear_config.get('enabled', False):
            print("\n⚠ 非線形因果探索が無効です（config.yaml: nonlinear_causal.enabled=false）")
            return
        
        if self.df_scaled is None:
            raise ValueError("スケーリングが実行されていません。")
        
        feature_names = FeatureEngineer.get_feature_names()
        kpi_col = self.config['kpi']['primary_label']
        
        method = nonlinear_config.get('method', 'NonlinearLiNGAM')
        subset_size = nonlinear_config.get('subset_size', 5000)
        target_groups = nonlinear_config.get('target_groups', ['positive'])
        
        print(f"\n[非線形因果探索設定]")
        print(f"  手法: {method}")
        print(f"  サブセットサイズ: {subset_size}")
        print(f"  対象グループ: {target_groups}")
        
        # グループ別分析の確認
        u_group_analysis = self.config.get('kpi', {}).get('u_group_analysis', {})
        enabled = u_group_analysis.get('enabled', False)
        
        if enabled and 'u_group' in self.df_scaled.columns:
            # グループ別に非線形因果探索を実行
            self.nonlinear_results = {}
            self.nonlinear_effects = {}
            
            for group in target_groups:
                df_group = self.df_scaled[self.df_scaled['u_group'] == group].copy()
                
                if len(df_group) < 30:
                    print(f"\n⚠ 警告: {group} のデータ点数が少なすぎます（{len(df_group)} < 30）。スキップします。")
                    continue
                
                print(f"\n--- グループ: {group} ---")
                print(f"  レコード数: {len(df_group)}")
                
                # 非線形因果探索の実行
                results, effects = run_nonlinear_causal_discovery(
                    features_df=df_group,
                    feature_cols=feature_names,
                    kpi_col=kpi_col,
                    method=method,
                    subset_size=subset_size,
                    random_state=nonlinear_config.get('random_state', 42),
                    output_path=str(self.output_dir / f"causal_results_{group}_nonlinear.pkl"),
                    verbose=True
                )
                
                self.nonlinear_results[group] = results
                self.nonlinear_effects[group] = effects
                
                # 効果をCSV保存
                output_path = self.output_dir / f"kpi_effects_{group}_nonlinear.csv"
                effects.to_csv(output_path, index=False)
                print(f"  [保存] {output_path}")
                
                # 線形版との比較（線形版が既に実行されている場合）
                if hasattr(self, 'effects_df') and self.effects_df is not None and isinstance(self.effects_df, dict) and group in self.effects_df:
                    linear_effects = self.effects_df[group]
                    print(f"\n  [線形 vs 非線形 比較]")
                    print(f"    線形版の有意効果数: {len(linear_effects)}")
                    print(f"    非線形版の有意効果数: {len(effects)}")
                    
                    if len(effects) > 0:
                        print(f"    非線形版の最大効果: {effects['effect'].abs().max():.6f}")
                else:
                    print(f"\n  ℹ 線形版の結果がありません。先に --step causal_discovery を実行すると比較できます。")
        else:
            # 全体で実行
            print(f"\n[全体分析モード]")
            
            results, effects = run_nonlinear_causal_discovery(
                features_df=self.df_scaled,
                feature_cols=feature_names,
                kpi_col=kpi_col,
                method=method,
                subset_size=subset_size,
                random_state=nonlinear_config.get('random_state', 42),
                output_path=str(self.output_dir / "causal_results_nonlinear.pkl"),
                verbose=True
            )
            
            self.nonlinear_results = results
            self.nonlinear_effects = effects
            
            # 効果をCSV保存
            output_path = self.output_dir / "kpi_effects_nonlinear.csv"
            effects.to_csv(output_path, index=False)
            print(f"\n[保存] {output_path}")
    
    def run_causal_discovery_bayesian(self):
        """
        ステップ7-10 (Phase 3.3c): ベイズ因果探索（v0.2.0: Student-t回帰）
        """
        print("\n" + "=" * 80)
        print("ステップ7-10: ベイズ因果探索（v0.2.0: PyMC + NUTS）")
        print("=" * 80)
        
        # ベイズ設定の確認
        bayesian_config = self.config.get('advanced_methods', {}).get('bayesian', {})
        
        if self.df_scaled is None:
            raise ValueError("スケーリングが実行されていません。")
        
        feature_names = FeatureEngineer.get_feature_names()
        kpi_col = self.config['kpi']['primary_label']
        
        n_draws = bayesian_config.get('n_draws', 1000)
        n_tune = bayesian_config.get('n_tune', 500)
        n_chains = bayesian_config.get('n_chains', 4)
        target_accept = bayesian_config.get('target_accept', 0.90)
        use_student_t = bayesian_config.get('use_student_t', True)
        
        print(f"\n[ベイズ因果探索設定]")
        print(f"  NUTS: {n_draws}ドロー × {n_chains}チェイン")
        print(f"  チューニング: {n_tune}イテレーション")
        print(f"  モデル: {'Student-t回帰' if use_student_t else 'ガウス回帰'}")
        
        # グループ別分析の確認
        u_group_analysis = self.config.get('kpi', {}).get('u_group_analysis', {})
        enabled = u_group_analysis.get('enabled', False)
        groups = u_group_analysis.get('groups', ['positive', 'negative'])
        
        if enabled and 'u_group' in self.df_scaled.columns:
            # グループ別にベイズ因果探索を実行
            self.bayesian_results = {}
            self.bayesian_effects = {}
            
            for group in groups:
                df_group = self.df_scaled[self.df_scaled['u_group'] == group].copy()
                
                if len(df_group) < 100:
                    print(f"\n⚠ 警告: {group} のデータ点数が少なすぎます（{len(df_group)} < 100）。スキップします。")
                    continue
                
                print(f"\n--- グループ: {group} ---")
                print(f"  レコード数: {len(df_group)}")
                
                # ベイズ因果探索の実行
                results, effects = run_bayesian_causal_discovery(
                    features_df=df_group,
                    feature_cols=feature_names,
                    kpi_col=kpi_col,
                    n_draws=n_draws,
                    n_tune=n_tune,
                    n_chains=n_chains,
                    target_accept=target_accept,
                    use_student_t=use_student_t,
                    random_seed=self.config['lingam']['random_state'],
                    output_path=str(self.output_dir / f"causal_results_{group}_bayesian.pkl"),
                    verbose=True
                )
                
                self.bayesian_results[group] = results
                self.bayesian_effects[group] = effects
                
                # 効果をCSV保存
                output_path = self.output_dir / f"kpi_effects_{group}_bayesian.csv"
                effects.to_csv(output_path, index=False)
                print(f"  [保存] {output_path}")
                
                # 線形版との比較
                if hasattr(self, 'effects_df') and self.effects_df is not None and isinstance(self.effects_df, dict) and group in self.effects_df:
                    linear_effects = self.effects_df[group]
                    print(f"\n  [線形 vs ベイズ 比較]")
                    print(f"    線形版の有意効果数: {len(linear_effects)}")
                    print(f"    ベイズ版の有意効果数: {effects['significant'].sum()}")
                    
                    if len(effects) > 0:
                        print(f"    ベイズ版の最大効果: {effects['effect'].abs().max():.6f}")
                else:
                    print(f"\n  ℹ 線形版の結果がありません。")
        else:
            # 全体で実行
            print(f"\n[全体分析モード]")
            
            results, effects = run_bayesian_causal_discovery(
                features_df=self.df_scaled,
                feature_cols=feature_names,
                kpi_col=kpi_col,
                n_draws=n_draws,
                n_tune=n_tune,
                n_chains=n_chains,
                target_accept=target_accept,
                use_student_t=use_student_t,
                random_seed=self.config['lingam']['random_state'],
                output_path=str(self.output_dir / "causal_results_bayesian.pkl"),
                verbose=True
            )
            
            self.bayesian_results = results
            self.bayesian_effects = effects
            
            # 効果をCSV保存
            output_path = self.output_dir / "kpi_effects_bayesian.csv"
            effects.to_csv(output_path, index=False)
            print(f"\n[保存] {output_path}")
    
    def run_group_comparison(self):
        """
        Phase 3.4: グループ間因果構造の比較
        """
        print("\n" + "=" * 80)
        print("Phase 3.4: グループ間因果構造の比較")
        print("=" * 80)
        
        # グループ別結果が存在するか確認
        if not isinstance(self.causal_results, dict) or not isinstance(self.effects_df, dict):
            print("  ⚠ グループ別分析が実行されていません。スキップします。")
            return
        
        from src.validation import compare_causal_structures
        
        feature_names = FeatureEngineer.get_feature_names()
        
        # グループ間比較の実行
        comparison_df, differential_df = compare_causal_structures(
            results_dict=self.causal_results,
            effects_dict=self.effects_df,
            feature_names=feature_names,
            output_dir=str(self.output_dir)
        )
        
        # 結果を保持
        self.group_comparison = comparison_df
        self.differential_effects = differential_df
    
    def run_visualization(self):
        """
        ステップ9-10 (Phase 4): 可視化と解釈（グループ別対応）
        """
        print("\n" + "=" * 80)
        print("ステップ9-10: 可視化と解釈")
        print("=" * 80)
        
        if self.causal_results is None or self.effects_df is None:
            raise ValueError("因果探索が実行されていません。")
        
        kpi_col = self.config['kpi']['primary_label']
        feature_names = FeatureEngineer.get_feature_names()
        
        # 可視化オブジェクトの初期化
        visualizer = CausalVisualizer(
            figure_size=tuple(self.config['visualization']['figure_size']),
            dpi=self.config['visualization']['dpi'],
            edge_width_scale=self.config['visualization']['edge_width_scale'],
            node_size_scale=self.config['visualization']['node_size_scale'],
            layout=self.config['visualization']['layout']
        )
        
        # グループ別分析の確認
        is_group_analysis = isinstance(self.causal_results, dict)
        
        if is_group_analysis:
            print("\n[グループ別可視化モード]")
            
            # v0.1.2: グループ別因果グラフ
            from src.visualization import (
                plot_ui_distribution,
                plot_group_comparison_graphs,
                plot_differential_effects,
                plot_divergent_effects_heatmap
            )
            
            groups = list(self.causal_results.keys())
            
            # 1. u_i分布のヒストグラム
            if self.ui_heterogeneity is not None:
                plot_ui_distribution(
                    heterogeneity_df=self.ui_heterogeneity,
                    groups=groups,
                    output_path=str(self.output_dir / "ui_distribution_comparison.png"),
                    dpi=self.config['visualization']['dpi']
                )
            
            # 2. グループ別因果グラフ
            plot_group_comparison_graphs(
                results_dict=self.causal_results,
                feature_names=feature_names,
                kpi_name=kpi_col,
                threshold=0.1,
                top_k=10,
                output_path=str(self.output_dir / "causal_graph_groups.png"),
                dpi=self.config['visualization']['dpi']
            )
            
            # 3. 因果効果の比較図
            plot_differential_effects(
                effects_dict=self.effects_df,
                feature_names=feature_names,
                groups=groups,
                output_path=str(self.output_dir / "effect_comparison_lineplot.png"),
                dpi=self.config['visualization']['dpi']
            )
            
            # 4. ダイバージェント・エフェクト・ヒートマップ
            if hasattr(self, 'differential_effects') and self.differential_effects is not None:
                plot_divergent_effects_heatmap(
                    differential_df=self.differential_effects,
                    groups=groups,
                    output_path=str(self.output_dir / "divergent_effects_heatmap.png"),
                    dpi=self.config['visualization']['dpi']
                )
            
            # 各グループの個別可視化
            for group in groups:
                print(f"\n  --- {group} 個別可視化 ---")
                
                # 因果グラフ
                visualizer.visualize_causal_graph(
                    adjacency_matrix=self.causal_results[group]['adjacency_matrix'],
                    feature_names=self.causal_results[group]['feature_names'],
                    kpi_name=kpi_col,
                    threshold=0.1,
                    top_k=15,
                    output_path=str(self.output_dir / f"causal_graph_{group}.png")
                )
                
                # 効果ヒートマップ
                visualizer.visualize_effect_heatmap(
                    adjacency_matrix=self.causal_results[group]['adjacency_matrix'],
                    feature_names=self.causal_results[group]['feature_names'],
                    kpi_name=kpi_col,
                    top_k=20,
                    output_path=str(self.output_dir / f"effect_heatmap_{group}.png")
                )
                
                # 解釈レポート
                report = generate_interpretation_report(
                    effects_df=self.effects_df[group],
                    output_path=str(self.output_dir / f"interpretation_report_{group}.md")
                )
        
        else:
            # 従来の方法（全体）
            print("\n[全体可視化モード]")
            
            # 因果グラフの可視化
            visualizer.visualize_causal_graph(
                adjacency_matrix=self.causal_results['adjacency_matrix'],
                feature_names=self.causal_results['feature_names'],
                kpi_name=kpi_col,
                threshold=0.1,
                top_k=15,
                output_path=str(self.output_dir / "causal_graph.png")
            )
            
            # 効果ヒートマップの可視化
            visualizer.visualize_effect_heatmap(
                adjacency_matrix=self.causal_results['adjacency_matrix'],
                feature_names=self.causal_results['feature_names'],
                kpi_name=kpi_col,
                top_k=20,
                output_path=str(self.output_dir / "effect_heatmap.png")
            )
            
            # 解釈レポートの生成
            report = generate_interpretation_report(
                effects_df=self.effects_df,
                output_path=str(self.output_dir / "interpretation_report.md")
            )
    
    def run_bootstrap_validation(self):
        """
        ステップ11: ブートストラップ検証
        """
        print("\n" + "=" * 80)
        print("ステップ11: ブートストラップ検証")
        print("=" * 80)
        
        if self.df_scaled is None or self.causal_results is None:
            raise ValueError("因果探索が実行されていません。")
        
        feature_names = FeatureEngineer.get_feature_names()
        kpi_col = self.config['kpi']['primary_label']
        
        # データ行列を準備
        all_cols = feature_names + [kpi_col]
        X = self.df_scaled[all_cols].dropna().values
        variable_names = all_cols
        
        # ブートストラップ検証
        stable_edges_df, edge_frequencies = run_bootstrap_validation(
            X=X,
            variable_names=variable_names,
            kpi_name=kpi_col,
            algorithm=self.config['lingam']['algorithm'],
            n_sampling=self.config['bootstrap']['n_sampling'],
            sample_ratio=self.config['bootstrap']['sample_ratio'],
            stability_threshold=self.config['bootstrap']['stability_threshold'],
            random_state=self.config['lingam']['random_state'],
            n_jobs=self.config['misc']['n_jobs'],
            output_dir=str(self.output_dir),
            verbose=True
        )
    
    def run_all(self, version='v0.1.2'):
        """
        全パイプラインを実行
        
        Args:
            version: 'v0.1.0' (従来の故障予測KPI) or 'v0.1.2' (劣化ハザードu_i)
        """
        if version == 'v0.1.2':
            # v0.1.2: ハザードモデル + 因果探索
            print("\n" + "=" * 80)
            print("実行モード: v0.1.2 (劣化ハザードランダム効果をKPIとして使用)")
            print("=" * 80)
            
            self.run_preprocessing()
            self.run_feature_engineering()
            self.run_hazard_preprocessing()
            self.run_hazard_estimation()
            self.run_ui_extraction_and_grouping()
            self.run_feature_ui_merge()
            self.run_scaling_and_correlation_check()
            self.run_causal_discovery()
            self.run_group_comparison()  # Phase 3.4
            self.run_visualization()
            self.run_bootstrap_validation()
        
        else:
            # v0.1.0: 従来の方法（故障予測KPI）
            print("\n" + "=" * 80)
            print("実行モード: v0.1.0 (従来の故障予測KPIを使用)")
            print("=" * 80)
            
            self.run_preprocessing()
            self.run_feature_engineering()
            self.run_scaling_and_correlation_check()
            self.run_causal_discovery()
            self.run_visualization()
            self.run_bootstrap_validation()
        
        print("\n" + "=" * 80)
        print("全パイプライン完了")
        print("=" * 80)
        print(f"\n出力ディレクトリ: {self.output_dir}")
        print("\n生成されたファイル:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(
        description="ポンプ設備LiNGAM因果探索MVPパイプライン v0.1.2"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='設定ファイルのパス（デフォルト: config.yaml）'
    )
    parser.add_argument(
        '--version',
        type=str,
        choices=['v0.1.0', 'v0.1.2'],
        default='v0.1.2',
        help='実行バージョン（デフォルト: v0.1.2）'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['preprocessing', 'feature_engineering', 'scaling', 
                 'causal_discovery', 'causal_discovery_nonlinear', 'causal_discovery_bayesian',
                 'visualization', 'validation',
                 'hazard_preprocessing', 'hazard_estimation', 'ui_extraction', 'feature_ui_merge',
                 'all'],
        default='all',
        help='実行するステップ（デフォルト: all）'
    )
    parser.add_argument(
        '--hazard-only',
        action='store_true',
        help='ハザードモデルのみ実行（u_i推定まで）'
    )
    parser.add_argument(
        '--causal-only',
        action='store_true',
        help='因果探索のみ実行（u_i既存を前提）'
    )
    
    args = parser.parse_args()
    
    # パイプラインの初期化
    pipeline = PumpCausalPipeline(config_path=args.config)
    
    # 特殊モード
    if args.hazard_only:
        print("\n[モード] ハザードモデルのみ実行")
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        pipeline.run_hazard_preprocessing()
        pipeline.run_hazard_estimation()
        pipeline.run_ui_extraction_and_grouping()
        return
    
    if args.causal_only:
        print("\n[モード] 因果探索のみ実行（u_i既存を前提）")
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        # u_iファイルが存在することを前提
        ui_path = pipeline.output_dir / "pump_heterogeneity.csv"
        if not ui_path.exists():
            raise FileNotFoundError(f"u_iファイルが見つかりません: {ui_path}")
        pipeline.ui_heterogeneity = pd.read_csv(ui_path)
        pipeline.run_feature_ui_merge()
        pipeline.run_scaling_and_correlation_check()
        pipeline.run_causal_discovery()
        pipeline.run_visualization()
        pipeline.run_bootstrap_validation()
        return
    
    # ステップの実行
    if args.step == 'all':
        pipeline.run_all(version=args.version)
    elif args.step == 'preprocessing':
        pipeline.run_preprocessing()
    elif args.step == 'feature_engineering':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
    elif args.step == 'hazard_preprocessing':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        pipeline.run_hazard_preprocessing()
    elif args.step == 'hazard_estimation':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        pipeline.run_hazard_preprocessing()
        pipeline.run_hazard_estimation()
    elif args.step == 'ui_extraction':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        pipeline.run_hazard_preprocessing()
        pipeline.run_hazard_estimation()
        pipeline.run_ui_extraction_and_grouping()
    elif args.step == 'feature_ui_merge':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        # u_iファイル読込
        ui_path = pipeline.output_dir / "pump_heterogeneity.csv"
        pipeline.ui_heterogeneity = pd.read_csv(ui_path)
        pipeline.run_feature_ui_merge()
    elif args.step == 'scaling':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        if args.version == 'v0.1.2':
            ui_path = pipeline.output_dir / "pump_heterogeneity.csv"
            pipeline.ui_heterogeneity = pd.read_csv(ui_path)
            pipeline.run_feature_ui_merge()
        pipeline.run_scaling_and_correlation_check()
    elif args.step == 'causal_discovery':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        if args.version == 'v0.1.2':
            ui_path = pipeline.output_dir / "pump_heterogeneity.csv"
            pipeline.ui_heterogeneity = pd.read_csv(ui_path)
            pipeline.run_feature_ui_merge()
        pipeline.run_scaling_and_correlation_check()
        pipeline.run_causal_discovery()
    elif args.step == 'causal_discovery_nonlinear':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        if args.version == 'v0.1.2':
            ui_path = pipeline.output_dir / "pump_heterogeneity.csv"
            pipeline.ui_heterogeneity = pd.read_csv(ui_path)
            pipeline.run_feature_ui_merge()
        pipeline.run_scaling_and_correlation_check()
        pipeline.run_causal_discovery_nonlinear()
    elif args.step == 'causal_discovery_bayesian':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        if args.version == 'v0.1.2':
            ui_path = pipeline.output_dir / "pump_heterogeneity.csv"
            pipeline.ui_heterogeneity = pd.read_csv(ui_path)
            pipeline.run_feature_ui_merge()
        pipeline.run_scaling_and_correlation_check()
        pipeline.run_causal_discovery_bayesian()
    elif args.step == 'visualization':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        if args.version == 'v0.1.2':
            ui_path = pipeline.output_dir / "pump_heterogeneity.csv"
            pipeline.ui_heterogeneity = pd.read_csv(ui_path)
            pipeline.run_feature_ui_merge()
        pipeline.run_scaling_and_correlation_check()
        pipeline.run_causal_discovery()
        pipeline.run_visualization()
    elif args.step == 'validation':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        if args.version == 'v0.1.2':
            ui_path = pipeline.output_dir / "pump_heterogeneity.csv"
            pipeline.ui_heterogeneity = pd.read_csv(ui_path)
            pipeline.run_feature_ui_merge()
        pipeline.run_scaling_and_correlation_check()
        pipeline.run_causal_discovery()
        pipeline.run_bootstrap_validation()


if __name__ == "__main__":
    main()
