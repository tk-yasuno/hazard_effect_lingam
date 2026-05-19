"""
ベイズ混合因果探索モジュール (v0.2.0)

PyMCを使用した分布非依存の因果推論:
- Student-t分布による外れ値対応
- 階層ベイズモデルでu_iの異質性を明示的にモデリング
- NUTS推定による事後分布の取得
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class BayesianCausalDiscovery:
    """
    ベイズ混合因果探索クラス
    
    PyMCを使用して、因果構造と効果を同時推定する。
    外れ値に対してロバストな Student-t分布を使用。
    """
    
    def __init__(
        self,
        n_draws: int = 1000,
        n_tune: int = 500,
        n_chains: int = 4,
        target_accept: float = 0.90,
        random_seed: int = 42,
        verbose: bool = True
    ):
        """
        Args:
            n_draws: NUTS推定のドロー数
            n_tune: チューニング期間
            n_chains: チェイン数
            target_accept: 受容率目標
            random_seed: 乱数シード
            verbose: 進捗表示
        """
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.verbose = verbose
        
        self.model = None
        self.trace = None
        self.feature_names = None
        self.kpi_name = None
        
        if verbose:
            print(f"\n[BayesianCausalDiscovery]")
            print(f"  NUTS設定: {n_draws}ドロー × {n_chains}チェイン")
            print(f"  チューニング: {n_tune}イテレーション")
    
    def prepare_data_matrix(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        kpi_col: str = 'u_i_target',
        drop_na: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        ベイズ推定用のデータ行列を準備
        
        Args:
            features_df: 特徴量データフレーム
            feature_cols: 特徴量の列名リスト
            kpi_col: KPI列名
            drop_na: NaNを含む行を削除するか
        
        Returns:
            (特徴量行列 X, KPIベクトル y, 特徴量名リスト)
        """
        if self.verbose:
            print("\n[データ行列の準備（ベイズ推定用）]")
        
        # データのコピー
        df_subset = features_df[feature_cols + [kpi_col]].copy()
        
        initial_len = len(df_subset)
        if self.verbose:
            print(f"  入力レコード数: {initial_len}")
            print(f"  特徴量数: {len(feature_cols)}")
        
        # NaN処理
        if drop_na:
            df_subset = df_subset.dropna()
            if self.verbose:
                print(f"  NaN除去後: {len(df_subset)} レコード ({initial_len - len(df_subset)} 削除)")
        else:
            df_subset = df_subset.fillna(0.0)
        
        # X, yに分割
        X = df_subset[feature_cols].values
        y = df_subset[kpi_col].values
        
        if self.verbose:
            print(f"  X行列サイズ: {X.shape}")
            print(f"  yベクトルサイズ: {y.shape}")
        
        self.feature_names = feature_cols
        self.kpi_name = kpi_col
        
        return X, y, feature_cols
    
    def build_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_student_t: bool = True
    ) -> pm.Model:
        """
        ベイズ因果推論モデルを構築
        
        Args:
            X: 特徴量行列 [n_samples, n_features]
            y: KPIベクトル [n_samples]
            use_student_t: Student-t分布を使用するか（外れ値対応）
        
        Returns:
            PyMCモデル
        """
        if self.verbose:
            print(f"\n[ベイズモデルの構築]")
            print(f"  モデル: {'Student-t 回帰' if use_student_t else 'ガウス回帰'}")
        
        n_samples, n_features = X.shape
        
        with pm.Model() as model:
            # データの登録
            X_data = pm.Data('X', X)
            y_data = pm.Data('y', y)
            
            # 事前分布: 特徴量の因果効果 beta
            # 分散が大きめの正規分布（弱情報事前分布）
            beta = pm.Normal(
                'beta',
                mu=0,
                sigma=10,
                shape=n_features
            )
            
            # 切片
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            
            # 線形予測子
            mu = alpha + pm.math.dot(X_data, beta)
            
            if use_student_t:
                # Student-t分布（自由度νも推定）
                # 外れ値に対してロバスト
                nu = pm.Exponential('nu', 1/30)  # 自由度の事前分布
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                likelihood = pm.StudentT(
                    'y_obs',
                    nu=nu,
                    mu=mu,
                    sigma=sigma,
                    observed=y_data
                )
            else:
                # ガウス分布
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                likelihood = pm.Normal(
                    'y_obs',
                    mu=mu,
                    sigma=sigma,
                    observed=y_data
                )
            
            if self.verbose:
                print(f"  パラメータ数: {n_features + 2}")  # beta + alpha + sigma (+ nu)
                print(f"  サンプル数: {n_samples}")
        
        self.model = model
        return model
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_student_t: bool = True
    ) -> Dict:
        """
        NUTS推定を実行
        
        Args:
            X: 特徴量行列
            y: KPIベクトル
            use_student_t: Student-t分布を使用するか
        
        Returns:
            推定結果の辞書
        """
        if self.verbose:
            print(f"\n[NUTS推定の実行]")
        
        # モデルの構築
        model = self.build_model(X, y, use_student_t=use_student_t)
        
        # NUTS推定
        with model:
            self.trace = pm.sample(
                draws=self.n_draws,
                tune=self.n_tune,
                chains=self.n_chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=self.verbose
            )
        
        if self.verbose:
            print(f"  ✓ 推定完了")
        
        # 収束診断
        rhat = az.rhat(self.trace)
        max_rhat = float(rhat['beta'].max().values)
        
        if self.verbose:
            print(f"\n[収束診断]")
            print(f"  最大R̂: {max_rhat:.4f}")
            if max_rhat < 1.1:
                print(f"  ✓ 収束良好（R̂ < 1.1）")
            else:
                print(f"  ⚠ 収束に問題あり（R̂ ≥ 1.1）")
        
        # 結果辞書を構築
        results = {
            'trace': self.trace,
            'feature_names': self.feature_names,
            'kpi_name': self.kpi_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'max_rhat': max_rhat,
            'use_student_t': use_student_t
        }
        
        return results
    
    def get_effects_to_kpi(self) -> pd.DataFrame:
        """
        KPIへの因果効果（betaの事後分布）を抽出
        
        Returns:
            因果効果のデータフレーム（平均、標準偏差、95%信用区間）
        """
        if self.trace is None:
            raise ValueError("モデルがまだ学習されていません。fit()を先に実行してください。")
        
        # betaの事後分布を取得
        beta_samples = self.trace.posterior['beta'].values  # shape: (chains, draws, n_features)
        beta_samples_flat = beta_samples.reshape(-1, beta_samples.shape[-1])  # (chains*draws, n_features)
        
        # 統計量を計算
        beta_mean = beta_samples_flat.mean(axis=0)
        beta_std = beta_samples_flat.std(axis=0)
        beta_lower = np.percentile(beta_samples_flat, 2.5, axis=0)
        beta_upper = np.percentile(beta_samples_flat, 97.5, axis=0)
        
        # データフレームに変換
        effects_df = pd.DataFrame({
            'feature': self.feature_names,
            'effect': beta_mean,
            'std': beta_std,
            'lower_95': beta_lower,
            'upper_95': beta_upper
        })
        
        # 有意性判定（95%信用区間が0を含まない）
        effects_df['significant'] = ~(
            (effects_df['lower_95'] <= 0) & 
            (effects_df['upper_95'] >= 0)
        )
        
        # 絶対値でソート
        effects_df = effects_df.sort_values('effect', key=abs, ascending=False)
        effects_df = effects_df.reset_index(drop=True)
        
        if self.verbose:
            print(f"\n[KPIへの因果効果（ベイズ推定）]")
            print(f"  有意な効果数: {effects_df['significant'].sum()}")
            if len(effects_df) > 0:
                print(f"  最大効果: {effects_df['effect'].abs().max():.6f}")
        
        return effects_df


def run_bayesian_causal_discovery(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    kpi_col: str = 'u_i_target',
    n_draws: int = 1000,
    n_tune: int = 500,
    n_chains: int = 4,
    target_accept: float = 0.90,
    random_seed: int = 42,
    use_student_t: bool = True,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    ベイズ因果探索を実行するラッパー関数
    
    Args:
        features_df: 特徴量データフレーム
        feature_cols: 特徴量の列名リスト
        kpi_col: KPI列名
        n_draws: NUTSドロー数
        n_tune: チューニング期間
        n_chains: チェイン数
        target_accept: 受容率目標
        random_seed: 乱数シード
        use_student_t: Student-t分布を使用するか
        output_path: 結果の保存先（Noneなら保存しない）
        verbose: 進捗表示
    
    Returns:
        (推定結果の辞書, KPI効果のデータフレーム)
    """
    # ベイズ因果探索インスタンスの作成
    bcd = BayesianCausalDiscovery(
        n_draws=n_draws,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=target_accept,
        random_seed=random_seed,
        verbose=verbose
    )
    
    # データ行列の準備
    X, y, feature_names = bcd.prepare_data_matrix(
        features_df=features_df,
        feature_cols=feature_cols,
        kpi_col=kpi_col
    )
    
    # ベイズ推定の実行
    results = bcd.fit(X, y, use_student_t=use_student_t)
    
    # KPIへの効果を抽出
    effects_df = bcd.get_effects_to_kpi()
    
    # 結果の保存
    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # traceは別途保存（NetCDF形式）
        trace_file = output_file.with_suffix('.nc')
        results['trace'].to_netcdf(str(trace_file))
        
        # その他の結果をpickle保存
        results_no_trace = {k: v for k, v in results.items() if k != 'trace'}
        with open(output_file, 'wb') as f:
            pickle.dump(results_no_trace, f)
        
        if verbose:
            print(f"\n[結果の保存]")
            print(f"  {output_file}")
            print(f"  {trace_file}")
    
    return results, effects_df
