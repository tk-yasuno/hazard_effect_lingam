"""
事後分析: ランダム効果 u_i の抽出とグループ分け
"""

import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from .config_hazard import HazardConfig


def extract_random_effects(trace, model_data: dict, config: HazardConfig) -> pd.DataFrame:
    """
    各ポンプの u_i 事後分布から平均・95%信用区間を計算
    
    Parameters:
        trace: InferenceData from NUTS sampling
        model_data: dict with keys: N_pumps, pump_equipment_id, pump_check_item_id
        config: HazardConfig instance
    
    Returns:
        DataFrame with columns: pump_idx, equipment_id, check_item_id, 
                               u_mean, u_lower, u_upper, rank, significant
    """
    print("\n[ランダム効果の抽出]")
    
    N_pumps = int(model_data['N_pumps'])
    pump_equipment_id = model_data['pump_equipment_id']
    pump_check_item_id = model_data.get('pump_check_item_id', np.zeros(N_pumps))
    
    # u の事後分布から統計量を計算
    u_samples = trace.posterior['u'].values  # shape: (chains, draws, N_pumps)
    
    # チェインとドローをまとめる
    u_samples_flat = u_samples.reshape(-1, N_pumps)  # shape: (chains*draws, N_pumps)
    
    # 各ポンプの統計量
    u_mean = u_samples_flat.mean(axis=0)
    u_lower = np.percentile(u_samples_flat, 2.5, axis=0)
    u_upper = np.percentile(u_samples_flat, 97.5, axis=0)
    
    # DataFrame化
    heterogeneity_df = pd.DataFrame({
        'pump_idx': np.arange(N_pumps),
        'equipment_id': pump_equipment_id,
        'check_item_id': pump_check_item_id,
        'u_mean': u_mean,
        'u_lower': u_lower,
        'u_upper': u_upper
    })
    
    # ランキング（u_mean の降順）
    heterogeneity_df = heterogeneity_df.sort_values('u_mean', ascending=False)
    heterogeneity_df['rank'] = np.arange(1, N_pumps + 1)
    
    # 有意性判定（95%信用区間が0を含まない）
    heterogeneity_df['significant'] = ~(
        (heterogeneity_df['u_lower'] <= 0) & 
        (heterogeneity_df['u_upper'] >= 0)
    )
    
    # 統計サマリ
    n_significant = heterogeneity_df['significant'].sum()
    n_fast = (heterogeneity_df['u_lower'] > 0).sum()  # 有意に速い
    n_slow = (heterogeneity_df['u_upper'] < 0).sum()  # 有意に遅い
    
    print(f"  ポンプ数: {N_pumps}")
    print(f"  u_mean 範囲: {u_mean.min():.4f} ~ {u_mean.max():.4f}")
    print(f"  有意に異なるポンプ: {n_significant} ({n_significant/N_pumps*100:.1f}%)")
    print(f"    - 有意に劣化が速い: {n_fast}")
    print(f"    - 有意に劣化が遅い: {n_slow}")
    
    # 保存
    output_path = config.get_absolute_path(config.pump_heterogeneity)
    heterogeneity_df.to_csv(output_path, index=False)
    print(f"  保存完了: {output_path}")
    
    return heterogeneity_df


def assign_ui_groups(heterogeneity_df: pd.DataFrame, config: HazardConfig) -> pd.DataFrame:
    """
    u_i に基づいてポンプをグループ分け（符号ベース：positive/negative）
    
    Parameters:
        heterogeneity_df: extract_random_effects() の出力
        config: HazardConfig instance
    
    Returns:
        heterogeneity_df with additional column 'u_group'
    
    v0.2.0 Changes:
        - Binary grouping based on sign of u_i
        - positive: u_i > 0 (faster deterioration)
        - negative: u_i <= 0 (slower deterioration, includes u_i == 0)
    """
    print("\n[グループ分け（符号ベース）]")
    
    # 符号ベースのグループ割り当て
    def assign_group(u_val):
        if u_val > 0:
            return 'positive'
        else:
            return 'negative'  # u_i <= 0 (includes zero)
    
    heterogeneity_df['u_group'] = heterogeneity_df['u_mean'].apply(assign_group)
    
    # グループ統計
    group_counts = heterogeneity_df['u_group'].value_counts()
    print(f"\n  グループ別ポンプ数:")
    for group in ['positive', 'negative']:
        count = group_counts.get(group, 0)
        pct = count / len(heterogeneity_df) * 100
        u_range = heterogeneity_df[heterogeneity_df['u_group'] == group]['u_mean']
        print(f"    {group}: {count} ({pct:.1f}%) - u_i range: [{u_range.min():.4f}, {u_range.max():.4f}]")
    
    # ゼロ付近のポンプ数を確認
    near_zero = ((heterogeneity_df['u_mean'] >= -0.01) & (heterogeneity_df['u_mean'] <= 0.01)).sum()
    print(f"\n  ゼロ付近（-0.01 < u_i < 0.01）のポンプ数: {near_zero}")
    
    return heterogeneity_df
