"""
GPU動作確認スクリプト (v0.2.0)

JAXとNumPyroのGPU動作を確認します。
"""

def check_jax_gpu():
    """JAXのGPU動作確認"""
    print("=" * 80)
    print("JAX GPU動作確認")
    print("=" * 80)
    
    try:
        import jax
        import jax.numpy as jnp
        
        print("\n[JAX バージョン]")
        print(f"  JAX: {jax.__version__}")
        
        print("\n[利用可能なデバイス]")
        devices = jax.devices()
        for i, device in enumerate(devices):
            print(f"  {i}: {device}")
        
        print("\n[デフォルトバックエンド]")
        default_backend = devices[0].platform
        print(f"  {default_backend}")
        
        if default_backend == 'gpu':
            print("\n  ✅ GPU が利用可能です！")
        elif default_backend == 'cpu':
            print("\n  ⚠️  CPUモードで動作しています")
            print("     GPUを使用するには:")
            print("     1. CUDAドライバーがインストールされているか確認: nvidia-smi")
            print("     2. JAX CUDA版がインストールされているか確認")
            print("        pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        elif default_backend == 'tpu':
            print("\n  🚀 TPU が利用可能です！")
        
        # 簡単な計算テスト
        print("\n[計算テスト]")
        x = jnp.ones((1000, 1000))
        y = jnp.dot(x, x)
        print(f"  行列積計算: {x.shape} @ {x.shape} = {y.shape}")
        print(f"  結果サンプル: {y[0, 0]}")
        
        return default_backend == 'gpu' or default_backend == 'tpu'
        
    except ImportError as e:
        print(f"\n  ❌ JAXがインストールされていません: {e}")
        print("     インストール:")
        print("     pip install jax jaxlib")
        return False
    except Exception as e:
        print(f"\n  ❌ エラー: {e}")
        return False


def check_numpyro():
    """NumPyroのインストール確認"""
    print("\n" + "=" * 80)
    print("NumPyro インストール確認")
    print("=" * 80)
    
    try:
        import numpyro
        print(f"\n  ✅ NumPyro {numpyro.__version__} がインストールされています")
        return True
    except ImportError:
        print("\n  ❌ NumPyroがインストールされていません")
        print("     インストール:")
        print("     pip install numpyro>=0.13.0")
        return False


def check_pymc_numpyro():
    """PyMCのNumPyroサンプラー対応確認"""
    print("\n" + "=" * 80)
    print("PyMC NumPyroサンプラー確認")
    print("=" * 80)
    
    try:
        import pymc as pm
        print(f"\n  PyMC バージョン: {pm.__version__}")
        
        # 簡単なモデルでNumPyroサンプラーをテスト
        print("\n  簡単なモデルでNumPyroサンプラーをテスト中...")
        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0, sigma=1)
            obs = pm.Normal('obs', mu=mu, sigma=1, observed=[0.5, 1.0, 1.5])
        
        with model:
            trace = pm.sample(
                draws=100,
                tune=100,
                chains=1,
                nuts_sampler='numpyro',
                progressbar=False,
                random_seed=42
            )
        
        print(f"  ✅ NumPyroサンプラーが正常に動作しました")
        print(f"     サンプル数: {len(trace.posterior['mu'].values.flatten())}")
        print(f"     推定平均: {trace.posterior['mu'].mean().item():.3f}")
        return True
        
    except ImportError as e:
        print(f"\n  ❌ PyMCがインストールされていません: {e}")
        print("     インストール:")
        print("     pip install pymc>=5.0.0")
        return False
    except Exception as e:
        print(f"\n  ❌ エラー: {e}")
        return False


def main():
    """メイン関数"""
    print("\n" + "=" * 80)
    print("GPU加速環境確認スクリプト (v0.2.0)")
    print("=" * 80)
    
    # JAXのGPU確認
    jax_gpu_ok = check_jax_gpu()
    
    # NumPyroの確認
    numpyro_ok = check_numpyro()
    
    # PyMCのNumPyroサンプラー確認
    pymc_numpyro_ok = check_pymc_numpyro()
    
    # 総合判定
    print("\n" + "=" * 80)
    print("総合判定")
    print("=" * 80)
    
    if jax_gpu_ok and numpyro_ok and pymc_numpyro_ok:
        print("\n  🎉 GPU加速の準備が完了しました！")
        print("\n  config.yamlで以下の設定を有効化してください:")
        print("  ```yaml")
        print("  hazard_model:")
        print("    use_gpu: true")
        print("    jax_platform: 'gpu'")
        print("  ```")
    elif not jax_gpu_ok:
        print("\n  ⚠️  JAX GPUサポートが利用できません")
        print("     CPU推定を使用するか、GPU環境をセットアップしてください")
        print("\n     セットアップ手順:")
        print("     1. NVIDIA GPUとCUDAドライバーを確認: nvidia-smi")
        print("     2. JAX CUDA版をインストール:")
        print("        pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    elif not numpyro_ok:
        print("\n  ⚠️  NumPyroがインストールされていません")
        print("     pip install numpyro>=0.13.0")
    elif not pymc_numpyro_ok:
        print("\n  ⚠️  PyMC NumPyroサンプラーが動作しません")
        print("     PyMCとNumPyroのバージョンを確認してください")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
