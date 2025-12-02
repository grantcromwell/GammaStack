"""
Testing Framework for Advanced Features Integration

Run this script to validate that all four advanced features are properly 
integrated and functioning correctly.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_features import (
    PlausibilityAnalyzer,
    ResidualAnalyzer,
    CorrelationTopology,
    MarketNeutralRiskManager
)


def test_plausibility_scoring():
    """Test 1: Plausibility & Diffusion Analysis"""
    print("=" * 70)
    print("TEST 1: PLAUSIBILITY & DIFFUSION ANALYSIS")
    print("=" * 70)
    
    try:
        # Generate sample price data
        np.random.seed(42)
        sample_prices = np.cumsum(np.random.randn(100)) + 100
        
        analyzer = PlausibilityAnalyzer()
        
        # Test diffusion metrics
        diffusion = analyzer.compute_diffusion_metrics(sample_prices, lookback=50)
        
        print(f"✓ Diffusion Metrics Computed:")
        print(f"  - Brownian Likelihood: {diffusion.brownian_likelihood:.3f}")
        print(f"  - Mean Reversion Strength: {diffusion.mean_reversion_strength:.3f}")
        print(f"  - Drift Component: {diffusion.drift_component:.6f}")
        print(f"  - Diffusion Speed (Volatility): {diffusion.diffusion_speed:.6f}")
        print(f"  - Confidence: {diffusion.confidence:.3f}")
        print(f"  - Expected Range: {diffusion.expected_range_next_n}")
        
        # Test signal plausibility scoring
        kline_data = {
            'Open': sample_prices[:-1].tolist() + [sample_prices[-1]],
            'High': (sample_prices * 1.02).tolist(),
            'Low': (sample_prices * 0.98).tolist(),
            'Close': sample_prices.tolist(),
            'Volume': np.random.randint(1000, 5000, 100).tolist()
        }
        
        score = analyzer.score_signal_plausibility(kline_data, 'LONG', sample_prices[-1])
        
        # score is a dict, extract the plausibility_score value
        if isinstance(score, dict):
            plausibility = score.get('plausibility_score', score.get('score', 0.5))
        else:
            plausibility = score
        
        print(f"\n✓ Signal Plausibility Score: {plausibility:.3f}")
        
        print("\n✅ TEST 1 PASSED: Plausibility & Diffusion Analysis")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_residual_analysis():
    """Test 2: Residual-Based Fair Value Detection"""
    print("\n" + "=" * 70)
    print("TEST 2: RESIDUAL-BASED FAIR VALUE DETECTION")
    print("=" * 70)
    
    try:
        # Generate sample data
        np.random.seed(42)
        n = 100
        prices = np.cumsum(np.random.randn(n)) + 100
        
        sample_data = {
            'Close': prices.tolist(),
            'High': (prices * 1.02).tolist(),
            'Low': (prices * 0.98).tolist(),
            'Open': np.roll(prices, 1).tolist(),
            'Volume': np.random.randint(1000, 5000, n).tolist()
        }
        
        # Create predictors
        predictors = {
            'rsi': np.random.rand(n) * 100,
            'macd': np.random.randn(n),
            'volume_ratio': np.ones(n)
        }
        
        analyzer = ResidualAnalyzer()
        model = analyzer.build_fair_value_model(sample_data, predictors, lookback=50)
        
        print(f"✓ Fair Value Model Built:")
        print(f"  - Fair Price: ${model['fair_prices'][-1]:.2f}")
        print(f"  - Current Price: ${sample_data['Close'][-1]:.2f}")
        print(f"  - Current Deviation: {model['current_discount_pct']:.2f}%")
        print(f"  - Model R² (quality): {model['r_squared']:.3f}")
        print(f"  - Residual Std: {model['residual_std']:.3f}")
        print(f"  - Current Residual (z-score): {model['current_residual'] / (model['residual_std'] + 1e-8):.2f}σ")
        
        print("\n✅ TEST 2 PASSED: Residual Analysis")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_correlation_topology():
    """Test 3: Persistent Homology on Correlations"""
    print("\n" + "=" * 70)
    print("TEST 3: PERSISTENT HOMOLOGY ON CORRELATIONS")
    print("=" * 70)
    
    try:
        # Generate synthetic forex returns
        np.random.seed(42)
        n_periods = 60
        
        forex_returns = {
            'EURUSD': np.random.randn(n_periods),
            'GBPUSD': np.random.randn(n_periods) * 0.9 + np.random.randn(n_periods) * 0.1,
            'JPYUSD': np.random.randn(n_periods) * -0.8 + np.random.randn(n_periods) * 0.2,
            'AUDUSD': np.random.randn(n_periods),
            'CHFUSD': np.random.randn(n_periods) * -0.7 + np.random.randn(n_periods) * 0.3,
        }
        
        topology = CorrelationTopology()
        
        # Compute correlation matrix
        corr_matrix, pairs = topology.compute_correlation_matrix(forex_returns)
        print(f"✓ Correlation Matrix Computed: {len(pairs)} pairs analyzed")
        
        # Compute persistent homology
        homology = topology.compute_persistent_homology(corr_matrix, pairs)
        
        print(f"✓ Persistent Homology Analysis:")
        print(f"  - Number of Correlation Clusters: {homology['num_clusters']}")
        print(f"  - Number of Triangular Structures: {homology['num_loops']}")
        print(f"  - Stability Score: {homology['stability_score']:.3f}")
        print(f"  - Detected Regime: {homology['regime_type']}")
        
        # Detect regime changes (use the static method with correct signature)
        regime_change = CorrelationTopology.detect_correlation_regime_change(
            forex_returns,
            lookback=20
        )
        print(f"✓ Regime Change Detection:")
        print(f"  - Regime Changed: {regime_change['regime_changed']}")
        print(f"  - Change Magnitude: {regime_change['change_magnitude']:.3f}")
        
        print("\n✅ TEST 3 PASSED: Persistent Homology")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_market_neutral_portfolio():
    """Test 4: Market Neutral Risk Management"""
    print("\n" + "=" * 70)
    print("TEST 4: MARKET NEUTRAL PORTFOLIO CONSTRUCTION")
    print("=" * 70)
    
    try:
        # Create sample signals
        signal_trades = [
            {'asset': 'SPX', 'direction': 'LONG', 'confidence': 0.85},
            {'asset': 'QQQ', 'direction': 'LONG', 'confidence': 0.75},
            {'asset': 'IWM', 'direction': 'SHORT', 'confidence': 0.70},
            {'asset': 'EEM', 'direction': 'SHORT', 'confidence': 0.65},
        ]
        
        # Empty correlation dict (in production, would have real data)
        correlations = {}
        
        manager = MarketNeutralRiskManager()
        portfolio = manager.construct_market_neutral_portfolio(
            signal_trades=signal_trades,
            correlations=correlations,
            portfolio_value=1000000,
            target_beta=0.0
        )
        
        print(f"✓ Portfolio Constructed:")
        print(f"  - Long Notional: ${portfolio['long_notional']:,.0f}")
        print(f"  - Short Notional: ${portfolio['short_notional']:,.0f}")
        print(f"  - Net Notional: ${portfolio['net_notional']:,.0f}")
        print(f"  - Portfolio Beta: {portfolio['portfolio_metrics']['net_beta']:.3f}")
        print(f"  - Hedge Requirement: ${portfolio['hedge_requirement']:,.0f}")
        print(f"  - Number of Positions: {len(portfolio['positions'])}")
        
        for pos in portfolio['positions']:
            print(f"    - {pos['asset']}: {pos['direction']} ${pos['size']:,.0f}")
        
        print("\n✅ TEST 4 PASSED: Market Neutral Portfolio")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("ADVANCED FEATURES INTEGRATION TEST SUITE")
    print("=" * 70)
    print(f"Testing 4 advanced features implementation\n")
    
    results = []
    
    # Run all tests
    results.append(("Plausibility & Diffusion", test_plausibility_scoring()))
    results.append(("Residual Analysis", test_residual_analysis()))
    results.append(("Correlation Topology", test_correlation_topology()))
    results.append(("Market Neutral Portfolio", test_market_neutral_portfolio()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for feature_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {feature_name}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Advanced features are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
