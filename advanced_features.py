"""
Advanced Features: Plausibility, Diffusion, Residual Analysis, Persistent Homology, Market Neutral Risk

This module provides cutting-edge quantitative tools for:
1. Plausibility & Diffusion scoring
2. Residual-based discount pricing detection
3. Persistent homology on correlation structure
4. Market neutral risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Annotated, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. PLAUSIBILITY & DIFFUSION ANALYSIS
# ============================================================================

@dataclass
class DiffusionMetrics:
    """Metrics describing price movement diffusion pattern"""
    brownian_likelihood: float      # 0-1: How much like random walk
    mean_reversion_strength: float  # 0-1: How much price reverts to mean
    drift_component: float          # % per candle
    diffusion_speed: float          # Volatility annualized
    expected_range_next_n: Tuple[float, float]  # Predicted price range
    confidence: float               # 0-1: How confident in prediction


class PlausibilityAnalyzer:
    """
    Rates the plausibility of detected trading signals.
    
    High plausibility signals = likely to work
    Low plausibility signals = likely to be fake/noise
    """
    
    @staticmethod
    def compute_diffusion_metrics(
        price_series: np.ndarray,
        lookback: int = 50,
        forecast_periods: int = 20
    ) -> DiffusionMetrics:
        """
        Analyze how price diffuses through the market.
        
        Mathematical basis:
        dS = μ·S·dt + σ·S·dW  (Geometric Brownian Motion)
        
        If residuals are white noise → pure Brownian motion
        If residuals autocorrelated → mean reversion or trends
        If residuals have ARCH → volatility clustering
        
        Args:
            price_series: Historical prices
            lookback: Window for analysis
            forecast_periods: Candles to forecast range
        
        Returns:
            DiffusionMetrics object
        """
        prices = np.asarray(price_series[-lookback:], dtype=np.float32)
        
        # Compute log returns (more stable for GBM)
        log_returns = np.diff(np.log(prices))
        
        # Step 1: Detect trend vs mean reversion
        # =====================================
        # Use AR(1) model: r_t = α·r_{t-1} + ε_t
        # α > 0.3 → Trending (momentum)
        # α < -0.3 → Mean reverting
        # |α| < 0.1 → Random walk
        
        ar_coef = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
        
        if ar_coef > 0.3:
            mean_reversion_str = 0.1  # Low MR, high trend
            brownian_like = 0.3
        elif ar_coef < -0.3:
            mean_reversion_str = 0.8  # High MR
            brownian_like = 0.2
        else:
            mean_reversion_str = 0.4
            brownian_like = 0.7
        
        # Step 2: Compute drift
        # ====================
        # μ = E[r_t] (annualized)
        drift = np.mean(log_returns) * 252  # Convert to annual
        
        # Step 3: Compute diffusion (volatility)
        # ====================================
        # σ = Std[r_t] (annualized)
        diffusion = np.std(log_returns) * np.sqrt(252)
        
        # Step 4: Forecast next N periods
        # ==============================
        # Using stochastic formula: E[S_t+N] = S_t * exp(μ·N/252)
        current_price = prices[-1]
        S_t = current_price
        
        # 68% confidence interval = ±1σ over N periods
        time_sqrt_n = np.sqrt(forecast_periods / 252)
        upper_bound = S_t * np.exp((drift / 252) * forecast_periods + diffusion * time_sqrt_n)
        lower_bound = S_t * np.exp((drift / 252) * forecast_periods - diffusion * time_sqrt_n)
        
        # Confidence decreases with longer forecasts
        confidence = 1.0 / (1.0 + forecast_periods / 10)
        
        return DiffusionMetrics(
            brownian_likelihood=brownian_like,
            mean_reversion_strength=mean_reversion_str,
            drift_component=drift,
            diffusion_speed=diffusion,
            expected_range_next_n=(lower_bound, upper_bound),
            confidence=confidence
        )
    
    @staticmethod
    def score_signal_plausibility(
        price_data: Dict,
        signal: str,  # 'LONG' or 'SHORT'
        entry_price: float,
        indicator_confidence: float = 0.5,
        diffusion_metrics: Optional[DiffusionMetrics] = None,
        lookback: int = 50
    ) -> Dict[str, float]:
        """
        Rate how plausible a detected signal is.
        
        Factors:
        1. Is price in trending or ranging regime?
        2. Is entry at support/resistance (mean reversion) or breakout (trend)?
        3. Is volatility stable or spiking?
        4. Is the signal aligned with market structure?
        
        Returns:
            {
                'plausibility_score': 0-1,
                'regime_alignment': 0-1,
                'entry_quality': 0-1,
                'volatility_regime': str,
                'recommendation': str
            }
        """
        df = pd.DataFrame(price_data).tail(lookback)
        prices = df['Close'].values
        
        if diffusion_metrics is None:
            diffusion_metrics = PlausibilityAnalyzer.compute_diffusion_metrics(prices)
        
        # Factor 1: Regime alignment
        # ==========================
        if diffusion_metrics.mean_reversion_strength > 0.6:
            # Mean reverting regime
            regime = "mean_reversion"
            # LONG plausible near support, SHORT plausible near resistance
            if signal == 'LONG':
                regime_align = 0.8 if entry_price < df['Close'].mean() else 0.3
            else:
                regime_align = 0.8 if entry_price > df['Close'].mean() else 0.3
        else:
            # Trending regime
            regime = "trending"
            # Check if price is above/below MA
            ma_20 = df['Close'].rolling(20).mean().iloc[-1]
            if signal == 'LONG':
                regime_align = 0.8 if entry_price > ma_20 else 0.4
            else:
                regime_align = 0.8 if entry_price < ma_20 else 0.4
        
        # Factor 2: Entry quality (where relative to recent range)
        # ========================================================
        recent_high = df['High'].tail(10).max()
        recent_low = df['Low'].tail(10).min()
        recent_range = recent_high - recent_low
        
        if recent_range > 0:
            entry_percentile = (entry_price - recent_low) / recent_range
        else:
            entry_percentile = 0.5
        
        # Good entries:
        # LONG: 30-50% from low (approaching support, not too far)
        # SHORT: 50-70% from low (approaching resistance)
        if signal == 'LONG':
            entry_quality = 1.0 - abs(entry_percentile - 0.35) * 2
        else:
            entry_quality = 1.0 - abs(entry_percentile - 0.65) * 2
        
        entry_quality = np.clip(entry_quality, 0, 1)
        
        # Factor 3: Volatility regime
        # ===========================
        atr = PlausibilityAnalyzer._compute_atr(df)
        vol_percentile = diffusion_metrics.diffusion_speed
        
        if vol_percentile > 0.03:  # >3% annual daily move
            vol_regime = "high_volatility"
            vol_quality = 0.6  # Higher vol = riskier
        elif vol_percentile < 0.01:
            vol_regime = "low_volatility"
            vol_quality = 0.4  # Too quiet, low signal quality
        else:
            vol_regime = "normal"
            vol_quality = 0.9  # Ideal
        
        # Factor 4: Trend strength
        # ========================
        # Use Hurst exponent (> 0.5 = trending, < 0.5 = mean reverting)
        hurst = PlausibilityAnalyzer._compute_hurst_exponent(prices)
        if signal == 'LONG':
            trend_align = hurst  # Want hurst > 0.5 for uptrend
        else:
            trend_align = 1.0 - hurst  # Want hurst < 0.5 for downtrend
        
        # Composite plausibility score
        # ============================
        weights = {
            'regime': 0.25,
            'entry': 0.25,
            'volatility': 0.20,
            'trend': 0.20,
            'indicator_conf': 0.10
        }
        
        plausibility = (
            regime_align * weights['regime'] +
            entry_quality * weights['entry'] +
            vol_quality * weights['volatility'] +
            trend_align * weights['trend'] +
            indicator_confidence * weights['indicator_conf']
        )
        
        # Generate recommendation
        if plausibility > 0.75:
            recommendation = "HIGH - Strong setup"
        elif plausibility > 0.60:
            recommendation = "MEDIUM - Acceptable setup"
        elif plausibility > 0.45:
            recommendation = "LOW - Weak setup, smaller position"
        else:
            recommendation = "VERY LOW - Skip this signal"
        
        return {
            'plausibility_score': float(np.clip(plausibility, 0, 1)),
            'regime_alignment': float(regime_align),
            'entry_quality': float(entry_quality),
            'volatility_regime': vol_regime,
            'trend_alignment': float(trend_align),
            'recommendation': recommendation,
            'regime': regime,
            'diffusion_metrics': {
                'brownian_likelihood': diffusion_metrics.brownian_likelihood,
                'mean_reversion_strength': diffusion_metrics.mean_reversion_strength,
                'drift': diffusion_metrics.drift_component,
                'volatility': diffusion_metrics.diffusion_speed
            }
        }
    
    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Average True Range"""
        df = df.copy()
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = (df['High'] - df['Close'].shift()).abs()
        df['tr3'] = (df['Low'] - df['Close'].shift()).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        return df['tr'].rolling(period).mean().iloc[-1]
    
    @staticmethod
    def _compute_hurst_exponent(prices: np.ndarray, lags: int = 20) -> float:
        """
        Hurst Exponent: measures trending vs mean reverting
        H > 0.5: Trending (momentum)
        H = 0.5: Random walk
        H < 0.5: Mean reverting
        """
        tau = []
        for lag in range(1, min(lags, len(prices) // 2)):
            # Compute rescaled range
            price_ret = np.diff(prices)
            y = np.cumsum(price_ret[:lag])
            mean_y = np.mean(y)
            y = y - mean_y
            r = np.max(y) - np.min(y)
            s = np.std(price_ret[:lag])
            if s > 0:
                tau.append(r / s)
        
        # H = log(R/S) / log(lag)
        if len(tau) > 1:
            poly = np.polyfit(np.log(range(1, len(tau)+1)), np.log(tau), 1)
            return poly[0]  # slope = Hurst exponent
        return 0.5


# ============================================================================
# 2. RESIDUAL-BASED DISCOUNT PRICING DETECTION
# ============================================================================

class ResidualAnalyzer:
    """
    Find mispriced assets using regression residuals.
    
    Idea: Build fair value model for each asset (using technical + macro inputs)
    Find deviations from fair value = opportunities for mean reversion
    """
    
    @staticmethod
    def build_fair_value_model(
        price_data: Dict,
        predictor_data: Dict,  # {'rsi': [...], 'macd': [...], 'volume': [...]}
        lookback: int = 100
    ) -> Dict:
        """
        Build regression model to predict fair price.
        
        Model: Fair_Price = β₀ + β₁·RSI + β₂·MACD + β₃·Volume + ε
        
        Args:
            price_data: Dict with 'Close' array
            predictor_data: Dict with indicator arrays
            lookback: Training window
        
        Returns:
            {
                'model_coefs': {indicator: beta},
                'r_squared': float,
                'residual_std': float,
                'fair_prices': [predicted values],
                'residuals': [actual - predicted],
                'discounts': [% deviation from fair]
            }
        """
        df = pd.DataFrame(price_data).tail(lookback)
        prices = df['Close'].values
        
        # Build feature matrix
        features = []
        for indicator, values in predictor_data.items():
            values_array = np.asarray(values[-lookback:])
            # Normalize to 0-1
            val_min, val_max = np.nanmin(values_array), np.nanmax(values_array)
            if val_max > val_min:
                normalized = (values_array - val_min) / (val_max - val_min)
            else:
                normalized = np.ones_like(values_array) * 0.5
            features.append(normalized)
        
        X = np.column_stack(features)
        y = prices
        
        # Add bias term
        X = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression: β = (X'X)⁻¹ X'y
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            beta = np.zeros(X.shape[1])
        
        # Predictions and residuals
        y_pred = X @ beta
        residuals = y - y_pred
        
        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Residual standard deviation (prediction error)
        residual_std = np.std(residuals)
        
        # Discount = (actual - fair) / fair
        discounts = residuals / (np.abs(y_pred) + 1e-8)
        
        return {
            'model_coefs': {name: float(beta[i+1]) for i, name in enumerate(predictor_data.keys())},
            'bias': float(beta[0]),
            'r_squared': float(r_squared),
            'residual_std': float(residual_std),
            'fair_prices': y_pred.tolist(),
            'residuals': residuals.tolist(),
            'discounts': discounts.tolist(),
            'current_residual': float(residuals[-1]),
            'current_discount_pct': float(discounts[-1] * 100)
        }
    
    @staticmethod
    def detect_mispriced_assets(
        assets_data: Dict[str, Dict],  # {'SPX': price_dict, 'ES': price_dict}
        predictors: Dict[str, Dict],   # {'SPX': {'rsi': [...], 'macd': [...]}, ...}
        threshold_std: float = 2.0
    ) -> Dict[str, Dict]:
        """
        Screen for mispriced assets (residuals > threshold).
        
        Opportunity: Price > Fair + 2σ → Overvalued (SHORT)
                   Price < Fair - 2σ → Undervalued (LONG)
        
        Args:
            assets_data: Price data for each asset
            predictors: Technical indicators for each asset
            threshold_std: How many std devs = mispriced (default 2.0 = 95% confidence)
        
        Returns:
            Dict with mispriced assets and opportunity scores
        """
        opportunities = {}
        
        for asset_name, price_dict in assets_data.items():
            if asset_name not in predictors:
                continue
            
            model = ResidualAnalyzer.build_fair_value_model(
                price_dict,
                predictors[asset_name],
                lookback=100
            )
            
            current_residual = model['current_residual']
            residual_std = model['residual_std']
            z_score = current_residual / (residual_std + 1e-8)
            
            if abs(z_score) > threshold_std:
                if z_score > threshold_std:
                    # Overvalued
                    signal = 'SHORT'
                    strength = min(z_score / threshold_std, 2.0)  # Cap at 2.0
                else:
                    # Undervalued
                    signal = 'LONG'
                    strength = min(abs(z_score) / threshold_std, 2.0)
                
                opportunities[asset_name] = {
                    'signal': signal,
                    'opportunity_strength': float(strength),
                    'z_score': float(z_score),
                    'current_price': float(price_dict['Close'][-1]),
                    'fair_price': float(model['fair_prices'][-1]),
                    'deviation_pct': float(model['current_discount_pct']),
                    'model_r_squared': model['r_squared'],
                    'mean_reversion_probability': float(1 - stats.norm.sf(abs(z_score)))  # CDF of normal
                }
        
        return opportunities


# ============================================================================
# 3. PERSISTENT HOMOLOGY ON CORRELATION STRUCTURE
# ============================================================================

class CorrelationTopology:
    """
    Analyze forex pair correlations using persistent homology.
    
    Detects:
    - Correlation regime changes (0-dim: when do correlations cluster)
    - Hidden multi-dimensional structure (1-dim: loops/cycles in correlations)
    - Fragmented markets (2-dim: voids/gaps in correlation space)
    """
    
    @staticmethod
    def compute_correlation_matrix(
        returns_dict: Dict[str, np.ndarray],
        window: int = 60
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute rolling correlation matrix for multiple forex pairs.
        
        Args:
            returns_dict: {'EURUSD': [...], 'GBPUSD': [...], ...}
            window: Rolling window
        
        Returns:
            (correlation_matrix, pair_names)
        """
        pair_names = list(returns_dict.keys())
        
        # Stack returns
        returns_array = np.column_stack([
            returns_dict[pair][-window:] for pair in pair_names
        ])
        
        # Compute correlation
        corr_matrix = np.corrcoef(returns_array.T)
        
        return corr_matrix, pair_names
    
    @staticmethod
    def compute_persistent_homology(
        corr_matrix: np.ndarray,
        pair_names: List[str],
        min_correlation: float = 0.3
    ) -> Dict:
        """
        Compute persistent homology on correlation graph.
        
        Process:
        1. Convert correlation matrix to distance metric: d = 1 - |ρ|
        2. Build simplicial complex by adding edges as correlation increases
        3. Track birth/death of topological features
        4. Identify stable features (long lifespans = real structure)
        
        Returns:
            {
                'correlation_clusters': List[List[str]],  # Strongly correlated pairs
                'correlation_loops': List[Tuple[str, str, str]],  # Triangular structure
                'regime_changes': List[Dict],  # Where correlations break
                'stability_score': 0-1  # How stable is correlation structure
            }
        """
        # Convert correlation to distance
        # Distance = 1 - |correlation|  (or use 1 - correlation for signed)
        distance_matrix = 1 - np.abs(corr_matrix)
        np.fill_diagonal(distance_matrix, 0)
        
        # Compute hierarchical clustering
        condensed_distances = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_distances, method='ward')
        
        # Identify clusters (0-dimensional homology)
        # Use distance threshold to cut dendrogram
        from scipy.cluster.hierarchy import fcluster
        clusters = fcluster(Z, t=0.5, criterion='distance')
        
        # Group pairs by cluster
        correlation_clusters = [
            [pair for pair, cluster_id in zip(pair_names, clusters) if cluster_id == c]
            for c in np.unique(clusters)
        ]
        correlation_clusters = [c for c in correlation_clusters if len(c) > 1]
        
        # Detect triangular structures (1-dimensional homology)
        # Look for triplets with strong pairwise correlations
        triangular_structures = []
        min_corr_threshold = min_correlation
        
        for i in range(len(pair_names)):
            for j in range(i+1, len(pair_names)):
                for k in range(j+1, len(pair_names)):
                    corr_ij = np.abs(corr_matrix[i, j])
                    corr_ik = np.abs(corr_matrix[i, k])
                    corr_jk = np.abs(corr_matrix[j, k])
                    
                    if all([corr_ij > min_corr_threshold,
                           corr_ik > min_corr_threshold,
                           corr_jk > min_corr_threshold]):
                        triangular_structures.append({
                            'pairs': [pair_names[i], pair_names[j], pair_names[k]],
                            'correlations': [corr_ij, corr_ik, corr_jk],
                            'strength': float(np.mean([corr_ij, corr_ik, corr_jk]))
                        })
        
        # Compute stability score
        # How many pairs are in clusters? Higher = more structure = less noise
        pairs_in_clusters = sum(len(c) for c in correlation_clusters)
        stability = pairs_in_clusters / max(len(pair_names), 1)
        
        return {
            'correlation_clusters': correlation_clusters,
            'correlation_loops': triangular_structures,
            'num_clusters': len(correlation_clusters),
            'num_loops': len(triangular_structures),
            'stability_score': float(stability),
            'regime_type': 'correlated' if stability > 0.7 else 'fragmented',
            'corr_matrix': corr_matrix.tolist()
        }
    
    @staticmethod
    def detect_correlation_regime_change(
        returns_data: Dict[str, np.ndarray],
        lookback: int = 60,
        comparison_periods: int = 3
    ) -> Dict:
        """
        Detect when forex correlations change (regime shift).
        
        Args:
            returns_data: Dict of returns series
            lookback: Window size
            comparison_periods: How many periods back to compare
        
        Returns:
            {
                'regime_changed': bool,
                'change_magnitude': 0-1,
                'current_stability': 0-1,
                'interpretation': str
            }
        """
        # Compute correlation for current and previous periods
        current_corr, _ = CorrelationTopology.compute_correlation_matrix(
            returns_data, window=lookback
        )
        
        previous_corr, _ = CorrelationTopology.compute_correlation_matrix(
            {k: v[:-lookback] for k, v in returns_data.items()},
            window=lookback
        )
        
        # Correlation change = Frobenius norm of difference
        corr_change = np.linalg.norm(current_corr - previous_corr, ord='fro')
        
        # Normalize by matrix size
        max_possible_change = np.sqrt(current_corr.size * 4)  # Max change is 2 per element
        change_magnitude = min(corr_change / max_possible_change, 1.0)
        
        # Current stability = how consistent are correlations
        current_stability = 1.0 - np.std(np.abs(current_corr[np.triu_indices_from(current_corr, k=1)]))
        current_stability = np.clip(current_stability, 0, 1)
        
        if change_magnitude > 0.3:
            regime_changed = True
            if change_magnitude > 0.6:
                interpretation = "MAJOR regime shift - Correlations breaking"
            else:
                interpretation = "Moderate correlation change"
        else:
            regime_changed = False
            interpretation = "Correlations stable"
        
        return {
            'regime_changed': regime_changed,
            'change_magnitude': float(change_magnitude),
            'current_stability': float(current_stability),
            'interpretation': interpretation
        }


# ============================================================================
# 4. MARKET NEUTRAL RISK MANAGEMENT MODEL
# ============================================================================

class MarketNeutralRiskManager:
    """
    Hedging and pairs trading for market-neutral portfolio construction.
    
    Strategies:
    1. Beta-neutral hedging: Offset portfolio beta with index futures
    2. Pairs trading: Long undervalued, short overvalued in same sector
    3. Statistical arbitrage: Exploit temporary divergences
    4. Correlation-based hedging: Use high-correlation assets to hedge
    """
    
    @staticmethod
    def compute_portfolio_beta(
        holdings: Dict[str, float],  # {'SPX': 100 shares, 'AAPL': 50 shares}
        asset_betas: Dict[str, float],  # {'SPX': 1.0, 'AAPL': 1.2}
        asset_prices: Dict[str, float]  # Current prices
    ) -> float:
        """
        Calculate portfolio beta.
        
        β_portfolio = Σ(weight_i × β_i)
        where weight_i = (price_i × shares_i) / total_value
        """
        portfolio_value = sum(
            holdings.get(asset, 0) * asset_prices.get(asset, 0)
            for asset in asset_prices.keys()
        )
        
        if portfolio_value == 0:
            return 1.0
        
        portfolio_beta = sum(
            (holdings.get(asset, 0) * asset_prices.get(asset, 1)) / portfolio_value * 
            asset_betas.get(asset, 1.0)
            for asset in asset_prices.keys()
        )
        
        return portfolio_beta
    
    @staticmethod
    def calculate_hedge_size(
        current_portfolio_beta: float,
        target_beta: float = 0.0,  # Market neutral = 0
        hedge_instrument_beta: float = 1.0,  # e.g., SPX futures have β=1
        portfolio_value: float = 1000000
    ) -> float:
        """
        Calculate hedge size to achieve target beta.
        
        Formula: Hedge_Size = (Current_Beta - Target_Beta) / Hedge_Beta × Portfolio_Value
        
        Returns:
            Notional value to short (in hedge instrument)
        
        Example:
            Portfolio β = 1.5 (15% more volatile than market)
            Want β = 0 (market neutral)
            SPX futures β = 1.0
            → Hedge = (1.5 - 0) / 1.0 × $1M = $1.5M short SPX
        """
        if hedge_instrument_beta == 0:
            return 0
        
        beta_difference = current_portfolio_beta - target_beta
        hedge_size = (beta_difference / hedge_instrument_beta) * portfolio_value
        
        return float(hedge_size)
    
    @staticmethod
    def pairs_trading_signal(
        long_asset: str,
        short_asset: str,
        long_price: float,
        short_price: float,
        correlation: float,
        residual: float,
        residual_std: float
    ) -> Dict:
        """
        Generate pairs trade signal (long one, short the other).
        
        Setup:
        - Find two correlated assets (ρ > 0.7)
        - One is overvalued (deviation > 1.5σ)
        - One is undervalued (deviation < -1.5σ)
        - Pair trade: LONG undervalued, SHORT overvalued
        
        Rationale: If correlation holds, deviation will revert
        
        Returns:
            {
                'signal': 'PAIRS_TRADE',
                'long_asset': str,
                'short_asset': str,
                'position_ratio': float,
                'confidence': 0-1,
                'target_spread': float,
                'stop_loss_spread': float
            }
        """
        # Position size ratio = inverse of price ratio (equal dollar exposure)
        ratio = long_price / short_price
        
        # Confidence = correlation strength × deviation magnitude
        deviation_z_score = abs(residual) / (residual_std + 1e-8)
        confidence = (correlation - 0.5) / 0.5 * (min(deviation_z_score / 2.0, 1.0))
        confidence = np.clip(confidence, 0, 1)
        
        # Mean reversion target: spread narrows by residual amount
        target_spread = residual * 0.5  # Assume 50% reversion
        
        # Risk: if correlation breaks
        stop_loss_spread = residual * 1.5  # Exit if diverges further
        
        return {
            'signal': 'PAIRS_TRADE',
            'long_asset': long_asset,
            'short_asset': short_asset,
            'position_ratio': float(ratio),
            'confidence': float(confidence),
            'current_spread': float(residual),
            'target_spread': float(target_spread),
            'stop_loss_spread': float(stop_loss_spread),
            'correlation': float(correlation),
            'recommendation': 'EXECUTE' if confidence > 0.6 else 'MONITOR'
        }
    
    @staticmethod
    def construct_market_neutral_portfolio(
        signal_trades: List[Dict],  # List of LONG/SHORT signals
        correlations: Dict[Tuple[str, str], float],  # Pair correlations
        portfolio_value: float = 1000000,
        target_beta: float = 0.0,
        max_correlation_in_portfolio: float = 0.5
    ) -> Dict:
        """
        Construct market-neutral portfolio from signals.
        
        Process:
        1. Take LONG/SHORT signals
        2. Ensure long and short notional are balanced
        3. Remove redundant positions (high correlation)
        4. Calculate hedge to neutralize beta
        
        Returns:
            {
                'positions': [{'asset': str, 'size': float, 'direction': str}, ...],
                'long_notional': float,
                'short_notional': float,
                'net_notional': float,  # Should be close to 0
                'hedge_requirement': float,  # Dollar size of hedge
                'portfolio_metrics': {
                    'net_beta': float,
                    'long_beta': float,
                    'short_beta': float,
                    'correlation_risk': float
                }
            }
        """
        
        # Separate LONG and SHORT trades
        long_trades = [t for t in signal_trades if t.get('signal_direction') == 'LONG']
        short_trades = [t for t in signal_trades if t.get('signal_direction') == 'SHORT']
        
        # Equal weight each signal
        long_notional = (portfolio_value / 2) / len(long_trades) if long_trades else 0
        short_notional = (portfolio_value / 2) / len(short_trades) if short_trades else 0
        
        positions = []
        for trade in long_trades:
            positions.append({
                'asset': trade['asset'],
                'size': long_notional,
                'direction': 'LONG',
                'confidence': trade.get('confidence', 0.5)
            })
        
        for trade in short_trades:
            positions.append({
                'asset': trade['asset'],
                'size': short_notional,
                'direction': 'SHORT',
                'confidence': trade.get('confidence', 0.5)
            })
        
        # Remove highly correlated positions to reduce redundancy
        filtered_positions = MarketNeutralRiskManager._remove_correlated_positions(
            positions, correlations, max_correlation_in_portfolio
        )
        
        # Calculate portfolio stats
        long_notional_actual = sum(p['size'] for p in filtered_positions if p['direction'] == 'LONG')
        short_notional_actual = sum(p['size'] for p in filtered_positions if p['direction'] == 'SHORT')
        net_notional = long_notional_actual - short_notional_actual
        
        # Average beta (assume all long β=1, all short β=1 for simplicity)
        long_beta = 1.0  # Simplified
        short_beta = 1.0
        net_beta = (long_notional_actual - short_notional_actual) / (portfolio_value + 1e-8)
        
        # Correlation risk: sum of off-diagonal correlation values
        corr_risk = sum(abs(c) for c in correlations.values()) / max(len(correlations), 1)
        
        return {
            'positions': filtered_positions,
            'long_notional': float(long_notional_actual),
            'short_notional': float(short_notional_actual),
            'net_notional': float(net_notional),
            'portfolio_value': float(portfolio_value),
            'hedge_requirement': float(abs(net_beta) * portfolio_value),
            'portfolio_metrics': {
                'net_beta': float(net_beta),
                'long_beta': float(long_beta),
                'short_beta': float(short_beta),
                'correlation_risk': float(corr_risk),
                'position_count': len(filtered_positions),
                'is_market_neutral': abs(net_notional) < portfolio_value * 0.05
            }
        }
    
    @staticmethod
    def _remove_correlated_positions(
        positions: List[Dict],
        correlations: Dict[Tuple[str, str], float],
        threshold: float = 0.5
    ) -> List[Dict]:
        """Keep highest confidence, remove lower confidence if correlated"""
        # Sort by confidence descending
        positions = sorted(positions, key=lambda x: x.get('confidence', 0), reverse=True)
        
        kept = []
        for pos in positions:
            # Check correlation with already-kept positions
            keep = True
            for kept_pos in kept:
                key = tuple(sorted([pos['asset'], kept_pos['asset']]))
                corr = correlations.get(key, 0)
                
                # Skip if highly correlated and same direction
                if (abs(corr) > threshold and 
                    pos['direction'] == kept_pos['direction']):
                    keep = False
                    break
            
            if keep:
                kept.append(pos)
        
        return kept


# ============================================================================
# WRAPPER FUNCTIONS FOR AGENT INTEGRATION
# ============================================================================

def analyze_signal_plausibility(
    signal: str,
    entry_price: float,
    kline_data: Dict,
    indicator_confidence: float = 0.5
) -> Dict:
    """Wrapper for agent integration"""
    analyzer = PlausibilityAnalyzer()
    return analyzer.score_signal_plausibility(
        kline_data, signal, entry_price, indicator_confidence
    )


def find_mispriced_assets(
    assets_returns: Dict[str, Dict],
    predictors: Dict[str, Dict]
) -> Dict:
    """Wrapper for screening multiple assets"""
    analyzer = ResidualAnalyzer()
    return analyzer.detect_mispriced_assets(assets_returns, predictors)


def analyze_forex_correlations(
    forex_returns: Dict[str, np.ndarray]
) -> Dict:
    """Wrapper for correlation structure analysis"""
    topology = CorrelationTopology()
    corr_matrix, pair_names = topology.compute_correlation_matrix(forex_returns)
    homology = topology.compute_persistent_homology(corr_matrix, pair_names)
    
    regime_change = topology.detect_correlation_regime_change(forex_returns)
    
    return {
        **homology,
        'regime_change': regime_change
    }


def construct_market_neutral_portfolio_wrapper(
    signals: List[Dict],
    correlation_matrix: Dict,
    portfolio_value: float = 1000000
) -> Dict:
    """Wrapper for portfolio construction"""
    manager = MarketNeutralRiskManager()
    return manager.construct_market_neutral_portfolio(signals, correlation_matrix, portfolio_value)
