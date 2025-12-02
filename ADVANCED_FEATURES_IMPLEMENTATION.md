# Implementation Guide: Advanced Features Integration

## Table of Contents
1. [Gramian Angular Field (GAF) - Detailed Implementation](#1-gramian-angular-field-gaf)
2. [Cumulative Delta Volume (CDV) - Detailed Implementation](#2-cumulative-delta-volume-cdv)
3. [DOM/Footprint Integration - Roadmap](#3-domfootprint)
4. [Macro Timezone Awareness - Complete Code](#4-macro-timezone)

---

## 1. Gramian Angular Field (GAF)

### What It Does
Converts price time-series into 2D images that capture temporal correlations as visual patterns. The LLM can then "see" and describe patterns that would be hard to express numerically.

### Mathematical Background
```
1. Normalize time series to [-1, 1]
   x_norm = 2 * (x - min(x)) / (max(x) - min(x)) - 1

2. Convert to angles in [-π/2, π/2]
   φᵢ = arccos(xᵢ)  where xᵢ ∈ [-1, 1]

3. Compute Gramian Angular Summation Field (GASF)
   GASF[i, j] = cos(φᵢ + φⱼ)
   
   OR Gramian Angular Difference Field (GADF)
   GADF[i, j] = sin(φᵢ - φⱼ)

4. Scale to [0, 255] for image visualization
   image_pixel = (matrix_value + 1) / 2 * 255
```

### Code Implementation

**File: graph_util.py (add to TechnicalTools class)**

```python
import cv2  # pip install opencv-python
from scipy.ndimage import gaussian_filter

class TechnicalTools:
    
    @staticmethod
    def normalize_to_range(series, min_val=-1, max_val=1):
        """Normalize time series to [min_val, max_val]"""
        s_min = series.min()
        s_max = series.max()
        if s_max == s_min:
            return np.ones_like(series) * (min_val + max_val) / 2
        return min_val + (series - s_min) / (s_max - s_min) * (max_val - min_val)
    
    @staticmethod
    def compute_gasf(time_series, image_size=32, method='imagining'):
        """
        Gramian Angular Summation Field (GASF)
        
        Use case: Captures overall temporal dependencies and correlations
        - Smooth texture = Trending behavior
        - Noisy texture = Choppy/ranging behavior
        
        Args:
            time_series: np.array or list of prices
            image_size: Size of output image (default 32x32)
            method: 'imagining' (rescale to image_size) or 'raw' (exact size)
        
        Returns:
            np.array: 2D matrix [0, 255] representing the image
        """
        # Convert to numpy array
        ts = np.asarray(time_series, dtype=np.float32)
        
        # Step 1: Normalize to [-1, 1]
        ts_norm = TechnicalTools.normalize_to_range(ts, -1, 1)
        
        # Step 2: Convert to angles (arccos maps [-1, 1] to [π, 0])
        ts_angle = np.arccos(np.clip(ts_norm, -1, 1))
        
        # Step 3: Compute Gramian matrix (GASF)
        # GASF[i, j] = cos(angle_i + angle_j)
        n = len(ts_angle)
        gasf = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                gasf[i, j] = np.cos(ts_angle[i] + ts_angle[j])
        
        # Step 4: Resize to target image size
        if method == 'imagining' and n != image_size:
            gasf = cv2.resize(gasf, (image_size, image_size))
        
        # Step 5: Scale to [0, 255] for visualization
        gasf_image = ((gasf + 1) / 2 * 255).astype(np.uint8)
        
        return gasf_image
    
    @staticmethod
    def compute_gadf(time_series, image_size=32, method='imagining'):
        """
        Gramian Angular Difference Field (GADF)
        
        Use case: Captures phase differences and momentum changes
        - Sharp transitions = Momentum reversals
        - Diagonal patterns = Consistent momentum
        
        Args:
            time_series: np.array or list of prices
            image_size: Size of output image (default 32x32)
            method: 'imagining' (rescale to image_size) or 'raw' (exact size)
        
        Returns:
            np.array: 2D matrix [0, 255] representing the image
        """
        ts = np.asarray(time_series, dtype=np.float32)
        ts_norm = TechnicalTools.normalize_to_range(ts, -1, 1)
        ts_angle = np.arccos(np.clip(ts_norm, -1, 1))
        
        # GADF[i, j] = sin(angle_i - angle_j)
        n = len(ts_angle)
        gadf = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                gadf[i, j] = np.sin(ts_angle[i] - ts_angle[j])
        
        if method == 'imagining' and n != image_size:
            gadf = cv2.resize(gadf, (image_size, image_size))
        
        gadf_image = ((gadf + 1) / 2 * 255).astype(np.uint8)
        
        return gadf_image
    
    @staticmethod
    @tool
    def generate_gaf_texture_images(
        kline_data: Annotated[
            dict,
            "Dictionary containing OHLCV data with 'Close' prices"
        ],
        lookback: Annotated[int, "Number of candles to analyze (default 50)"] = 50,
        image_size: Annotated[int, "Output image size (default 32)"] = 32
    ) -> dict:
        """
        Generate Gramian Angular Field (GAF) images from price data.
        
        This creates two texture images:
        - GASF: Shows overall trend/correlation strength
        - GADF: Shows momentum and phase changes
        
        The LLM can analyze these images to understand:
        - Is price trending (smooth diagonal pattern)?
        - Is price ranging (noisy/scattered pattern)?
        - Is momentum accelerating or decelerating?
        
        Args:
            kline_data: OHLCV data dictionary
            lookback: Candles to include (default 50)
            image_size: GAF output size (32x32 or 64x64)
        
        Returns:
            dict with:
                - gaf_gasf_image: Base64 GASF image
                - gaf_gadf_image: Base64 GADF image
                - gaf_interpretation: Text description
                - gaf_smoothness_score: 0-1 (1=perfect trend)
        """
        df = pd.DataFrame(kline_data)
        closes = df['Close'].tail(lookback).values
        
        # Generate both GAF images
        gasf_image = TechnicalTools.compute_gasf(closes, image_size=image_size)
        gadf_image = TechnicalTools.compute_gadf(closes, image_size=image_size)
        
        # Analyze texture characteristics
        gasf_variance = np.var(gasf_image)
        gadf_variance = np.var(gadf_image)
        
        # Smoothness score (0-1): how diagonal/smooth is GASF?
        # Extract diagonal: smoother = trending
        gasf_diagonal = np.diag(gasf_image)
        smoothness = 1 - (np.std(gasf_diagonal) / 255)
        
        # Convert images to base64
        def img_to_base64(img_array):
            _, buffer = cv2.imencode('.png', img_array)
            return base64.b64encode(buffer).decode('utf-8')
        
        gasf_b64 = img_to_base64(gasf_image)
        gadf_b64 = img_to_base64(gadf_image)
        
        # Generate interpretation
        if smoothness > 0.7:
            trend_desc = "Strong trend (diagonal GASF pattern)"
        elif smoothness > 0.5:
            trend_desc = "Moderate trend with some consolidation"
        else:
            trend_desc = "Choppy/ranging behavior (scattered GASF)"
        
        return {
            "gaf_gasf_image": gasf_b64,
            "gaf_gadf_image": gadf_b64,
            "gaf_interpretation": f"{trend_desc}. GADF shows {'smooth momentum changes' if gadf_variance < 2500 else 'abrupt momentum reversals'}.",
            "gaf_smoothness_score": float(smoothness),
            "gaf_trend_confirmation": "strong" if smoothness > 0.7 else "weak",
            "gaf_description": f"GASF (Trend Correlation): variance={gasf_variance:.0f}. GADF (Momentum): variance={gadf_variance:.0f}"
        }
```

### Integration: Create GAF Agent

**File: gaf_agent.py (new file)**

```python
"""
GAF Texture Analysis Agent - Interprets Gramian Angular Field images
"""

def create_gaf_agent(llm, toolkit):
    """
    Create a GAF Texture Analysis agent that uses vision to understand
    price pattern textures and confirms trends via pattern recognition.
    """
    
    def gaf_agent_node(state):
        # Generate GAF images
        gaf_result = toolkit.generate_gaf_texture_images(
            state["kline_data"],
            lookback=50
        )
        
        tools = [toolkit.generate_gaf_texture_images]
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a texture analysis expert in trading. You specialize in analyzing 
                Gramian Angular Field (GAF) images that visualize price patterns.
                
                GASF (Gramian Angular Summation Field):
                - Diagonal pattern from bottom-left to top-right = Strong uptrend
                - Diagonal pattern from top-left to bottom-right = Strong downtrend
                - Scattered/noisy pattern = Choppy/ranging market
                - Smooth gradient = Consistent trend
                
                GADF (Gramian Angular Difference Field):
                - Bright band along diagonal = Stable momentum
                - Dark areas = Momentum reversals or extreme moves
                - Scattered bright spots = Erratic momentum changes
                
                Given the GAF texture images below, describe:
                1. Is the market TRENDING or RANGING?
                2. What is the trend STRENGTH (weak/moderate/strong)?
                3. Are there signs of MOMENTUM CHANGES?
                4. What is the CONFIDENCE in your assessment (0-100%)?
                """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Build messages
        messages = state.get("messages", [])
        if not messages:
            messages = [
                HumanMessage(
                    content=f"Analyze these GAF texture images:\n\nGASF Image: {gaf_result['gaf_gasf_image'][:100]}...\nGADF Image: {gaf_result['gaf_gadf_image'][:100]}...\n\nInterpretation: {gaf_result['gaf_interpretation']}"
                )
            ]
        
        chain = prompt | llm.bind_tools(tools)
        ai_response = chain.invoke(messages)
        messages.append(ai_response)
        
        # Process tool calls if any
        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for call in ai_response.tool_calls:
                tool_result = toolkit.generate_gaf_texture_images(**call["args"])
                messages.append(ToolMessage(tool_call_id=call["id"], content=json.dumps(tool_result)))
            
            # Get final response after tool use
            final_response = chain.invoke(messages)
            messages.append(final_response)
        else:
            final_response = ai_response
        
        # Extract final report
        gaf_report = final_response.content if hasattr(final_response, 'content') else str(final_response)
        
        return {
            "gaf_report": gaf_report,
            "gaf_images": gaf_result,
            "messages": messages,
            "gaf_trend_confirmation": gaf_result["gaf_trend_confirmation"]
        }
    
    return gaf_agent_node
```

### Integration into Main Graph

**File: graph_setup.py (modify set_graph method)**

```python
from gaf_agent import create_gaf_agent  # Add import

def set_graph(self):
    # ... existing code ...
    
    all_agents = ["indicator", "pattern", "trend", "gaf"]  # Add "gaf"
    
    # Create GAF agent node
    agent_nodes["gaf"] = create_gaf_agent(self.agent_llm, self.toolkit)
    
    # Add to graph
    graph.add_node("GAF Texture Agent", agent_nodes["gaf"])
    
    # Update edges to insert GAF before Decision Maker
    # Old: Trend → Decision Maker
    # New: Trend → GAF Texture → Decision Maker
    
    graph.remove_edge("Trend Agent", "Decision Maker")  # If it exists
    graph.add_edge("Trend Agent", "GAF Texture Agent")
    graph.add_edge("GAF Texture Agent", "Decision Maker")
```

### How LLM Uses This
```
Indicator Report: "MACD positive, RSI 45, no divergence"
Pattern Report: "Ascending triangle forming"
Trend Report: "Uptrend within channel"
GAF Report: "GASF shows smooth diagonal trend confirmation, GADF shows stable momentum. Strong confidence 85%."

Decision Agent conclusion:
"All signals aligned (✓ Indicator, ✓ Pattern, ✓ Trend, ✓ GAF texture).
High confidence = LONG with RR 1.5"
```

---

## 2. Cumulative Delta Volume (CDV)

### What It Does
- Tracks buying vs selling pressure by analyzing volume in up vs down candles
- Detects divergences between price and volume commitment
- Critical for confirming moves and detecting reversals

### Code Implementation

**File: graph_util.py (add to TechnicalTools class)**

```python
class TechnicalTools:
    
    @staticmethod
    @tool
    def compute_cumulative_delta_volume(
        kline_data: Annotated[
            dict,
            "Dictionary with OHLCV data"
        ],
        method: Annotated[str, "Calculation method: 'close_based' or 'hlc'"] = 'close_based'
    ) -> dict:
        """
        Compute Cumulative Delta Volume (CDV) and detect buying/selling imbalances.
        
        The idea: In each candle, allocate volume to buying vs selling based on where price closed.
        Candle with Close > Open = mostly buying volume
        Candle with Close < Open = mostly selling volume
        
        CDV cumulates this over time to show net buying/selling pressure.
        
        Returns:
            {
                'cdv': List of cumulative delta values,
                'cdv_normal': Normalized CDV [-1, 1],
                'volume_imbalance': Bullish/bearish/neutral,
                'divergence_signals': List of detected divergences
            }
        """
        df = pd.DataFrame(kline_data)
        
        if method == 'close_based':
            # Simple method: If Close > Open, all volume is "up", else "down"
            df['up'] = df['Close'] > df['Open']
            df['body_size'] = (df['Close'] - df['Open']).abs()
            df['total_range'] = (df['High'] - df['Low']).abs()
            
            # Allocate volume proportionally to body size
            df['up_volume'] = df['Volume'] * df['body_size'] / (df['total_range'] + 1e-8)
            df['down_volume'] = df['Volume'] - df['up_volume']
            
            # Correct for direction
            mask = ~df['up']
            df.loc[mask, ['up_volume', 'down_volume']] = df.loc[mask, ['down_volume', 'up_volume']].values
        
        elif method == 'hlc':
            # Relative method: Where did close fall relative to high/low?
            df['hl_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
            df['up_volume'] = df['Volume'] * df['hl_position']
            df['down_volume'] = df['Volume'] * (1 - df['hl_position'])
        
        # Calculate delta and cumulative
        df['delta'] = df['up_volume'] - df['down_volume']
        df['cdv'] = df['delta'].cumsum()
        
        # Normalize CDV
        cdv_max = df['cdv'].abs().max()
        df['cdv_normal'] = df['cdv'] / (cdv_max + 1e-8)
        
        # Detect current imbalance
        recent_imbalance = df['delta'].tail(1).values[0]
        if recent_imbalance > df['delta'].std():
            volume_imbalance = "bullish"
        elif recent_imbalance < -df['delta'].std():
            volume_imbalance = "bearish"
        else:
            volume_imbalance = "neutral"
        
        # Detect divergences
        divergences = TechnicalTools._detect_cdv_divergences(df, lookback=14)
        
        return {
            'cdv': df['cdv'].tolist(),
            'cdv_normal': df['cdv_normal'].tolist(),
            'volume_imbalance': volume_imbalance,
            'volume_imbalance_strength': float(abs(recent_imbalance / (df['delta'].std() + 1e-8))),
            'divergence_signals': divergences,
            'cdv_description': f"CDV shows {volume_imbalance} volume. {len(divergences)} divergence signals detected."
        }
    
    @staticmethod
    def _detect_cdv_divergences(df, lookback=14):
        """
        Detect buying/selling divergences:
        - Bearish: Price makes new high but CDV doesn't → Selling pressure
        - Bullish: Price makes new low but CDV doesn't → Buying pressure
        """
        signals = []
        
        for i in range(lookback, len(df)):
            window = df.iloc[max(0, i-lookback):i+1]
            current = df.iloc[i]
            
            # Price made new high
            if current['Close'] == window['High'].max() and len(window) > 1:
                prev_window = df.iloc[max(0, i-lookback):i]
                prev_high_idx = prev_window['Close'].idxmax()
                prev_high_cdv = df.loc[prev_high_idx, 'cdv']
                
                if current['cdv'] < prev_high_cdv:  # CDV didn't confirm
                    divergence_strength = (prev_high_cdv - current['cdv']) / (abs(prev_high_cdv) + 1e-8)
                    signals.append({
                        'index': i,
                        'type': 'bearish_divergence',
                        'description': f"Price new high but CDV lower → Selling pressure",
                        'strength': min(1.0, float(divergence_strength)),
                        'candle_count_back': i - prev_high_idx
                    })
            
            # Price made new low
            if current['Close'] == window['Low'].min() and len(window) > 1:
                prev_window = df.iloc[max(0, i-lookback):i]
                prev_low_idx = prev_window['Close'].idxmin()
                prev_low_cdv = df.loc[prev_low_idx, 'cdv']
                
                if current['cdv'] > prev_low_cdv:  # CDV didn't confirm
                    divergence_strength = (current['cdv'] - prev_low_cdv) / (abs(prev_low_cdv) + 1e-8)
                    signals.append({
                        'index': i,
                        'type': 'bullish_divergence',
                        'description': f"Price new low but CDV higher → Buying pressure",
                        'strength': min(1.0, float(divergence_strength)),
                        'candle_count_back': i - prev_low_idx
                    })
        
        # Keep only strongest recent signals
        return sorted(signals, key=lambda x: x['strength'], reverse=True)[:5]
    
    @staticmethod
    @tool
    def compute_order_imbalance_ratio(
        kline_data: Annotated[dict, "OHLCV dictionary"],
        window: Annotated[int, "Lookback window for imbalance ratio"] = 14
    ) -> dict:
        """
        Calculate Order Imbalance Ratio:
        - Ratio of buying to selling volume momentum
        - Extreme ratios (>1.5 or <0.67) suggest reversal likelihood
        
        Formula:
        Imbalance = (Cumsum of up volume - Cumsum of down volume) / (Total volume in window)
        """
        df = pd.DataFrame(kline_data)
        
        # Calculate up/down volumes
        df['up'] = df['Close'] >= df['Open']
        df['body_size'] = (df['Close'] - df['Open']).abs()
        df['total_range'] = (df['High'] - df['Low']).abs()
        df['up_volume'] = df['Volume'] * df['body_size'] / (df['total_range'] + 1e-8)
        df['down_volume'] = df['Volume'] - df['up_volume']
        
        df.loc[~df['up'], ['up_volume', 'down_volume']] = df.loc[~df['up'], ['down_volume', 'up_volume']].values
        
        # Rolling imbalance
        df['total_vol'] = df['up_volume'] + df['down_volume']
        df['up_vol_ratio'] = df['up_volume'] / (df['total_vol'] + 1e-8)
        df['imbalance'] = df['up_vol_ratio'].rolling(window).mean()
        
        # Interpret
        current_imbalance = df['imbalance'].iloc[-1]
        if current_imbalance > 0.6:
            signal = "strong_buying"
        elif current_imbalance > 0.55:
            signal = "buying"
        elif current_imbalance < 0.4:
            signal = "strong_selling"
        elif current_imbalance < 0.45:
            signal = "selling"
        else:
            signal = "balanced"
        
        return {
            'imbalance_ratio': float(current_imbalance),
            'signal': signal,
            'buying_pressure': float(df['up_vol_ratio'].tail(window).mean()),
            'last_n_imbalances': df['imbalance'].tail(window).tolist()
        }
```

### Integration with Indicator Agent

**File: indicator_agent.py (modify to include CDV)**

```python
def create_indicator_agent(llm, toolkit):
    def indicator_agent_node(state):
        # Add CDV tools
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
            toolkit.compute_cumulative_delta_volume,  # NEW
            toolkit.compute_order_imbalance_ratio,     # NEW
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a high-frequency trading analyst.
                
                NEW: Use volume-based tools (CDV, Order Imbalance) to CONFIRM price moves.
                
                **Volume Confirmation Rules:**
                1. Strong indicator signal (MACD cross) + Bullish CDV = HIGH confidence
                2. Strong indicator signal (MACD cross) + Bearish CDV = CONFLICTING (warn)
                3. Weak indicator signal + Extreme imbalance = Setup for reversal
                
                Always call compute_cumulative_delta_volume and compute_order_imbalance_ratio.
                Use results to validate other indicator signals.
                """,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        # ... rest of agent code ...
```

---

## 3. DOM/Footprint Integration Roadmap

### Not Recommended for Now

**Reason:** Requires real-time order book access
- Yahoo Finance doesn't provide DOM data
- Would require exchange API (Binance, Kraken, etc.)
- Different schema per exchange
- Significant complexity for marginal ROI

### When to Implement (Future)

**Triggers:**
1. When switching from Yahoo Finance to live exchange feed
2. For crypto/futures trading (where DOM is critical)
3. When deploying to proprietary exchanges

### Placeholder Code (for future reference)

```python
# graph_util.py - Add when live DOM available

class DepthOfMarketAnalyzer:
    
    @staticmethod
    @tool
    def analyze_dom_snapshot(
        dom_snapshot: Annotated[dict, "{'bids': [[price, size], ...], 'asks': [[price, size], ...]}"],
        levels: Annotated[int, "Number of price levels to analyze"] = 10
    ) -> dict:
        """Analyze buy/sell order imbalance at current price level"""
        
        bids = dom_snapshot.get('bids', [])[:levels]
        asks = dom_snapshot.get('asks', [])[:levels]
        
        bid_vol = sum(v for _, v in bids)
        ask_vol = sum(v for _, v in asks)
        
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        
        return {
            'bid_ask_imbalance': float(imbalance),
            'interpretation': 'bullish' if imbalance > 0.2 else 'bearish' if imbalance < -0.2 else 'neutral',
            'bid_pressure': float(bid_vol),
            'ask_pressure': float(ask_vol),
            'spread': float(asks[0][0] - bids[0][0]) if asks and bids else None
        }
```

---

## 4. Macro Timezone Awareness - Complete Code

**File: session_awareness.py (create new file)**

```python
"""
Session Awareness & Macro Calendar Integration

Tracks trading sessions, major economic events, and adjusts trading parameters accordingly.
"""

from datetime import datetime, timedelta
import pytz
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SessionInfo:
    """Information about current trading session"""
    name: str
    volatility: str  # low, medium, high, extreme
    liquidity: str   # low, medium, high, very_high
    time_until_end: int  # minutes
    recommended_trade_size: float  # multiplier: 0.5 = half normal


class SessionAwareness:
    """Manages trading session context and macro awareness"""
    
    # Define trading sessions in UTC
    SESSIONS = {
        'Asian': {
            'start_hour': 0,
            'end_hour': 8,
            'volatility': 'low',
            'liquidity': 'low',
            'typical_spreads': 'wide'
        },
        'European': {
            'start_hour': 8,
            'end_hour': 16,
            'volatility': 'medium',
            'liquidity': 'high',
            'typical_spreads': 'medium'
        },
        'US': {
            'start_hour': 13,  # 9:30 AM ET = 13:30 UTC (accounting for DST variance)
            'end_hour': 21,
            'volatility': 'high',
            'liquidity': 'very_high',
            'typical_spreads': 'tight'
        },
        'Overlap_EU_US': {
            'start_hour': 13,
            'end_hour': 16,
            'volatility': 'very_high',
            'liquidity': 'very_high',
            'typical_spreads': 'extremely_tight'
        }
    }
    
    # Major macro events - these should be fetched from real calendar
    MACRO_EVENTS = {
        'FOMC Decision': {
            'volatility_impact': 'extreme',
            'affected_instruments': ['SPX', 'ES', 'NQ', 'DXY', 'UST'],
            'avoid_trade_hours_before': 2,
            'typical_move_pips': 500,
            'session': 'US'
        },
        'NFP Release': {
            'volatility_impact': 'extreme',
            'affected_instruments': ['SPX', 'DXY', 'EURUSD'],
            'avoid_trade_hours_before': 1,
            'typical_move_pips': 300,
            'session': 'US',
            'time': '13:30 UTC'  # 8:30 AM ET
        },
        'ECB Decision': {
            'volatility_impact': 'very_high',
            'affected_instruments': ['EURUSD', 'DAX', 'European stocks'],
            'avoid_trade_hours_before': 2,
            'typical_move_pips': 400,
            'session': 'European'
        },
        'China PMI': {
            'volatility_impact': 'medium',
            'affected_instruments': ['A50', 'CNY', 'Commodities'],
            'avoid_trade_hours_before': 1,
            'typical_move_pips': 100,
            'session': 'Asian'
        },
        'US Jobs Report': {
            'volatility_impact': 'extreme',
            'affected_instruments': ['All indices', 'Gold', 'Bonds'],
            'avoid_trade_hours_before': 1,
            'typical_move_pips': 400,
            'session': 'US'
        }
    }
    
    @staticmethod
    def get_current_session(timestamp_utc: datetime) -> SessionInfo:
        """
        Determine current trading session.
        
        Args:
            timestamp_utc: datetime in UTC
        
        Returns:
            SessionInfo object with session details
        """
        hour = timestamp_utc.hour
        
        # Check for EU/US overlap
        if 13 <= hour < 16:
            session_name = 'Overlap_EU_US'
        elif 0 <= hour < 8:
            session_name = 'Asian'
        elif 8 <= hour < 13:
            session_name = 'European'
        elif 13 <= hour < 21:
            session_name = 'US'
        else:
            session_name = 'Post-US'  # 21:00-00:00
        
        session_config = SessionAwareness.SESSIONS.get(
            session_name,
            SessionAwareness.SESSIONS['Asian']  # default
        )
        
        # Calculate time until session end
        end_hour = session_config['end_hour']
        end_time = timestamp_utc.replace(hour=end_hour, minute=0, second=0)
        if end_time <= timestamp_utc:
            end_time += timedelta(days=1)
        time_until_end = int((end_time - timestamp_utc).total_seconds() / 60)
        
        # Size multiplier based on liquidity
        liquidity_multipliers = {
            'low': 0.5,           # Reduce to 50% on low liquidity
            'medium': 0.75,
            'high': 1.0,          # Normal size
            'very_high': 1.25     # Can increase on high liquidity
        }
        trade_size_mult = liquidity_multipliers.get(session_config['liquidity'], 0.5)
        
        return SessionInfo(
            name=session_name,
            volatility=session_config['volatility'],
            liquidity=session_config['liquidity'],
            time_until_end=time_until_end,
            recommended_trade_size=trade_size_mult
        )
    
    @staticmethod
    def get_macro_event_today(
        timestamp_utc: datetime,
        asset_name: str
    ) -> Optional[Tuple[str, Dict]]:
        """
        Check if there's a major macro event today that affects this asset.
        
        Args:
            timestamp_utc: Current time in UTC
            asset_name: Asset being traded (e.g., 'SPX', 'DXY')
        
        Returns:
            (event_name, event_details) or None if no major event
        """
        # In production, integrate with: FRED API, Trading Economics, Investing.com
        # For now, return example
        
        # Placeholder: Check if Friday (often has events)
        if timestamp_utc.weekday() == 4:  # Friday
            for event_name, event_info in SessionAwareness.MACRO_EVENTS.items():
                if asset_name in event_info['affected_instruments']:
                    hours_until = 12  # Placeholder - would fetch from calendar
                    if hours_until < event_info['avoid_trade_hours_before']:
                        return (event_name, event_info)
        
        return None
    
    @staticmethod
    def get_correlation_regime(
        dxy_change: float,  # % change in DXY
        vix_level: float,    # VIX value
        bond_yield_change: float  # % change in US 10Y
    ) -> str:
        """
        Determine market regime from macro indicators.
        
        Risk-On (equities rally):
        - DXY down, VIX low (<15), yields falling or flat
        
        Risk-Off (equities crash):
        - DXY up, VIX high (>30), yields rising
        
        Mixed:
        - Conflicting signals
        """
        signals = {
            'dxy': 'risk_off' if dxy_change > 0.5 else 'risk_on' if dxy_change < -0.5 else 'neutral',
            'vix': 'risk_off' if vix_level > 25 else 'risk_on' if vix_level < 12 else 'neutral',
            'yields': 'risk_off' if bond_yield_change > 10 else 'risk_on' if bond_yield_change < -10 else 'neutral'
        }
        
        risk_off_count = sum(1 for v in signals.values() if v == 'risk_off')
        risk_on_count = sum(1 for v in signals.values() if v == 'risk_on')
        
        if risk_off_count >= 2:
            return 'RISK_OFF'
        elif risk_on_count >= 2:
            return 'RISK_ON'
        else:
            return 'MIXED'
    
    @staticmethod
    def get_session_adjustment_factors(session_info: SessionInfo) -> Dict[str, float]:
        """
        Calculate trading parameter adjustments based on session.
        
        Returns:
            {
                'position_size_multiplier': 0.5-1.5,
                'risk_reward_multiplier': 0.8-1.5,
                'confidence_reduction': 0.0-0.3,  # How much to reduce LLM confidence
                'stop_loss_buffer': 0.5-2.0,  # Widen stops on low liquidity
            }
        """
        liquidity_map = {
            'low': {'size': 0.4, 'rr': 0.8, 'conf': 0.3, 'buffer': 2.0},
            'medium': {'size': 0.7, 'rr': 0.9, 'conf': 0.15, 'buffer': 1.3},
            'high': {'size': 1.0, 'rr': 1.0, 'conf': 0.0, 'buffer': 1.0},
            'very_high': {'size': 1.3, 'rr': 1.1, 'conf': 0.0, 'buffer': 0.8},
        }
        
        adj = liquidity_map.get(session_info.liquidity, liquidity_map['medium'])
        
        return {
            'position_size_multiplier': adj['size'],
            'risk_reward_multiplier': adj['rr'],
            'confidence_reduction': adj['conf'],
            'stop_loss_buffer': adj['buffer']
        }


def integrate_session_awareness_in_decision_agent():
    """
    Example of how to integrate session awareness into decision agent.
    """
    
    # In decision_agent.py:
    def create_final_trade_decider(llm, session_awareness):
        def trade_decision_node(state) -> dict:
            # Get current session
            timestamp = pd.to_datetime(state['timestamp'])
            timestamp_utc = timestamp.tz_convert('UTC') if timestamp.tz else timestamp
            
            session = session_awareness.get_current_session(timestamp_utc)
            macro_event = session_awareness.get_macro_event_today(
                timestamp_utc,
                state['stock_name']
            )
            
            # Check for macro event warning
            event_warning = ""
            if macro_event:
                event_name, event_details = macro_event
                hours_before = event_details['avoid_trade_hours_before']
                event_warning = f"""
                ⚠️ WARNING: {event_name} occurring in < {hours_before} hours!
                - Expected volatility: {event_details['volatility_impact']}
                - Typical move: ± {event_details['typical_move_pips']} pips
                - REDUCE position size by 50%
                - Use TIGHTER stops
                """
            
            # Adjust risk-reward based on session
            session_adj = session_awareness.get_session_adjustment_factors(session)
            
            # Update decision prompt
            prompt = f"""
            Current Trading Session: {session.name}
            Volatility Profile: {session.volatility}
            Liquidity: {session.liquidity}
            
            Session Adjustments:
            - Position Size: {session_adj['position_size_multiplier']:.1f}x normal
            - Risk-Reward: {session_adj['risk_reward_multiplier']:.1f}x normal
            - Reduce confidence: -{session_adj['confidence_reduction']*100:.0f}%
            - Widen stops by: {session_adj['stop_loss_buffer']:.1f}x
            
            {event_warning}
            
            ... rest of decision prompt ...
            """
            
            # ... rest of agent code ...
```

**Integration in trading_graph.py:**

```python
from session_awareness import SessionAwareness

class TradingGraph:
    def __init__(self, config=None):
        # ... existing code ...
        self.session_awareness = SessionAwareness()
        
        # Pass to decision agent
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
            self.session_awareness  # NEW
        )
```

---

## Summary

These implementations provide:

1. **GAF Texture Analysis**: LLM-friendly visual representation of price patterns
2. **Cumulative Delta Volume**: Order flow confirmation of price moves  
3. **DOM/Footprint**: Placeholder for future live trading integration
4. **Macro Timezone Awareness**: Session-based risk adjustments

All can be integrated independently and in any order based on your priorities.

