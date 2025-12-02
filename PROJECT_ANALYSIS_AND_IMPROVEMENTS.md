# GammaStack: Project Analysis & Improvement Roadmap

## 📋 Project Overview

**QuantAgent/GammaStack** is a sophisticated multi-agent LLM-powered trading analysis system that leverages LangChain and LangGraph for high-frequency trading (HFT) analysis. The system combines technical indicators, pattern recognition, and trend analysis to provide actionable trade signals.

### Current Architecture
```
Web Interface (Flask)
    ↓
TradingGraph (Orchestrator)
    ↓
SetGraph (LangGraph Engine)
    ├── Indicator Agent (RSI, MACD, ROC, Stochastic, Williams %R)
    ├── Pattern Agent (Candlestick pattern recognition)
    ├── Trend Agent (Support/Resistance trendlines)
    └── Decision Agent (Final LONG/SHORT decision)
    ↓
TechnicalTools Toolkit (TA-Lib based calculations)
```

---

## 🎯 Priority Improvements (HIGH to LOW)

### PRIORITY 1: CRITICAL - Multi-Agent Coordination & Ensemble Quality
**Impact: HIGHEST | Effort: MEDIUM | Risk: MEDIUM**

**Current Issue:**
- Agents operate in **sequential isolation** without active cross-validation
- No consensus mechanism or confidence scoring
- Decision agent makes binary LONG/SHORT without uncertainty quantification
- No agent weighting based on historical performance or market regime
- Lack of contradictory signal detection and resolution

**Improvements:**
1. **Implement Agent Feedback Loop**
   - Add bidirectional communication between agents
   - Pattern agent validates indicator signals (e.g., "MACD crossover confirms Head-Shoulders pattern")
   - Trend agent validates both: "Price near support + RSI oversold" = STRONG BUY signal
   
2. **Confidence Scoring System**
   - Each agent outputs: `(signal, confidence: 0.0-1.0, reasoning)`
   - Decision agent weights outputs by confidence
   - Report conflicting signals explicitly to LLM
   
3. **Agent Performance Tracking**
   - Track win rate per agent type and market regime
   - Use performance-weighted ensemble instead of equal weighting
   - Adapt agent emphasis based on recent accuracy

4. **Consensus Mechanism**
   - Require minimum 2/3 agent agreement before strong signals
   - Neutral/mixed signals → Default to trend (lower risk)

---

### PRIORITY 2: CRITICAL - Real-time Volume & Liquidity Analysis
**Impact: VERY HIGH | Effort: MEDIUM | Risk: LOW**

**Current Issue:**
- **Complete absence of volume analysis** - Only uses OHLCV (ignores V)
- No On-Balance Volume (OBV), Volume Rate of Change, or Volume Profile
- Missing Accumulation/Distribution Line
- Confirms false patterns without volume validation
- No liquidity risk assessment

**Why It Matters for HFT:**
- Volume confirms breakouts (true move vs. wick)
- Detects institutional accumulation before price move
- Prevents trading on illiquid wicks or gaps
- Identifies potential slippage/execution risk

**Improvements:**
1. **Add Volume-Based Indicators**
   ```python
   # New tools to add to TechnicalTools
   - compute_obv() → On-Balance Volume
   - compute_ad_line() → Accumulation/Distribution
   - compute_vpt() → Volume Price Trend
   - compute_cmf() → Chaikin Money Flow
   - compute_volume_sma() → Volume moving average & spike detection
   ```

2. **Volume Confirmation Validator**
   - Pattern agent: "Pattern valid only if volume > 1.5x MA"
   - Indicator agent: "MACD crossover confirmed by volume spike"
   
3. **Liquidity Risk Assessment**
   - Alert if trading near support/resistance with thin volume
   - Adjust risk-reward ratio based on bid-ask spread implications

---

### PRIORITY 3: CRITICAL - Macroeconomic Context & Market Regime Detection
**Impact: VERY HIGH | Effort: HIGH | Risk: MEDIUM**

**Current Issue:**
- **No market regime awareness** - Treats all markets identically
- Missing macroeconomic indicators (DXY, Bonds, Fear Index)
- No correlation analysis with macro assets
- Ignores session times and trading hours
- No Fed event calendar integration

**Why It Matters:**
- Bull vs. Bear regimes need different parameters
- Risk-off days: Short indices, Long gold/DXY
- Crypto heavily correlated with Nasdaq volatility
- Asian/European session behavior differs from US hours

**Improvements:**
1. **Market Regime Detection**
   ```python
   class MarketRegimeDetector:
       - Detect: Bull/Bear/Consolidation/High Volatility
       - Use VIX + Price trend + Volume profile
       - Adjust thresholds per regime
   ```

2. **Macro Context Agent** (NEW - 5th Agent)
   - Input: DXY, Treasury yields, BTC correlation, Fed calendar
   - Output: "Risk-On", "Risk-Off", "Neutral" context
   - Weighting: Penalize short signals on "Risk-On" days

3. **Trading Hours Awareness**
   ```python
   def get_session_context(timestamp):
       return {
           "session": "Asian" | "European" | "US",
           "hours_to_major_event": N,
           "volatility_profile": "low" | "medium" | "high",
           "typical_volume": expected_vol
       }
   ```

4. **Macro Correlation Matrix**
   - Track DXY/Equity, Bond Yield/Equity, Crypto/Nasdaq
   - Adjust decision confidence if correlations break

---

### PRIORITY 4: CRITICAL - Enhanced Risk Management
**Impact: VERY HIGH | Effort: MEDIUM | Risk: LOW**

**Current Issue:**
- Risk-reward ratio hardcoded (1.2-1.8 range)
- No position sizing guidance
- No stop-loss exit logic integration
- Missing tail-risk hedging strategies
- No Kelly Criterion for bet sizing

**Improvements:**
1. **Dynamic Risk-Reward Calculation**
   ```python
   def calculate_risk_reward(
       entry_price, 
       nearest_support, 
       nearest_resistance,
       volatility_atr,
       position_size_guidance,
       account_risk_pct=2.0
   ):
       # Consider: 
       # - Distance to support/resistance
       # - Current volatility (ATR)
       # - Account risk (Kelly Criterion)
       # Return: (stop_loss, take_profit, position_size, rr_ratio)
   ```

2. **Multi-Level Exit Strategy**
   - Partial TP at 1.0x RR (lock in profit)
   - Trailing stop after 0.5x RR hit
   - Hard stop at nearest support/resistance

3. **Drawdown Monitoring**
   - Track consecutive losses
   - Reduce position size if equity down >5%
   - Suggest trade pause if >3 consecutive losses

---

### PRIORITY 5: HIGH - Advanced Chart Pattern Recognition
**Impact: HIGH | Effort: HIGH | Risk: MEDIUM**

**Current Issue:**
- Pattern agent relies purely on **LLM image recognition**
- No algorithmic pattern validation (e.g., symmetry, geometry)
- Can't distinguish between similar patterns (H&S vs Inverted H&S)
- No retest detection after breakout
- Missing confluence zones

**Improvements:**
1. **Algorithmic Pattern Detection**
   ```python
   class PatternDetector:
       def detect_head_shoulders(prices)
       def detect_triangles(prices)
       def detect_double_tops_bottoms(prices)
       def detect_wedges(prices)
       def detect_channels(prices)
       
       # Returns: (pattern_type, confidence, breakout_level, geometry_metrics)
   ```

2. **Confluence Zone Finder**
   - Support/Resistance intersections
   - Fibonacci levels + Trendlines
   - Moving average clusters
   - Previous swing highs/lows
   
3. **Pattern Breakout Validator**
   - Confirm breakout with candle close beyond level
   - Retest detection: "Retest of broken support = strong buy"
   - False breakout detection: Volume drop = likely failure

---

### PRIORITY 6: HIGH - Improve LLM Prompt Engineering & Chain-of-Thought
**Impact: HIGH | Effort: LOW | Risk: LOW**

**Current Issue:**
- Indicator prompts are generic (doesn't emphasize divergences)
- Pattern prompts assume perfect image quality
- Trend prompts don't guide LLM to prioritize recent data
- Decision logic prompt is verbose and potentially confusing

**Improvements:**
1. **Specialist Agent Prompts**
   ```python
   # Indicator Agent - Focus on Divergences & Extremes
   "Look for RSI DIVERGENCE - Price makes new high but RSI doesn't (bearish divergence)"
   
   # Pattern Agent - Clarity on breakout
   "Only call pattern COMPLETE when price closes beyond level, not just touches"
   
   # Trend Agent - Recent bias
   "Weight the last 5 candles 3x heavier than older ones for current support/resistance"
   ```

2. **Few-Shot Learning in Prompts**
   - Add example: "Strong signal: MACD crossover + RSI > 70 + Price above 20-SMA"
   - Add example: "Weak signal: RSI 55 (neutral) + Pattern forming"

3. **Explicit Error Handling**
   - "If RSI and Stochastic conflict, choose based on which is in extreme (>80, <20)"
   - "If no clear pattern, state 'No actionable pattern' instead of guessing"

---

### PRIORITY 7: HIGH - Add Time-Series Decomposition & Cyclical Analysis
**Impact: HIGH | Effort: MEDIUM | Risk: LOW**

**Current Issue:**
- No decomposition into Trend/Seasonal/Residual components
- Missing cyclical pattern detection (daily, weekly cycles)
- No momentum acceleration measurement
- Ignores higher timeframe context

**Improvements:**
1. **STL Decomposition** (Seasonal-Trend decomposition using LOESS)
   ```python
   trend, seasonal, residual = seasonal_decompose(close_prices)
   
   # Trend: long-term direction (compare to current price)
   # Seasonal: predictable cycles (e.g., "Fridays tend to close higher")
   # Residual: anomalies (potential trading opportunities)
   ```

2. **Multi-Timeframe Analysis**
   - Trend Agent: Include 1h and 4h trendlines alongside 15m
   - Confirm 15m breakout only if aligns with 1h/4h trend
   - Avoid counter-trend trades

3. **Acceleration Detection**
   ```python
   momentum = RSI
   momentum_of_momentum = rate_of_change(RSI)  # How fast is RSI rising?
   if momentum_of_momentum > threshold: "ACCELERATION - Strong move incoming"
   ```

---

## 🎯 NEW FEATURES: ADVANCED TECHNIQUES

### FEATURE 1: Gramian Angular Field (GAF) Texture Analysis
**Priority: MEDIUM | Effort: HIGH | Impact: MEDIUM**

**What is GAF?**
- Transforms time-series data into images using angular coordinate mapping
- Captures temporal dependencies as visual patterns
- Enables CNN-based pattern recognition or LLM image understanding

**Integration Path:**
```python
# File: graph_util.py - Add new tool

class GramianAngularField:
    @staticmethod
    def compute_gasf(time_series, image_size=32):
        """
        Gramian Angular Summation Field
        - Converts price series into correlation-based image
        - High correlation regions → Bright pixels
        - Useful for identifying recurring patterns
        """
        # 1. Normalize time series to [-1, 1]
        # 2. Compute arccos to convert to angles
        # 3. Create 2D matrix where M[i,j] = cos(phi_i + phi_j)
        # 4. Rescale to [0, 255] for image
        # 5. Return as PNG image
        
    @staticmethod
    def compute_gadf(time_series, image_size=32):
        """
        Gramian Angular Difference Field
        - Alternative: M[i,j] = sin(phi_i - phi_j)
        - Captures phase differences (momentum changes)
        """
        
    @tool
    def generate_gaf_image(kline_data, field_type="gasf"):
        """Generate GAF texture and return base64 image"""
        # Close prices → Normalize → GAF → PNG → Base64
        # Returns: gaf_image, gaf_description
```

**How to Use:**
1. New agent: **GAF Texture Agent**
   - Input: Close prices last 50 candles
   - Generate GASF & GADF images
   - LLM: "Describe texture patterns. Smooth = trend. Jagged = chop"
   - Output: "High correlation texture" → Trend continuation likely

2. Image Comparison:
   - Compare current 50-candle GAF to historical library
   - "Current pattern matches 85% with 3 previous trends (all +200 pips)"

3. Deep Learning Enhancement (Optional):
   - Train small CNN: GAF image → {Trend, Chop, Reversal}
   - Use as confidence modifier

---

### FEATURE 2: Cumulative Delta Volume (CDV) & Order Flow Analysis
**Priority: MEDIUM | Effort: MEDIUM | Impact: HIGH**

**What is CDV?**
- Cumulative sum of (Upvolume - Downvolume) per candle
- Tracks institutional buying/selling pressure
- Divergences = trend reversals

**Integration Path:**
```python
# File: graph_util.py - Add new tools

@tool
def compute_cumulative_delta_volume(kline_data):
    """
    Calculate CDV using price-based volume allocation
    
    Args:
        kline_data: OHLCV dictionary
    
    Returns:
        {
            'cdv': list of cumulative delta values,
            'cdv_divergence_signals': list of {
                'candle_index': N,
                'type': 'bullish_divergence' | 'bearish_divergence',
                'strength': 0.0-1.0
            }
        }
    """
    df = pd.DataFrame(kline_data)
    
    # Step 1: Allocate volume to up vs down candles
    df['is_up'] = df['Close'] > df['Open']
    df['body_size'] = (df['Close'] - df['Open']).abs()
    df['total_range'] = df['High'] - df['Low']
    
    # Estimate up vs down volume
    df['up_volume'] = df['Volume'] * (df['body_size'] / df['total_range'])
    df['down_volume'] = df['Volume'] - df['up_volume']
    
    # Correct for direction
    df.loc[~df['is_up'], ['up_volume', 'down_volume']] = \
        df.loc[~df['is_up'], ['down_volume', 'up_volume']].values
    
    # Step 2: Compute CDV
    df['delta'] = df['up_volume'] - df['down_volume']
    df['cdv'] = df['delta'].cumsum()
    
    # Step 3: Detect divergences (price makes new high but CDV doesn't)
    signals = detect_cdv_divergences(df)
    
    return {
        'cdv': df['cdv'].tolist(),
        'cdv_signals': signals
    }

def detect_cdv_divergences(df, lookback=14):
    """
    Detect bullish/bearish divergences:
    - Bearish: New price high but CDV lower than previous high
    - Bullish: New price low but CDV higher than previous low
    """
    signals = []
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        current = df.iloc[i]
        
        # Check if price made new high
        if current['Close'] == window['High'].max():
            prev_high_idx = window['High'].idxmax()
            if df.loc[prev_high_idx, 'cdv'] > current['cdv']:
                signals.append({
                    'index': i,
                    'type': 'bearish_divergence',
                    'price_high': current['Close'],
                    'cdv_high': current['cdv'],
                    'strength': (df.loc[prev_high_idx, 'cdv'] - current['cdv']) / df.loc[prev_high_idx, 'cdv']
                })
    
    return signals

@tool
def compute_order_imbalance(kline_data, window=14):
    """
    Calculate Cumulative Volume Delta Imbalance
    - Positive imbalance = Buying pressure
    - Negative imbalance = Selling pressure
    - Extreme imbalance = Reversal likely
    """
    df = pd.DataFrame(kline_data)
    df['delta'] = compute_cumulative_delta_volume(kline_data)['cdv']
    df['imbalance'] = df['delta'].rolling(window).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.abs().sum() if x.abs().sum() > 0 else 0
    )
    return {
        'imbalance': df['imbalance'].tolist()[-window:],
        'imbalance_signal': 'buying_pressure' if df['imbalance'].iloc[-1] > 0.3 else 'selling_pressure' if df['imbalance'].iloc[-1] < -0.3 else 'neutral'
    }
```

**How to Use:**
1. **New Indicator: Cumulative Delta Volume**
   - Add to Indicator Agent tools
   - Alert: "Bearish divergence: Price +2%, CDV -5% → Likely reversal"
   
2. **Order Flow Confluence**
   - Combine with Support/Resistance: "Support + Buying CDV imbalance → STRONG BUY"
   
3. **Volume Confirmation**
   - MACD crossover without CDV spike = weak signal
   - MACD crossover WITH CDV spike = strong signal

---

### FEATURE 3: Footprint / DOM (Depth of Market) Integration
**Priority: MEDIUM-LOW | Effort: VERY HIGH | Impact: MEDIUM**

**What is DOM?**
- Order book snapshot showing buy/sell orders at each price level
- Footprint = DOM evolution over time
- Shows institutional support/resistance before price reaches it

**Challenge:** 
- Requires real-time order book API (Binance WebSocket, etc.)
- Unavailable for most Yahoo Finance data
- Different per exchange

**Integration Path:**
```python
# File: graph_util.py (optional/advanced)

class DepthOfMarketAnalyzer:
    """For crypto/futures with real-time order book access"""
    
    @tool
    def analyze_dom_imbalance(dom_snapshot, levels=10):
        """
        Analyze buy/sell order imbalance
        
        DOM snapshot format:
        {
            'bids': [[price, volume], ...],  # Highest bid first
            'asks': [[price, volume], ...]   # Lowest ask first
        }
        """
        bid_volume = sum(v for _, v in dom_snapshot['bids'][:levels])
        ask_volume = sum(v for _, v in dom_snapshot['asks'][:levels])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        return {
            'imbalance_ratio': imbalance,  # -1 to 1
            'interpretation': 'bullish' if imbalance > 0.3 else 'bearish' if imbalance < -0.3 else 'neutral',
            'bid_pressure': bid_volume,
            'ask_pressure': ask_volume
        }
    
    @tool
    def detect_dom_walls(dom_snapshot, min_size_percentile=80):
        """
        Detect large orders (walls) that could provide support/resistance
        """
        # Find unusually large orders
        # Return: price levels with significant hidden support/resistance
        
    @tool
    def footprint_profile(order_book_history):
        """
        Heatmap of where volume concentrated over time
        """
```

**Recommendation:** 
- **For now: Skip real-time DOM**
- Instead: Use **Tape Replay** (historical order book from API)
- Future enhancement when moving to live exchange APIs

---

### FEATURE 4: Macro Timezone & Session Awareness
**Priority: HIGH | Effort: LOW | Impact: HIGH**

**What is It?**
- Track trading sessions: Asian, European, US opening/closing
- Know when major macro events occur (FOMC, NFP, etc.)
- Adjust volatility expectations per session

**Integration Path:**
```python
# File: new file - session_awareness.py

from datetime import datetime, timezone
import pytz

class SessionAwareness:
    SESSIONS = {
        'Asian': {
            'start': (0, 0),      # 00:00 UTC
            'end': (8, 0),        # 08:00 UTC
            'volatility': 'low',
            'liquidity': 'low'
        },
        'European': {
            'start': (8, 0),      # 08:00 UTC
            'end': (16, 0),       # 16:00 UTC
            'volatility': 'medium',
            'liquidity': 'high'
        },
        'US': {
            'start': (13, 30),    # 13:30 UTC (9:30 AM ET)
            'end': (21, 0),       # 21:00 UTC (5:00 PM ET)
            'volatility': 'high',
            'liquidity': 'very_high'
        }
    }
    
    MACRO_EVENTS = {
        'FOMC Decision': {'volatility': 'extreme', 'direction_bias': 'unpredictable'},
        'NFP Release': {'volatility': 'extreme', 'session': 'US'},
        'ECB Decision': {'volatility': 'very_high', 'session': 'European'},
        'China PMI': {'volatility': 'medium', 'session': 'Asian'},
        'Fed Speakers': {'volatility': 'medium'}
    }
    
    @staticmethod
    def get_current_session(timestamp_utc):
        """Return current trading session info"""
        hour = timestamp_utc.hour
        
        for session_name, times in SessionAwareness.SESSIONS.items():
            if times['start'] <= hour < times['end']:
                return {
                    'session': session_name,
                    'volatility_profile': times['volatility'],
                    'liquidity': times['liquidity'],
                    'recommended_risk_level': 'low' if times['liquidity'] == 'low' else 'normal'
                }
        
        return {'session': 'overlap', 'volatility_profile': 'variable'}
    
    @staticmethod
    def hours_to_next_event(current_time, event_name):
        """Calculate hours until next macro event"""
        # Lookup event calendar from external API
        # Return hours remaining
        
    @staticmethod
    def is_high_impact_event_today(timestamp):
        """Check if today has FOMC, NFP, or ECB decision"""
        # Lookup against calendar
        
    @tool
    def get_session_context(timestamp):
        """
        Returns dict with:
        - session: str (Asian/European/US)
        - volatility: str (low/medium/high/extreme)
        - hours_to_major_event: int
        - key_event_today: str or None
        - recommended_trade_size: float (0.5 = half normal on low liquidity)
        """

# Integration in indicator_agent.py
def create_indicator_agent(llm, toolkit, session_awareness):
    def indicator_agent_node(state):
        session = session_awareness.get_current_session(state['timestamp'])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are analyzing {state['stock_name']} on {state['time_frame']} timeframe.
            Current Trading Session: {session['session']}
            Expected Volatility: {session['volatility']}
            
            {f"⚠️ MAJOR EVENT TODAY: {session.get('key_event_today')} in {session.get('hours_to_event')} hours" if session.get('hours_to_event', 999) < 4 else ""}
            
            Adjust your confidence in longer-term predictions if a major event is imminent.
            """)
        ])
        # ... rest of agent logic
```

**Usage:**
1. **Modify Decision Agent Prompt**
   ```python
   decision_prompt = f"""
   Current session: {session['session']}
   Volatility environment: {session['volatility']}
   
   If NFP in 1 hour: Reduce position size by 50% and use tighter stops
   If Asian session: Expect lower volume confirmation on breakouts
   """
   ```

2. **Adjust Risk Parameters**
   ```python
   risk_reward_ratio = base_rr * session['liquidity_multiplier']
   position_size = base_size / (1 + session['volatility_multiplier'])
   ```

---

## 📊 Implementation Priority Summary Table

| Priority | Feature | Impact | Effort | Timeline | Dependencies |
|----------|---------|--------|--------|----------|--------------|
| P1 | Multi-Agent Coordination | ★★★★★ | ⭐⭐⭐ | 2-3 weeks | None |
| P2 | Volume Analysis | ★★★★★ | ⭐⭐ | 1-2 weeks | None |
| P3 | Macro Context Agent | ★★★★★ | ⭐⭐⭐⭐ | 3-4 weeks | API access |
| P4 | Risk Management | ★★★★★ | ⭐⭐ | 1-2 weeks | None |
| P5 | Advanced Patterns | ★★★★ | ⭐⭐⭐⭐ | 3-4 weeks | scipy |
| P6 | Prompt Engineering | ★★★★ | ★ | 1 week | None |
| P7 | Time-Series Decomposition | ★★★★ | ⭐⭐⭐ | 2 weeks | statsmodels |
| F1 | Gramian Angular Field | ★★★ | ⭐⭐⭐⭐ | 2-3 weeks | numpy, scipy |
| F2 | Cumulative Delta Volume | ★★★★ | ⭐⭐⭐ | 2 weeks | None |
| F3 | DOM/Footprint | ★★ | ⭐⭐⭐⭐⭐ | 4+ weeks | Exchange API |
| F4 | Macro Timezone Awareness | ★★★★ | ⭐ | 1 week | None |

---

## 🚀 Quick Win (Next 2 Weeks)

Recommended starting point for maximum ROI:

1. **Week 1:**
   - [ ] Add Volume tools (OBV, CMF, Volume MA)
   - [ ] Implement Session Awareness module
   - [ ] Improve indicator agent prompts (divergence focus)

2. **Week 2:**
   - [ ] Add confidence scoring to agents
   - [ ] Implement multi-agent consensus check
   - [ ] Dynamic risk-reward calculation based on ATR

---

## 📈 Long-Term Roadmap (3-6 Months)

1. **Month 1:** Volume + Session + Confidence (P2, P4, P6)
2. **Month 2:** Macro context agent + Pattern validation (P3, P5)
3. **Month 3:** GAF texture analysis + CDV integration (F1, F2)
4. **Month 4-6:** Advanced features, backtesting, live testing

---

## 🔍 Code Quality Observations

**Strengths:**
✅ Clean separation of concerns (agents, toolkit, graph setup)
✅ Good use of LangChain/LangGraph abstractions
✅ Web interface is well-structured
✅ Tool definitions are clear

**Areas for Improvement:**
⚠️ No error handling in agent nodes
⚠️ Hardcoded parameters (14 periods for RSI, etc.)
⚠️ No caching of expensive computations
⚠️ Limited logging and debugging info
⚠️ No unit tests
⚠️ Documentation could be more detailed

---

## 📝 Next Steps

1. **Review this analysis** with team
2. **Prioritize features** based on your trading strategy
3. **Create GitHub issues** for each priority item
4. **Start with P2 (Volume)** - fastest ROI, no dependencies
5. **Parallelize P4 (Risk Mgmt)** and P6 (Prompts)

