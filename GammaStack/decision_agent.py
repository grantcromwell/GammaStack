"""
Agent for making final trade decisions in high-frequency trading (HFT) context.
Combines indicator, pattern, and trend reports to issue a LONG or SHORT order.
"""


def create_final_trade_decider(llm):
    """
    Create a trade decision agent node. The agent uses LLM to synthesize indicator, pattern, and trend reports
    and outputs a final trade decision (LONG or SHORT) with justification and risk-reward ratio.
    """

    def trade_decision_node(state) -> dict:
        indicator_report = state.get("indicator_report", "")
        pattern_report = state.get("pattern_report", "")
        trend_report = state.get("trend_report", "")
        time_frame = state.get("time_frame", "unknown")
        stock_name = state.get("stock_name", "UNKNOWN")
        
        # NEW: Extract advanced signal data
        plausibility_score = state.get("plausibility_score", 0.5)
        signal_regime = state.get("signal_regime", "unknown")
        diffusion_metrics = state.get("diffusion_metrics", {})
        fair_value_model = state.get("fair_value_model", {})
        correlation_report = state.get("correlation_report", "")
        correlation_topology = state.get("correlation_topology", {})
        market_neutral_report = state.get("market_neutral_report", "")
        portfolio_beta = state.get("portfolio_beta", 0.0)
        hedge_requirement = state.get("hedge_requirement", 0)

        # --- Enhanced System prompt for LLM with ALL signals ---
        prompt = f"""You are a high-frequency quantitative trading (HFT) analyst operating on the current {time_frame} K-line chart for {stock_name}. Your task is to issue an immediate execution order: LONG or SHORT. HOLD is prohibited due to HFT constraints.

Your decision should forecast the market move over the next N candlesticks.

Base your decision on the combined strength, alignment, and timing of ALL SIX reports:

### 1. Technical Indicator Report (with Plausibility Analysis):
- Evaluate momentum (MACD, ROC) and oscillators (RSI, Stochastic, Williams %R).
- PLAUSIBILITY SCORE: {plausibility_score:.2f} (0-1 scale, >0.75 = high quality)
- MARKET REGIME: {signal_regime} (trending/mean_reversion/choppy)
- Diffusion Metrics: Brownian {diffusion_metrics.get('brownian_likelihood', 'N/A')}, Mean Reversion {diffusion_metrics.get('mean_reversion_strength', 'N/A')}

### 2. Pattern Report:
- Only act on bullish or bearish patterns if clearly recognizable with confirmed breakout.

### 3. Trend Report (with Fair Value Analysis):
- Fair Price: ${fair_value_model.get('fair_price', 'N/A')}
- Current Price: ${fair_value_model.get('current_price', 'N/A')}
- Deviation: {fair_value_model.get('deviation_pct', 0):.2f}% (>5% = strong mean reversion opportunity)
- Model Quality (R²): {fair_value_model.get('r_squared', 0):.2f}

### 4. Correlation Topology Report:
- Correlation Stability: {correlation_topology.get('stability_score', 0.5):.2f} (>0.7 = stable)
- Regime Changed: {correlation_topology.get('regime_changed', False)}
- Status: {correlation_topology.get('description', 'Unknown')}

### 5. Market Neutral Risk Report:
- Portfolio Beta: {portfolio_beta:.3f} (0.0 = market neutral)
- Hedge Required: ${hedge_requirement:,.0f}
- Status: {'NEUTRAL' if abs(portfolio_beta) < 0.05 else 'SKEWED'}

### Decision Framework:
1. Require Plausibility > 0.75 for high confidence entries
2. Multiple signals aligned (>=3/5 agree on direction)
3. Fair value supports the move (or not contradicting)
4. Correlation regime is stable
5. Portfolio can be hedged to market neutral

Position Sizing: Reduce 50% if Plausibility < 0.60

Output as JSON with decision, confidence, justification, and risk_reward_ratio (1.2-1.8).

---
TECHNICAL INDICATOR REPORT:  
{indicator_report}

PATTERN REPORT:  
{pattern_report}

TREND REPORT:  
{trend_report}

CORRELATION TOPOLOGY REPORT:  
{correlation_report}

MARKET NEUTRAL RISK REPORT:  
{market_neutral_report}
"""

        # --- LLM call for enhanced decision ---
        response = llm.invoke(prompt)

        return {
            "final_trade_decision": response.content,
            "messages": [response],
            "decision_prompt": prompt,
            "decision_metadata": {
                "plausibility": plausibility_score,
                "regime": signal_regime,
                "fair_value_deviation": fair_value_model.get('deviation_pct', 0),
                "correlation_stable": not correlation_topology.get('regime_changed', True),
                "portfolio_beta": portfolio_beta,
                "hedge_required": hedge_requirement > 0,
            }
        }

    return trade_decision_node
