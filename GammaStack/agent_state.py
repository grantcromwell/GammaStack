from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class IndicatorAgentState(TypedDict):
    """State type for the Indicator Agent including messages, input data, and analysis result."""

    kline_data: Annotated[
        dict, "OHLCV dictionary used for computing technical indicators"
    ]
    time_frame: Annotated[str, "time period for k line data provided"]
    stock_name: Annotated[dict, "stock name for prompt"]

    # Indicator Agent Tools output values (explicitly added per indicator)
    rsi: Annotated[List[float], "Relative Strength Index values"]
    macd: Annotated[List[float], "MACD line values"]
    macd_signal: Annotated[List[float], "MACD signal line values"]
    macd_hist: Annotated[List[float], "MACD histogram values"]
    stoch_k: Annotated[List[float], "Stochastic Oscillator %K values"]
    stoch_d: Annotated[List[float], "Stochastic Oscillator %D values"]
    roc: Annotated[List[float], "Rate of Change values"]
    willr: Annotated[List[float], "Williams %R values"]
    indicator_report: Annotated[
        str, "Final indicator agent summary report to be used by downstream agents"
    ]

    # Pattern Agent
    pattern_image: Annotated[
        str, "Base64-encoded K-line chart for pattern recognition agent use"
    ]
    pattern_image_filename: Annotated[
        str, "Local file path to saved K-line chart image"
    ]
    pattern_image_description: Annotated[
        str, "Brief description of the generated K-line image"
    ]
    pattern_report: Annotated[
        str, "Final pattern agent summary report to be used by downstream agents"
    ]

    # Trend Agent
    trend_image: Annotated[
        str,
        "Base64-encoded trend-annotated candlestick (K-line) chart for trend recognition agent use",
    ]
    trend_image_filename: Annotated[
        str, "Local file path to saved trendline-enhanced K-line chart image"
    ]
    trend_image_description: Annotated[
        str,
        "Brief description of the chart, including presence of support/resistance lines and visual characteristics",
    ]
    trend_report: Annotated[
        str,
        "Final trend analysis summary, describing structure, directional bias, and technical observations for downstream agents",
    ]

    # Advanced Features: Plausibility & Diffusion
    plausibility_score: Annotated[
        float, "Signal plausibility rating (0-1) based on regime and entry quality"
    ]
    diffusion_metrics: Annotated[
        dict, "Brownian motion analysis: brownian_likelihood, mean_reversion_strength, drift, volatility"
    ]
    signal_regime: Annotated[
        str, "Trading regime: 'trending', 'mean_reversion', or 'choppy'"
    ]
    
    # Advanced Features: Residual Analysis
    mispricing_opportunities: Annotated[
        dict, "Residual-based discount pricing signals: {asset: {signal, deviation_pct, z_score}}"
    ]
    fair_value_model: Annotated[
        dict, "Fair value regression model: {r_squared, residual_std, coefs}"
    ]
    
    # Advanced Features: Correlation Topology
    correlation_topology: Annotated[
        dict, "Persistent homology results: clusters, loops, stability_score"
    ]
    correlation_regime_change: Annotated[
        dict, "Forex correlation shift detection: regime_changed, magnitude, interpretation"
    ]
    
    # Advanced Features: Market Neutral
    market_neutral_signals: Annotated[
        dict, "Pairs trading and hedging opportunities"
    ]
    portfolio_beta: Annotated[
        float, "Current portfolio beta for market neutrality check"
    ]
    hedge_requirement: Annotated[
        float, "Notional value needed to hedge (make market neutral)"
    ]
    
    # Final analysis and messaging context
    analysis_results: Annotated[str, "Computed result of the analysis or decision"]
    messages: Annotated[
        List[BaseMessage], "List of chat messages used in LLM prompt construction"
    ]
    decision_prompt: Annotated[str, "decision prompt for reflection"]
    final_trade_decision: Annotated[
        str, "Final BUY or SELL decision made after analyzing indicators"
    ]
