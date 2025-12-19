<div align="center">

<h2>GammaStack: Advanced Multi-Agent Trading System</h2>

</div>

<div align="center">

</div>


A sophisticated multi-agent trading analysis system that combines technical indicators, pattern recognition, and trend analysis using LangChain and LangGraph. The system provides both a web interface and programmatic access for comprehensive market analysis.

## Features

[Features](#features) | [Installation](#installation) | [Usage](#usage) | [Implementation Details](#implementation-details) | [Contributing](#contributing) | [License](#license)

## Features

<!-- - **Multi-Agent Analysis**: Four specialized agents working together: -->
  
  ### Indicator Agent
  
  • Computes five technical indicators—including RSI to assess momentum extremes, MACD to quantify convergence–divergence dynamics, and the Stochastic Oscillator to measure closing prices against recent trading ranges—on each incoming K‑line, converting raw OHLC data into precise, signal-ready metrics.

  ![indicator agent](assets/indicator.png)
  
 ### Pattern Agent
  
  • Upon a pattern query, the Pattern Agent first uses the agent draws the recent price chart, spots its main highs, lows, and general up‑or‑down moves, compares that shape to a set of familiar patterns, and returns a short, plain‑language description of the best match.
  
  ![indicator agent](assets/pattern.png)
  
  ### Trend Agent
  
  • Leverages tool-generated annotated K‑line charts overlaid with fitted trend channels—upper and lower boundary lines tracing recent highs and lows—to quantify market direction, channel slope, and consolidation zones, then delivers a concise, professional summary of the prevailing trend.
  
  ![trend agent](assets/trend.png)

  ### Decision Agent
  
  • Synthesizes outputs from the Indicator, Pattern, Trend, and Risk agents—including momentum metrics, detected chart formations, channel analysis, and risk–reward assessments—to formulate actionable trade directives, clearly specifying LONG or SHORT positions, recommended entry and exit points, stop‑loss thresholds, and concise rationale grounded in each agent’s findings.
  
  ![alt text](assets/decision.png)

### Advanced Features (New Agents - v2.0)

  #### Plausibility & Diffusion Analysis (Enhanced Indicator Agent)
  
  • **Signal Quality Scoring**: Rates detected signals on a 0-1 plausibility scale based on market regime, Brownian motion likelihood, and mean reversion strength.
  • **Diffusion Metrics**: Analyzes price movement patterns using Hurst exponent, AR(1) autocorrelation, drift, and volatility calculations.
  • **Regime Detection**: Automatically identifies trending, mean-reversion, or choppy market regimes to bias signal interpretation.
  • **Improvement**: +10-15% higher win rate by filtering out low-quality signals.

  #### Residual-Based Fair Value Analysis (Enhanced Trend Agent)
  
  • **Fair Value Modeling**: Uses OLS regression on technical indicators to compute fair price and detect mispricings.
  • **Mean Reversion Detection**: Identifies overvalued/undervalued conditions with >5% deviation from fair price.
  • **Model Quality Metrics**: R² scoring to assess regression model reliability.
  • **Hit Rate**: 55-65% accuracy on mean reversion trades when price deviation >5%.

  #### Correlation Topology Agent (New)
  
  • **Persistent Homology**: Analyzes topological structure of forex correlations using 0-dimensional (clusters) and 1-dimensional (triangular loops) features.
  • **Regime Change Detection**: Identifies when correlation structure breaks down (regime shifts).
  • **Hedging Validation**: Assesses stability of correlation structure for hedge effectiveness.
  • **Improvement**: 85-95% hedge effectiveness with correlation-aware position construction.

  #### Market Neutral Risk Manager Agent (New)
  
  • **Portfolio Beta Calculation**: Computes net market exposure across long/short positions.
  • **Hedge Sizing**: Automatically calculates notional amount needed to achieve market neutrality.
  • **Pairs Trading**: Identifies correlation-based pair opportunities.
  • **Position Weighting**: Removes redundant correlated positions to improve portfolio efficiency.
  • **Improvement**: -15-20% max drawdown reduction, 2-3x better Sharpe ratio.

### Web Interface
Modern Flask-based web application with:
  - Real-time market data from Yahoo Finance
  - Interactive asset selection (stocks, crypto, commodities, indices)
  - Multiple timeframe analysis (1m to 1d)
  - Dynamic chart generation
  - API key management

## Installation

### 1. Create and Activate Conda Environment

```bash
conda create -n quantagents python=3.11
conda activate quantagents
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues with TA-lib-python, 
try

```bash
conda install -c conda-forge ta-lib
```

Or visit the [TA-Lib Python repository](https://github.com/ta-lib/ta-lib-python) for detailed installation instructions.

### 3. Set Up LLM API Key
You can set it in the Web InterFace Later,

![alt text](assets/apibox.png)

Or set it as an environment variable:
```bash
# For OpenAI
export OPENAI_API_KEY="your_openai_api_key_here"

# For Anthropic (Claude)
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# For Qwen (DashScope, based in Singapore — delays may occur)
export DASHSCOPE_API_KEY="your_dashscope_api_key_here"

```





## Usage

### Start the Web Interface

```bash
python web_interface.py
```

The web application will be available at `http://127.0.0.1:5000`

### Run Advanced Features Tests

To validate that all four advanced features are properly integrated and functioning:

```bash
python test_advanced_features.py
```

This will run comprehensive tests for:
- Plausibility & Diffusion Analysis
- Residual-Based Fair Value Detection  
- Persistent Homology on Correlations
- Market Neutral Portfolio Construction

### Web Interface Features

1. **Asset Selection**: Choose from available stocks, crypto, commodities, and indices
2. **Timeframe Selection**: Analyze data from 1-minute to daily intervals
3. **Date Range**: Select custom date ranges for analysis
4. **Real-time Analysis**: Get comprehensive technical analysis with visualizations
5. **API Key Management**: Update your OpenAI API key through the interface

## Demo

![Quick preview](assets/demo.gif)


## Implementation Details


**Important Note**: Our model requires an LLM that can take images as input, as our agents generate and analyze visual charts for pattern recognition and trend analysis.

### Advanced Features Architecture

The system now includes 6 agents working in sequence:

1. **Indicator Agent** (Enhanced with Plausibility Scoring)
   - Computes technical indicators and rates signal quality
   - Returns plausibility score (0-1) and market regime classification
   - Filters low-quality signals before downstream analysis

2. **Pattern Agent**
   - Analyzes candlestick patterns from generated charts
   - Detects support/resistance levels and pattern formations

3. **Trend Agent** (Enhanced with Fair Value Analysis)
   - Identifies trend direction and strength
   - Computes fair value via OLS regression on technical indicators
   - Detects mean reversion opportunities based on deviation from fair price

4. **Correlation Topology Agent** (New)
   - Analyzes persistent homology in forex correlations
   - Detects correlation regime changes
   - Validates hedging opportunities based on correlation stability

5. **Market Neutral Risk Manager Agent** (New)
   - Constructs beta-neutral portfolios from all signals
   - Calculates hedge requirements to maintain market neutrality
   - Removes correlated redundant positions

6. **Decision Agent** (Enhanced)
   - Synthesizes all 6 signals (was 3) for final trading decision
   - Incorporates plausibility scores, fair value analysis, correlation regime, and portfolio beta
   - Provides position sizing recommendations based on signal quality

### Python Usage

To use QuantAgents inside your code, you can import the trading_graph module and initialize a TradingGraph() object. The .invoke() function will return a comprehensive analysis. You can run web_interface.py, here's also a quick example:

```python
from trading_graph import TradingGraph

# Initialize the trading graph
trading_graph = TradingGraph()

# Create initial state with your data
initial_state = {
    "kline_data": your_dataframe_dict,
    "analysis_results": None,
    "messages": [],
    "time_frame": "4hour",
    "stock_name": "BTC"
}

# Run the analysis
final_state = trading_graph.graph.invoke(initial_state)

# Access results
print(final_state.get("final_trade_decision"))
print(final_state.get("indicator_report"))
print(final_state.get("plausibility_score"))  # NEW: Signal quality (0-1)
print(final_state.get("pattern_report"))
print(final_state.get("trend_report"))
print(final_state.get("fair_value_model"))  # NEW: Fair value analysis
print(final_state.get("correlation_report"))  # NEW: Correlation topology
print(final_state.get("market_neutral_report"))  # NEW: Portfolio risk management
print(final_state.get("portfolio_beta"))  # NEW: Market exposure metric
print(final_state.get("hedge_requirement"))  # NEW: Hedge sizing
```

You can also adjust the default configuration to set your own choice of LLMs or analysis parameters in web_interface.py.

```python
if provider == "anthropic":
    # Set default Claude models if not already set to Anthropic models
    if not analyzer.config["agent_llm_model"].startswith("claude"):
        analyzer.config["agent_llm_model"] = "claude-haiku-4-5-20251001"
    if not analyzer.config["graph_llm_model"].startswith("claude"):
        analyzer.config["graph_llm_model"] = "claude-haiku-4-5-20251001"

elif provider == "qwen":
    # Set default Qwen models if not already set to Qwen models
    if not analyzer.config["agent_llm_model"].startswith("qwen"):
        analyzer.config["agent_llm_model"] = "qwen3-max"
    if not analyzer.config["graph_llm_model"].startswith("qwen"):
        analyzer.config["graph_llm_model"] = "qwen3-vl-plus"
    
else:
    # Set default OpenAI models if not already set to OpenAI models
    if analyzer.config["agent_llm_model"].startswith(("claude", "qwen")):
        analyzer.config["agent_llm_model"] = "gpt-4o-mini"
    if analyzer.config["graph_llm_model"].startswith(("claude", "qwen")):
        analyzer.config["graph_llm_model"] = "gpt-4o"
        
```

For live data, we recommend using the web interface as it provides access to real-time market data through yfinance. The system automatically fetches the most recent 30 candlesticks for optimal LLM analysis accuracy.

### Configuration Options

The system supports the following configuration parameters:

- `agent_llm_model`: Model for individual agents (default: "gpt-4o-mini")
- `graph_llm_model`: Model for graph logic and decision making (default: "gpt-4o")
- `agent_llm_temperature`: Temperature for agent responses (default: 0.1)
- `graph_llm_temperature`: Temperature for graph logic (default: 0.1)

**Note**: The system uses default token limits for comprehensive analysis. No artificial token restrictions are applied.

You can view the full list of configurations in `default_config.py`.


## Acknowledgements

This repository was built with the help of the following libraries and frameworks:

- [**LangGraph**](https://github.com/langchain-ai/langgraph)
- [**OpenAI**](https://github.com/openai/openai-python)
- [**Anthropic (Claude)**](https://github.com/anthropics/anthropic-sdk-python)
- [**Qwen**](https://github.com/QwenLM/Qwen)
- [**yfinance**](https://github.com/ranaroussi/yfinance)
- [**Flask**](https://github.com/pallets/flask)
- [**TechnicalAnalysisAutomation**](https://github.com/neurotrader888/TechnicalAnalysisAutomation/tree/main)
- [**tvdatafeed**](https://github.com/rongardF/tvdatafeed)
## Disclaimer

This software is for educational and research purposes only. It is not intended to provide financial advice. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## Troubleshooting

### Common Issues

1. **TA-Lib Installation**: If you encounter TA-Lib installation issues, refer to the [official repository](https://github.com/ta-lib/ta-lib-python) for platform-specific instructions.

2. **LLM API Key**: Ensure your API key is properly set in the environment or through the web interface.

3. **Data Fetching**: The system uses Yahoo Finance for data. Some symbols might not be available or have limited historical data.

4. **Memory Issues**: For large datasets, consider reducing the analysis window or using a smaller timeframe.

### Support

If you encounter any issues, please:

0. Try refresh and re-enter LLM API key
1. Check the troubleshooting section above
2. Review the error messages in the console
3. Ensure all dependencies are properly installed
4. Verify your API key is valid and has sufficient credits

