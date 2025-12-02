"""
INTEGRATION GUIDE: Advanced Features into GammaStack Trading System

This document shows how to integrate the four new advanced features into your existing
trading graph and agents.
"""

# ============================================================================
# 1. INDICATOR AGENT INTEGRATION - Plausibility & Diffusion Scoring
# ============================================================================

# File: indicator_agent.py
# Add at the top with other imports:

from advanced_features import PlausibilityAnalyzer, DiffusionMetrics
import numpy as np

# Modify the create_indicator_agent function:

def create_indicator_agent(llm, toolkit):
    """
    Enhanced indicator agent with plausibility and diffusion analysis.
    """
    
    def indicator_agent_node(state):
        # ... existing indicator computation code ...
        
        # NEW: Add plausibility analysis
        messages = state.get("messages", [])
        
        # Run standard indicators first (existing code)
        # ... (your existing indicator agent logic) ...
        
        # AFTER computing indicators, add plausibility check
        if state.get("kline_data"):
            analyzer = PlausibilityAnalyzer()
            
            # Compute diffusion metrics
            prices = state["kline_data"]["Close"][-50:]
            diffusion = analyzer.compute_diffusion_metrics(
                np.array(prices),
                lookback=50,
                forecast_periods=20
            )
            
            # Score the regime and pattern
            regime_score = analyzer.score_signal_plausibility(
                kline_data=state["kline_data"],
                signal="LONG",  # Or SHORT depending on indicators
                entry_price=state["kline_data"]["Close"][-1],
                indicator_confidence=0.6  # Will be updated by LLM
            )
            
            # Add to state
            state["plausibility_score"] = regime_score["plausibility_score"]
            state["diffusion_metrics"] = regime_score["diffusion_metrics"]
            state["signal_regime"] = regime_score["regime"]
        
        # Update system prompt to emphasize plausibility
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a technical indicator expert in trading. You analyze price action using technical indicators.

CRITICAL NEW REQUIREMENTS:

1. PLAUSIBILITY SCORING:
   Rate how plausible your signal is:
   - High plausibility (>0.75): Strong regime alignment + good entry quality
   - Medium plausibility (0.60-0.75): Acceptable but some concerns
   - Low plausibility (<0.60): Weak setup, suggest skipping
   
   Factors to evaluate:
   - Is price in trending or ranging regime?
   - Is entry near support (long) or resistance (short)?
   - Is volatility normal or extreme?
   
2. DIFFUSION ANALYSIS:
   Analyze how price moves:
   - If Hurst exponent >0.5: Price is trending (momentum)
   - If Hurst exponent <0.5: Price is mean reverting
   - If AR(1) coefficient positive: Price continues previous direction
   - If AR(1) coefficient negative: Price likely to reverse
   
   Use this to confirm signal validity.

SIGNAL FORMAT:
Return your analysis as JSON:
{{
    "signal": "LONG|SHORT|SKIP",
    "plausibility": 0.0-1.0,
    "confidence": 0.0-1.0,
    "regime": "trending|mean_reversion|choppy",
    "entry_reasoning": "...",
    "plausibility_reasoning": "Why is this setup plausible/implausible?"
}}
""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # ... rest of agent logic ...
        return state


# ============================================================================
# 2. TREND AGENT ENHANCEMENT - Residual Analysis for Fair Value
# ============================================================================

# File: trend_agent.py
# Add integration for residual-based mean reversion:

from advanced_features import ResidualAnalyzer

def create_trend_agent_enhanced(llm, toolkit):
    """
    Enhanced trend agent with fair value and mispricing detection.
    """
    
    def trend_agent_node(state):
        # ... existing trend analysis code ...
        
        # NEW: Analyze residuals for mean reversion setups
        analyzer = ResidualAnalyzer()
        
        # Build predictor data from indicators already computed
        predictors = {
            'rsi': state.get('rsi', []),
            'macd': state.get('macd', []),
            'volume_ratio': np.array(state.get('volume', []) / np.mean(state.get('volume', [1])) if state.get('volume') else [1])
        }
        
        model = analyzer.build_fair_value_model(
            state['kline_data'],
            predictors,
            lookback=100
        )
        
        current_price = state['kline_data']['Close'][-1]
        fair_price = model['fair_prices'][-1]
        deviation = model['current_discount_pct']
        
        # Store results
        state['fair_value_model'] = {
            'fair_price': fair_price,
            'current_price': current_price,
            'deviation_pct': deviation,
            'r_squared': model['r_squared'],
            'recommendation': 'Mean revert to fair' if abs(deviation) > 5 else 'Fair valued'
        }
        
        # Update prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are a trend analysis expert. Analyze price trends and support/resistance.

NEW FEATURE - FAIR VALUE ANALYSIS:

The price is currently:
- Fair Price (regression model): ${fair_price:.2f}
- Current Price: ${current_price:.2f}
- Deviation: {deviation:.2f}%

Interpretation:
- Deviation > +5%: OVERVALUED (consider SHORT/sell)
- Deviation < -5%: UNDERVALUED (consider LONG/buy)
- |Deviation| < 5%: FAIR VALUED (normal range)

Model quality: R² = {model['r_squared']:.2f} (>0.6 is good)

Use this to bias your trend analysis towards mean reversion setups.
""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # ... rest of agent logic ...
        return state


# ============================================================================
# 3. NEW AGENT: CORRELATION TOPOLOGY AGENT
# ============================================================================

# File: correlation_agent.py (CREATE NEW FILE)

from advanced_features import CorrelationTopology
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_correlation_topology_agent(llm):
    """
    New agent that analyzes forex pair correlations using persistent homology.
    Detects correlation regime changes and identifies hedging opportunities.
    """
    
    def correlation_agent_node(state):
        """
        Analyze correlation structure between major forex pairs.
        
        Input: Returns of major pairs (EURUSD, GBPUSD, JPYUSD, etc.)
        Output: Correlation clusters, regime changes, hedging signals
        """
        
        # Build returns data (example - in production fetch real forex data)
        # For now, derive from indicator data as proxy
        returns_dict = {}
        
        # Placeholder: In production, fetch real forex data
        # This is simplified for integration
        
        topology = CorrelationTopology()
        
        # Analyze correlation structure
        try:
            forex_pairs = ['EURUSD', 'GBPUSD', 'NZDUSD', 'AUDUSD']
            
            # Create synthetic returns from price data for demo
            # In production: fetch from exchange API
            returns_dict = {
                pair: np.random.randn(60)  # Placeholder
                for pair in forex_pairs
            }
            
            # Compute persistent homology
            homology_results = topology.compute_persistent_homology(
                *topology.compute_correlation_matrix(returns_dict),
                min_correlation=0.4
            )
            
            # Detect regime changes
            regime_change = topology.detect_correlation_regime_change(
                returns_dict,
                lookback=60
            )
            
            # Store in state
            state['correlation_topology'] = homology_results
            state['correlation_regime_change'] = regime_change
            
        except Exception as e:
            # Fallback if forex data unavailable
            state['correlation_topology'] = {
                'error': str(e),
                'correlation_clusters': [],
                'stability_score': 0.5
            }
        
        # Create prompt for LLM interpretation
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert in foreign exchange and correlation analysis.

Your role: Interpret persistent homology results and forex correlation regime.

CONTEXT:
Persistent homology reveals the topological structure of correlations:
- 0-dimensional features (H0): Clusters of strongly correlated pairs
- 1-dimensional features (H1): Triangular structures (cyclical patterns)
- Stability: How robust is the correlation structure

KEY INSIGHTS TO PROVIDE:
1. Are major pairs clustered or fragmented?
2. Is correlation regime stable or changing?
3. What are the hedging implications?
4. Are there arbitrage opportunities?

Be concise and actionable.
""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        messages = state.get("messages", [])
        if not messages:
            messages = [
                HumanMessage(
                    content=f"""Analyze this correlation structure:

Correlation Clusters: {homology_results['correlation_clusters']}
Stability Score: {homology_results['stability_score']:.2f}
Regime Change Detected: {regime_change['regime_changed']}
Change Magnitude: {regime_change['change_magnitude']:.2f}

Provide trading implications."""
                )
            ]
        
        chain = prompt | llm
        response = chain.invoke({'messages': messages})
        messages.append(response)
        
        state['correlation_report'] = response.content if hasattr(response, 'content') else str(response)
        state['messages'] = messages
        
        return state
    
    return correlation_agent_node


# ============================================================================
# 4. NEW AGENT: MARKET NEUTRAL RISK MANAGER
# ============================================================================

# File: market_neutral_agent.py (CREATE NEW FILE)

from advanced_features import MarketNeutralRiskManager

def create_market_neutral_agent(llm):
    """
    New agent that constructs market-neutral portfolios and hedges.
    
    Takes all trading signals and combines them into a beta-neutral portfolio.
    """
    
    def market_neutral_agent_node(state):
        """
        Construct market-neutral portfolio from all signals.
        
        Process:
        1. Collect all LONG/SHORT signals
        2. Balance long and short notional
        3. Calculate hedge requirement
        4. Identify pairs trading opportunities
        """
        
        manager = MarketNeutralRiskManager()
        
        # Gather signals from state
        signals = []
        if state.get('final_trade_decision'):
            # Parse decision into signal
            decision = state['final_trade_decision']
            signal_type = 'LONG' if 'LONG' in decision else 'SHORT'
            
            signals.append({
                'asset': state.get('stock_name', 'Unknown'),
                'signal_direction': signal_type,
                'confidence': state.get('plausibility_score', 0.5)
            })
        
        # Build correlations from correlation agent
        correlations = {}
        if state.get('correlation_topology'):
            # Simplified correlation matrix
            topology = state['correlation_topology']
            # Extract pair correlations from clusters
            # (In production, full correlation matrix)
        
        # Construct neutral portfolio
        portfolio = manager.construct_market_neutral_portfolio(
            signals=signals,
            correlation_matrix=correlations,
            portfolio_value=1000000,
            target_beta=0.0
        )
        
        state['market_neutral_signals'] = portfolio
        state['portfolio_beta'] = portfolio['portfolio_metrics']['net_beta']
        state['hedge_requirement'] = portfolio['hedge_requirement']
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a portfolio manager specializing in market-neutral strategies.

Your role: Manage hedging and construct beta-neutral portfolios.

CURRENT PORTFOLIO STATE:
- Long Notional: ${:.0f}
- Short Notional: ${:.0f}
- Net Notional: ${:.0f}
- Portfolio Beta: {:.3f}
- Hedge Requirement: ${:.0f}

Market Neutral Status: {'✓ NEUTRAL' if abs(portfolio['portfolio_metrics']['net_beta']) < 0.05 else '✗ NOT NEUTRAL'}

Recommendations:
1. Execute hedge if beta > 0.05
2. Balance long/short if net notional > 5% of portfolio
3. Monitor correlation risk
""".format(
                portfolio['long_notional'],
                portfolio['short_notional'],
                portfolio['net_notional'],
                portfolio['portfolio_metrics']['net_beta'],
                portfolio['hedge_requirement']
            ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        messages = state.get("messages", [])
        if not messages:
            messages = [
                HumanMessage(
                    content="Advise on portfolio hedging and market neutrality."
                )
            ]
        
        chain = prompt | llm
        response = chain.invoke({'messages': messages})
        messages.append(response)
        
        state['market_neutral_report'] = response.content if hasattr(response, 'content') else str(response)
        state['messages'] = messages
        
        return state
    
    return market_neutral_agent_node


# ============================================================================
# 5. UPDATE GRAPH_SETUP.PY TO INTEGRATE NEW AGENTS
# ============================================================================

# File: graph_setup.py
# Modify the set_graph method:

from correlation_agent import create_correlation_topology_agent
from market_neutral_agent import create_market_neutral_agent

def set_graph(self):
    """Build the trading analysis graph with new agents"""
    
    graph = StateGraph(IndicatorAgentState)
    
    # Original agents
    agent_nodes = {
        "indicator": create_indicator_agent(self.agent_llm, self.toolkit),
        "pattern": create_pattern_agent(self.agent_llm, self.toolkit),
        "trend": create_trend_agent_enhanced(self.agent_llm, self.toolkit),
        
        # NEW AGENTS
        "correlation": create_correlation_topology_agent(self.agent_llm),
        "market_neutral": create_market_neutral_agent(self.agent_llm),
        
        # Final decision maker
        "decision": create_decision_agent(self.agent_llm, self.toolkit)
    }
    
    # Add nodes to graph
    for agent_name, agent_func in agent_nodes.items():
        graph.add_node(agent_name, agent_func)
    
    # Define execution order
    # Original: Indicator → Pattern → Trend → Decision
    # New: Indicator → Pattern → Trend → Correlation → Market Neutral → Decision
    
    graph.add_edge("START", "indicator")
    graph.add_edge("indicator", "pattern")
    graph.add_edge("pattern", "trend")
    graph.add_edge("trend", "correlation")          # NEW
    graph.add_edge("correlation", "market_neutral") # NEW
    graph.add_edge("market_neutral", "decision")
    graph.add_edge("decision", "END")
    
    self.graph = graph.compile()


# ============================================================================
# 6. UPDATE DECISION_AGENT.PY TO USE NEW SIGNALS
# ============================================================================

# File: decision_agent.py
# Modify the create_decision_agent function:

def create_decision_agent(llm, toolkit):
    """Enhanced decision maker with advanced signal integration"""
    
    def trade_decision_node(state):
        
        # Collect advanced signals
        advanced_signals = {
            'plausibility': state.get('plausibility_score', 0.5),
            'fair_value_deviation': state.get('fair_value_model', {}).get('deviation_pct', 0),
            'correlation_regime_stable': not state.get('correlation_regime_change', {}).get('regime_changed', True),
            'portfolio_beta': state.get('portfolio_beta', 1.0),
            'hedge_available': state.get('hedge_requirement', 0) > 0
        }
        
        # Build decision prompt incorporating all new analyses
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are the final decision maker in a multi-agent trading system.

INTEGRATE ALL SIGNALS:

1. INDICATOR SIGNAL (Technical)
   Confidence: {{confidence}}

2. PATTERN SIGNAL (Chart Patterns)
   Type: {{pattern_type}}

3. TREND SIGNAL (Structure)
   Type: {{trend_type}}
   Fair Value Deviation: {advanced_signals['fair_value_deviation']:.2f}%

4. PLAUSIBILITY SCORE
   Score: {advanced_signals['plausibility']:.2f}
   Interpretation: High plausibility = better risk/reward

5. CORRELATION ANALYSIS
   Regime Stable: {advanced_signals['correlation_regime_stable']}
   Implication: Correlations {'are' if advanced_signals['correlation_regime_stable'] else 'are NOT'} stable

6. PORTFOLIO HEDGING
   Current Beta: {advanced_signals['portfolio_beta']:.3f}
   Status: {'Market neutral' if abs(advanced_signals['portfolio_beta']) < 0.05 else 'Requires hedging'}

DECISION FRAMEWORK:
- HIGH CONFIDENCE: All signals aligned + plausibility > 0.75 + correlations stable
- MEDIUM CONFIDENCE: 2/3 signals aligned + plausibility > 0.60
- LOW CONFIDENCE: Signals conflict OR plausibility < 0.60
- SKIP: Correlation regime shift OR portfolio unhedged

OUTPUT REQUIRED:
1. Final signal: LONG / SHORT / SKIP
2. Confidence level: 0-1
3. Position size: relative to normal (0.5x to 1.5x)
4. Rationale: integrate all 6 signal types
5. Risk management: stop loss, take profit
""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # ... execute prompt and return decision ...
        
        return state
    
    return trade_decision_node


# ============================================================================
# 7. TESTING THE INTEGRATION
# ============================================================================

def test_advanced_features():
    """Test script for new features"""
    
    from advanced_features import (
        PlausibilityAnalyzer,
        ResidualAnalyzer,
        CorrelationTopology,
        MarketNeutralRiskManager
    )
    
    # Test 1: Plausibility scoring
    print("=" * 60)
    print("TEST 1: PLAUSIBILITY & DIFFUSION")
    print("=" * 60)
    
    sample_prices = np.random.cumsum(np.random.randn(100)) + 100
    analyzer = PlausibilityAnalyzer()
    
    diffusion = analyzer.compute_diffusion_metrics(sample_prices)
    print(f"Brownian Likelihood: {diffusion.brownian_likelihood:.2f}")
    print(f"Mean Reversion Strength: {diffusion.mean_reversion_strength:.2f}")
    print(f"Drift: {diffusion.drift_component:.4f}")
    print(f"Volatility: {diffusion.diffusion_speed:.4f}")
    
    # Test 2: Residual analysis
    print("\n" + "=" * 60)
    print("TEST 2: RESIDUAL-BASED FAIR VALUE")
    print("=" * 60)
    
    sample_data = {
        'Close': sample_prices,
        'High': sample_prices * 1.02,
        'Low': sample_prices * 0.98,
        'Open': sample_prices[:-1].tolist() + [sample_prices[-1]],
        'Volume': np.random.randint(1000, 5000, 100)
    }
    
    predictors = {
        'rsi': np.random.rand(100) * 100,
        'macd': np.random.randn(100)
    }
    
    residual_analyzer = ResidualAnalyzer()
    model = residual_analyzer.build_fair_value_model(sample_data, predictors)
    
    print(f"Fair Value Model R²: {model['r_squared']:.3f}")
    print(f"Current Deviation: {model['current_discount_pct']:.2f}%")
    print(f"Current Residual (std): {model['current_residual'] / model['residual_std']:.2f}σ")
    
    # Test 3: Correlation topology
    print("\n" + "=" * 60)
    print("TEST 3: PERSISTENT HOMOLOGY")
    print("=" * 60)
    
    forex_returns = {
        'EURUSD': np.random.randn(60),
        'GBPUSD': np.random.randn(60),
        'JPYUSD': np.random.randn(60),
        'AUDUSD': np.random.randn(60)
    }
    
    topology = CorrelationTopology()
    corr_matrix, pairs = topology.compute_correlation_matrix(forex_returns)
    homology = topology.compute_persistent_homology(corr_matrix, pairs)
    
    print(f"Number of Correlation Clusters: {homology['num_clusters']}")
    print(f"Number of Triangular Structures: {homology['num_loops']}")
    print(f"Correlation Stability: {homology['stability_score']:.2f}")
    print(f"Regime Type: {homology['regime_type']}")
    
    # Test 4: Market neutral portfolio
    print("\n" + "=" * 60)
    print("TEST 4: MARKET NEUTRAL PORTFOLIO")
    print("=" * 60)
    
    signals = [
        {'asset': 'SPX', 'signal_direction': 'LONG', 'confidence': 0.8},
        {'asset': 'QQQ', 'signal_direction': 'LONG', 'confidence': 0.7},
        {'asset': 'IWM', 'signal_direction': 'SHORT', 'confidence': 0.6}
    ]
    
    mn_manager = MarketNeutralRiskManager()
    portfolio = mn_manager.construct_market_neutral_portfolio(
        signals, {}, portfolio_value=1000000
    )
    
    print(f"Long Notional: ${portfolio['long_notional']:,.0f}")
    print(f"Short Notional: ${portfolio['short_notional']:,.0f}")
    print(f"Net Notional: ${portfolio['net_notional']:,.0f}")
    print(f"Portfolio Beta: {portfolio['portfolio_metrics']['net_beta']:.3f}")
    print(f"Is Market Neutral: {portfolio['portfolio_metrics']['is_market_neutral']}")
    print(f"Hedge Required: ${portfolio['hedge_requirement']:,.0f}")


if __name__ == "__main__":
    test_advanced_features()
