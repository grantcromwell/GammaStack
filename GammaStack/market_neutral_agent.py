"""
Market Neutral Risk Manager Agent - Constructs and manages market-neutral portfolios

Balances long and short positions, calculates hedge requirements, and maintains 
beta-neutral (market-neutral) portfolio construction.
"""

import json
import numpy as np
from typing import Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from advanced_features import MarketNeutralRiskManager


def create_market_neutral_agent(llm):
    """
    Create a market neutral risk manager agent that constructs beta-neutral portfolios.
    
    Process:
    1. Collect all LONG/SHORT signals from previous agents
    2. Balance long and short notional values
    3. Calculate portfolio beta and hedge requirement
    4. Identify pairs trading opportunities
    5. Generate portfolio construction recommendations
    """
    
    def market_neutral_agent_node(state):
        """
        Construct market-neutral portfolio from all signals.
        
        Inputs:
        - final_trade_decision: Current signal
        - correlation_topology: Correlation structure
        - fair_value_model: Fair value analysis
        
        Outputs:
        - market_neutral_signals: Portfolio construction
        - portfolio_beta: Market exposure
        - hedge_requirement: Notional to hedge
        """
        
        # Gather signals from state
        signals = []
        
        # Extract current trade signal if available
        if state.get("final_trade_decision"):
            try:
                decision_text = state["final_trade_decision"]
                
                # Parse decision text (may contain JSON or natural language)
                signal_direction = 'LONG' if 'LONG' in decision_text.upper() else (
                    'SHORT' if 'SHORT' in decision_text.upper() else 'SKIP'
                )
                
                # Try to extract confidence level
                confidence = 0.5
                if 'confidence' in decision_text.lower():
                    # Simple extraction - in production, use JSON parsing
                    parts = decision_text.lower().split('confidence')
                    if len(parts) > 1:
                        try:
                            conf_str = parts[1].split()[0].replace(':', '').replace('"', '')
                            confidence = float(conf_str)
                        except:
                            confidence = state.get("plausibility_score", 0.5)
                
                if signal_direction != 'SKIP':
                    signals.append({
                        'asset': state.get("stock_name", "PRIMARY"),
                        'signal_direction': signal_direction,
                        'confidence': min(max(confidence, 0), 1),  # Clamp 0-1
                        'fair_value_deviation': state.get("fair_value_model", {}).get("deviation_pct", 0),
                    })
            except Exception as e:
                print(f"Warning: Failed to extract signal from decision: {e}")
        
        # Build correlations from correlation agent if available
        correlations = {}
        try:
            if state.get("correlation_topology"):
                topology = state["correlation_topology"]
                # In production, this would include actual correlation matrix
                # For now, use a simple stub
                correlations = {
                    'stability': topology.get('stability_score', 0.5),
                    'regime_changed': topology.get('regime_changed', False),
                }
        except Exception as e:
            print(f"Warning: Failed to extract correlations: {e}")
        
        # Construct market-neutral portfolio
        portfolio = {
            'long_notional': 0,
            'short_notional': 0,
            'net_notional': 0,
            'positions': [],
            'portfolio_metrics': {
                'net_beta': 0.0,
                'is_market_neutral': True,
            },
            'hedge_requirement': 0,
            'pairs_trading_opportunities': [],
            'recommendation': 'Hold - waiting for clear signals'
        }
        
        try:
            if signals:
                manager = MarketNeutralRiskManager()
                
                # Construct portfolio
                portfolio = manager.construct_market_neutral_portfolio(
                    signals=signals,
                    correlation_matrix=correlations,
                    portfolio_value=1000000,
                    target_beta=0.0
                )
        except Exception as e:
            print(f"Warning: Portfolio construction failed: {e}")
            # Provide fallback structure
            if signals:
                for signal in signals:
                    if signal['signal_direction'] == 'LONG':
                        portfolio['long_notional'] += 500000
                        portfolio['positions'].append({
                            'asset': signal['asset'],
                            'direction': 'LONG',
                            'notional': 500000,
                            'weight': 0.5
                        })
                    else:
                        portfolio['short_notional'] += 500000
                        portfolio['positions'].append({
                            'asset': signal['asset'],
                            'direction': 'SHORT',
                            'notional': 500000,
                            'weight': 0.5
                        })
                
                portfolio['net_notional'] = portfolio['long_notional'] - portfolio['short_notional']
                portfolio['portfolio_metrics']['net_beta'] = portfolio['net_notional'] / 1000000
                portfolio['portfolio_metrics']['is_market_neutral'] = abs(portfolio['net_notional']) < 50000
                portfolio['hedge_requirement'] = max(0, abs(portfolio['net_notional']))
        
        # Create LLM prompt for portfolio management
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are a portfolio manager specializing in market-neutral strategies and risk management.

Your role: Analyze the current portfolio construction and provide hedging/management recommendations.

CURRENT PORTFOLIO STATE:
- Long Notional: ${portfolio['long_notional']:,.0f}
- Short Notional: ${portfolio['short_notional']:,.0f}
- Net Notional: ${portfolio['net_notional']:,.0f}
- Portfolio Beta: {portfolio['portfolio_metrics']['net_beta']:.3f}
- Market Neutral Status: {'✓ NEUTRAL (beta < 0.05)' if abs(portfolio['portfolio_metrics']['net_beta']) < 0.05 else f'✗ SKEWED (beta = {portfolio['portfolio_metrics']['net_beta']:.3f})'}
- Hedge Requirement: ${portfolio['hedge_requirement']:,.0f}
- Correlation Regime Stable: {not state.get('correlation_regime_change', {}).get('regime_changed', True)}

RECOMMENDATIONS:
1. {'✓ Portfolio is market neutral' if portfolio['portfolio_metrics']['is_market_neutral'] else f'⚠️ Execute hedge of ${portfolio['hedge_requirement']:,.0f}'} to achieve beta neutrality
2. {'✓ No rebalancing needed' if abs(portfolio['long_notional'] - portfolio['short_notional']) < 100000 else '⚠️ Rebalance long/short exposure'} 
3. Monitor correlation stability: {'✓ Stable' if not state.get('correlation_regime_change', {}).get('regime_changed', True) else '⚠️ Unstable - consider reducing leverage'}
4. Mean reversion opportunity: {f"Price is {state.get('fair_value_model', {}).get('deviation_pct', 0):.1f}% from fair value" if state.get('fair_value_model', {}).get('deviation_pct', 0) else 'None detected'}

Provide actionable guidance for portfolio management and risk control.""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Prepare messages
        messages = state.get("messages", [])
        if not messages:
            messages = [
                HumanMessage(
                    content="Analyze portfolio construction and provide hedging recommendations."
                )
            ]
        
        # Invoke LLM
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        messages.append(response)
        
        # Extract response content
        market_neutral_report = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "messages": messages,
            "market_neutral_report": market_neutral_report,
            "market_neutral_signals": portfolio,
            "portfolio_beta": portfolio['portfolio_metrics']['net_beta'],
            "hedge_requirement": portfolio['hedge_requirement'],
        }
    
    return market_neutral_agent_node
