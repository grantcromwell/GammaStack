"""
Correlation Topology Agent - Analyzes persistent homology in forex correlations

Detects correlation regime changes, identifies hedging opportunities, and assesses 
correlation stability for portfolio construction.
"""

import json
import numpy as np
from typing import Dict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from advanced_features import CorrelationTopology


def create_correlation_topology_agent(llm):
    """
    Create a correlation topology agent that analyzes forex pair correlations 
    using persistent homology.
    
    Detects:
    - Correlation clusters (groups of synchronized pairs)
    - Triangular structures (cyclical patterns)
    - Regime changes (when correlations break down)
    - Hedging opportunities
    """
    
    def correlation_agent_node(state):
        """
        Analyze correlation structure between major forex pairs.
        
        Input: Returns of major pairs (from indicator data or forex feeds)
        Output: Correlation clusters, regime changes, hedging signals
        """
        
        # Build returns data from available state
        # In production, this would fetch real forex data
        # For now, we derive synthetic correlations from indicator states
        forex_returns = {}
        
        try:
            # Try to extract or compute returns
            if state.get("kline_data") and "Close" in state["kline_data"]:
                closes = np.array(state["kline_data"]["Close"][-60:])
                if len(closes) > 1:
                    # Simple returns: log(P_t / P_{t-1})
                    returns = np.diff(np.log(closes))
                    
                    # Create synthetic forex pair returns (in production, fetch real data)
                    forex_returns = {
                        'EURUSD': returns,
                        'GBPUSD': returns + np.random.randn(len(returns)) * 0.005,
                        'JPYUSD': returns - np.random.randn(len(returns)) * 0.003,
                        'AUDUSD': returns + np.random.randn(len(returns)) * 0.007,
                        'CHFUSD': returns - np.random.randn(len(returns)) * 0.004,
                    }
        except Exception as e:
            print(f"Warning: Forex returns extraction failed: {e}")
            # Provide empty but valid return structure
            forex_returns = {}
        
        # Analyze correlation topology
        correlation_topology = {
            'num_clusters': 0,
            'num_loops': 0,
            'stability_score': 0.5,
            'regime_type': 'stable',
            'regime_changed': False,
            'hedging_pairs': [],
            'description': 'Insufficient data for correlation analysis'
        }
        
        try:
            if len(forex_returns) > 2:
                topology = CorrelationTopology()
                
                # Compute correlation structure
                corr_matrix, pairs = topology.compute_correlation_matrix(forex_returns)
                
                # Analyze persistent homology
                homology = topology.compute_persistent_homology(corr_matrix, pairs)
                
                # Detect regime changes
                regime_change = topology.detect_correlation_regime_changes(
                    corr_matrix, 
                    lookback=20
                )
                
                correlation_topology = {
                    'num_clusters': homology.get('num_clusters', 0),
                    'num_loops': homology.get('num_loops', 0),
                    'stability_score': homology.get('stability_score', 0.5),
                    'regime_type': homology.get('regime_type', 'stable'),
                    'regime_changed': regime_change.get('regime_changed', False),
                    'hedging_pairs': regime_change.get('hedging_pairs', []),
                    'description': f"Detected {homology.get('num_clusters', 0)} clusters, "
                                  f"{homology.get('num_loops', 0)} triangular structures, "
                                  f"stability: {homology.get('stability_score', 0.5):.2f}"
                }
        except Exception as e:
            print(f"Warning: Persistent homology calculation failed: {e}")
        
        # Create LLM prompt for interpretation
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are an expert in foreign exchange and correlation analysis.

Your role: Interpret persistent homology results and forex correlation regime.

CONTEXT:
Persistent homology reveals the topological structure of correlations:
- 0-dimensional features (H0): Clusters of strongly correlated pairs
- 1-dimensional features (H1): Triangular structures (cyclical patterns)
- Stability: How robust is the correlation structure (0-1 scale)

CURRENT ANALYSIS RESULTS:
- Number of Correlation Clusters: {correlation_topology['num_clusters']}
- Number of Triangular Structures: {correlation_topology['num_loops']}
- Correlation Stability Score: {correlation_topology['stability_score']:.2f}
- Regime Type: {correlation_topology['regime_type']}
- Regime Changed: {correlation_topology['regime_changed']}
- Suggested Hedging Pairs: {', '.join(correlation_topology['hedging_pairs']) if correlation_topology['hedging_pairs'] else 'None'}

KEY INSIGHTS TO PROVIDE:
1. Are major pairs clustered or fragmented?
2. Is correlation regime stable or changing?
3. What are the hedging implications for this trade?
4. Are there obvious arbitrage or diversification opportunities?

Be concise and actionable in your analysis.""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Prepare messages
        messages = state.get("messages", [])
        if not messages:
            messages = [
                HumanMessage(
                    content="Analyze the correlation topology and provide insights on hedging and regime changes."
                )
            ]
        
        # Invoke LLM
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        messages.append(response)
        
        # Extract response content
        correlation_report = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "messages": messages,
            "correlation_report": correlation_report,
            "correlation_topology": correlation_topology,
            "correlation_regime_change": {
                'regime_changed': correlation_topology['regime_changed'],
                'regime_type': correlation_topology['regime_type'],
                'stability': correlation_topology['stability_score'],
            }
        }
    
    return correlation_agent_node
