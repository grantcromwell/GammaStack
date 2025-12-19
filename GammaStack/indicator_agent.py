"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
Enhanced with Plausibility & Diffusion Analysis.
"""

import copy
import json
import numpy as np

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from advanced_features import PlausibilityAnalyzer


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        # --- Tool definitions ---
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        time_frame = state["time_frame"]
        
        # --- NEW: Compute plausibility and diffusion metrics ---
        plausibility_score = 0.5  # Default
        regime_type = "unknown"
        diffusion_info = {}
        
        try:
            if state.get("kline_data") and "Close" in state["kline_data"]:
                prices = np.array(state["kline_data"]["Close"][-50:])
                if len(prices) > 10:
                    analyzer = PlausibilityAnalyzer()
                    diffusion = analyzer.compute_diffusion_metrics(prices, lookback=50)
                    
                    # Score plausibility (0-1)
                    plausibility_score = analyzer.score_signal_plausibility(
                        state["kline_data"], 
                        'LONG',  # Will be updated based on actual signal
                        prices[-1]
                    )
                    
                    # Determine regime
                    if diffusion.brownian_likelihood > 0.6:
                        regime_type = "trending"
                    elif diffusion.mean_reversion_strength > 0.6:
                        regime_type = "mean_reversion"
                    else:
                        regime_type = "choppy"
                    
                    diffusion_info = {
                        "brownian_likelihood": diffusion.brownian_likelihood,
                        "mean_reversion_strength": diffusion.mean_reversion_strength,
                        "drift": diffusion.drift_component,
                        "volatility": diffusion.diffusion_speed,
                        "confidence": diffusion.confidence
                    }
        except Exception as e:
            print(f"Warning: Plausibility calculation failed: {e}")
        
        # --- System prompt for LLM (ENHANCED) ---
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a high-frequency trading (HFT) analyst assistant operating under time-sensitive conditions. 
You must analyze technical indicators to support fast-paced trading execution.

You have access to tools: compute_rsi, compute_macd, compute_roc, compute_stoch, and compute_willr. 
Use them by providing appropriate arguments like `kline_data` and the respective periods.

⚠️ The OHLC data provided is from a {time_frame} intervals, reflecting recent market behavior. 
You must interpret this data quickly and accurately.

NEW FEATURE - SIGNAL PLAUSIBILITY ANALYSIS:
- Current Plausibility Score: {plausibility_score:.2f} (0-1 scale)
- Market Regime: {regime_type}
- Diffusion Info: {json.dumps(diffusion_info, indent=2)}

GUIDANCE:
- High plausibility (>0.75): Strong signal with good entry quality
- Medium plausibility (0.60-0.75): Acceptable but monitor for confirmation
- Low plausibility (<0.60): Weak signal, consider skipping or requiring extra confirmation

Prioritize indicators that align with the detected market regime.

Here is the OHLC data:
{json.dumps(state["kline_data"], indent=2)}

Call necessary tools, and analyze the results. Rate the plausibility of any detected signals.""",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | llm.bind_tools(tools)
        # messages = state["messages"]
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content="Begin indicator analysis.")]


        # --- Step 1: Ask for tool calls ---
        ai_response = chain.invoke(messages)
        messages.append(ai_response)
        
        # --- Step 2: Collect tool results ---
        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for call in ai_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                # Always provide kline_data
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                # Lookup tool by name
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                # Append result as ToolMessage
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )

        # --- Step 3: Re-run the chain with tool results ---
        # Keep invoking until we get a text response (not another tool call)
        # This is important for Claude which may make multiple tool calls
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        final_response = None
        
        while iteration < max_iterations:
            iteration += 1
            final_response = chain.invoke(messages)
            messages.append(final_response)
            
            # If there are no tool calls, we have the final answer
            if not hasattr(final_response, "tool_calls") or not final_response.tool_calls:
                break
            
            # If there are more tool calls, execute them
            for call in final_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )

        # Extract content - handle both string and empty content cases
        if final_response:
            report_content = final_response.content
            # If content is empty or None, try to get text from recent messages
            if not report_content or (isinstance(report_content, str) and not report_content.strip()):
                # Check if there's any text content in the messages (skip tool calls)
                for msg in reversed(messages):
                    if (hasattr(msg, 'content') and msg.content and 
                        isinstance(msg.content, str) and msg.content.strip() and 
                        not hasattr(msg, 'tool_calls')):
                        report_content = msg.content
                        break
        else:
            report_content = "Indicator analysis completed, but no detailed report was generated."

        return {
            "messages": messages,
            "indicator_report": report_content if report_content else "Indicator analysis completed.",
            "plausibility_score": plausibility_score,
            "signal_regime": regime_type,
            "diffusion_metrics": diffusion_info,
        }

    return indicator_agent_node
