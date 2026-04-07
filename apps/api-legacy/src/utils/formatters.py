def format_prediction_result(predictions: list[dict] | dict) -> str:
    """
    Format prediction result as structured text.

    Args:
        predictions: List of prediction candidate dicts, or a single prediction dict

    Returns:
        Formatted string
    """
    # Backward compatibility: wrap single dict in list
    if isinstance(predictions, dict):
        predictions = [predictions]

    lines = ["=" * 50]
    for i, result in enumerate(predictions, 1):
        lines.append(f"   [{i}순위]")
        lines.append(f"   Predicted Code: {result.get('predicted_code', 'N/A')}")
        lines.append(f"   Predicted PPS Code: {result.get('predicted_pps_code', 'N/A')}")
        lines.append(f"   Confidence: {result.get('confidence', 'N/A')}")
        lines.append(f"   Reasoning: {result.get('reasoning', 'N/A')}")
        if i < len(predictions):
            lines.append("-" * 30)
    lines.append("=" * 50)
    return "\n".join(lines)
