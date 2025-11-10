def format_prediction_result(result: dict) -> str:
    """
    예측 결과를 구조화된 텍스트로 포맷

    Args:
        result: 예측 결과 딕셔너리

    Returns:
        포맷된 문자열
    """
    lines = [
        "=" * 50,
        f"   Predicted Code: {result.get('predicted_code', 'N/A')}",
        f"   Confidence: {result.get('confidence', 'N/A')}",
        f"   Reasoning: {result.get('reasoning', 'N/A')}",
        "=" * 50
    ]
    return "\n".join(lines)


