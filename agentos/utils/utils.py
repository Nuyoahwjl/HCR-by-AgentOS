from openai import OpenAI


def call_model(messages, api_key: str | None = None, model: str = "deepseek-chat"):
    """Call DeepSeek API (OpenAI-compatible).

    Args:
        messages: List of {"role": ..., "content": ...} message dicts.
        api_key: DeepSeek API key. If None, uses DEEPSEEK_API_KEY env var.
        model: Model name. "deepseek-chat" for DeepSeek V3, "deepseek-reasoner" for R1.

    Returns:
        The model response string.
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return completion.choices[0].message.content


def call_model_stream(messages, api_key: str | None = None, model: str = "deepseek-chat"):
    """Call DeepSeek API with streaming.

    Args:
        messages: List of {"role": ..., "content": ...} message dicts.
        api_key: DeepSeek API key.
        model: Model name.

    Yields:
        Content chunks as they arrive.
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
