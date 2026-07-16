import pytest

from rag_modules.generation_integration import GenerationIntegrationModule


def test_kimi_k2_temperature_is_normalized_to_supported_value():
    assert GenerationIntegrationModule._normalize_temperature("kimi-k2.6", 0.2) == 0.6


def test_non_k2_temperature_is_preserved():
    assert GenerationIntegrationModule._normalize_temperature("moonshot-v1-8k", 0.2) == 0.2


def test_kimi_k2_6_disables_thinking_for_short_rag_answers():
    assert GenerationIntegrationModule._model_extra_body("kimi-k2.6") == {
        "thinking": {"type": "disabled"}
    }


def test_empty_answer_stream_raises_clear_error():
    with pytest.raises(RuntimeError, match="no answer content"):
        list(GenerationIntegrationModule._nonempty_chunks(["", ""]))
