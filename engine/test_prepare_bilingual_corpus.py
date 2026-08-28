from prepare_bilingual_corpus import text_from_source_row, validate_minimum_language_capacity


def main() -> None:
    source = {
        "id": "math-contract",
        "required_text_fields": ["problem", "solution", "answer"],
        "required_aligned_true_fields": ["is_reasoning_complete", "correctness_math_verify"],
        "text_template": "P:{problem}\nS:{solution}\nA:{answer}",
    }
    accepted = {
        "problem": "2 + 2",
        "solution": "Sumamos dos y dos.",
        "answer": "4",
        "is_reasoning_complete": [False, True],
        "correctness_math_verify": [False, True],
        "generations": ["No debe usarse"],
    }
    assert text_from_source_row(source, accepted) == "P:2 + 2\nS:Sumamos dos y dos.\nA:4"
    assert text_from_source_row(source, {**accepted, "correctness_math_verify": [True, False]}) is None
    assert text_from_source_row(source, {key: value for key, value in accepted.items() if key != "answer"}) is None
    assert text_from_source_row({"id": "plain", "text_column": "text"}, {"text": "texto"}) == "texto"
    validate_minimum_language_capacity(
        [{"id": "web-en", "language": "en", "document_limit": 100}, {"id": "math-en", "language": "en", "document_limit": 20}],
        {"en": 120},
    )
    try:
        validate_minimum_language_capacity(
            [{"id": "web-en", "language": "en", "document_limit": 100}], {"en": 120}
        )
    except RuntimeError as error:
        assert "Plan de datos inviable para en" in str(error)
    else:
        raise AssertionError("La capacidad inglesa insuficiente debe detener el plan antes de red")
    print("AETHEL_EDGE_SOURCE_TEXT_CONTRACT_OK")


if __name__ == "__main__":
    main()
