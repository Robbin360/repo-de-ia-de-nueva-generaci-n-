from prepare_bilingual_corpus import validate_hf_configurations


def fake_config_names(dataset: str, **kwargs) -> list[str]:
    assert kwargs["revision"] == "fixed-revision"
    assert set(kwargs) == {"revision"}
    return ["default", "sample-10BT"] if dataset == "HuggingFaceFW/fineweb" else ["spa_Latn"]


def main() -> None:
    valid = [
        {
            "id": "fineweb-en",
            "kind": "hf_text",
            "dataset": "HuggingFaceFW/fineweb",
            "config": "sample-10BT",
            "revision": "fixed-revision",
        },
        {
            "id": "hplt-es",
            "kind": "hf_text",
            "dataset": "HPLT/HPLT2.0_cleaned",
            "config": "spa_Latn",
            "revision": "fixed-revision",
        },
    ]
    validate_hf_configurations(valid, fake_config_names)
    invalid = [{**valid[0], "id": "invalid", "config": "eng_Latn"}]
    try:
        validate_hf_configurations(invalid, fake_config_names)
    except RuntimeError as error:
        assert "Configuración HF no disponible" in str(error)
        assert "eng_Latn" in str(error)
    else:
        raise AssertionError("La configuración inexistente debe fallar antes de crear una salida.")
    print("AETHEL_HF_CONFIG_PREFLIGHT_CONTRACT_OK")


if __name__ == "__main__":
    main()
