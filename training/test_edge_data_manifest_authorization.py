"""Contrato estático de las únicas fuentes autorizadas para construir el corpus Edge V1."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "fineweb-en-sample-10bt-proposed": ("HuggingFaceFW/fineweb", "sample-10BT", "9bb295ddab0e05d785b879661af7260fed5140fc"),
    "fineweb2-es-proposed": ("HuggingFaceFW/fineweb-2", "spa_Latn", "af9c13333eb981300149d5ca60a8e9d659b276b9"),
    "hplt2-es-proposed": ("HPLT/HPLT2.0_cleaned", "spa_Latn", "d1324a5283f762ee62c2a5c81de08fc6450ea540"),
    "openr1-math-proposed": ("open-r1/OpenR1-Math-220k", "default", "e4e141ec9dea9f8326f4d347be56105859b2bd68"),
}


def main() -> None:
    manifest = json.loads((ROOT / "aethel_edge_v1.manifest.json").read_text(encoding="utf-8"))
    assert manifest["approval_required"] is True
    assert manifest["authorization_scope"] == "network_data_build_only"
    sources = {source["id"]: source for source in manifest["sources"]}
    assert set(sources) == set(EXPECTED)
    for source_id, (dataset, config, revision) in EXPECTED.items():
        source = sources[source_id]
        assert source["dataset"] == dataset
        assert source["config"] == config
        assert source["revision"] == revision
        assert source["approved"] is True
        assert source["enabled"] is True
    fineweb_en = sources["fineweb-en-sample-10bt-proposed"]
    assert fineweb_en["required_values"] == {"language": "en"}
    openr1 = sources["openr1-math-proposed"]
    assert openr1["required_aligned_true_fields"] == ["is_reasoning_complete", "correctness_math_verify"]
    assert openr1["required_text_fields"] == ["problem", "solution", "answer"]
    assert manifest["minimum_documents_by_language"] == {"en": 120000, "es": 120000}
    capacity = {}
    for source in sources.values():
        capacity[source["language"]] = capacity.get(source["language"], 0) + source["document_limit"]
    for language, minimum in manifest["minimum_documents_by_language"].items():
        assert capacity[language] >= minimum
    print("AETHEL_EDGE_DATA_MANIFEST_AUTHORIZATION_OK")


if __name__ == "__main__":
    main()
