import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from materialize_aethel_knowledge_corpus import clean_wikitext, domains_for, stable_split


class KnowledgeCorpusMaterializerTests(unittest.TestCase):
    def test_wikitext_cleaning_keeps_link_text_and_removes_markup(self) -> None:
        source = "== Algebra ==\n{{stub}} [[Equation|equation]] <ref>citation</ref> [https://example.org source]"
        self.assertEqual(clean_wikitext(source), "Algebra equation source")

    def test_domain_labels_are_metadata_and_cover_expected_english_topics(self) -> None:
        labels = domains_for("Algebra", "Mathematics uses equations and algorithms.", "en")
        self.assertIn("general", labels)
        self.assertIn("mathematics", labels)
        self.assertIn("programming", labels)

    def test_split_is_deterministic_and_partitioned(self) -> None:
        first = stable_split("document-1", "seed", 95)
        self.assertEqual(first, stable_split("document-1", "seed", 95))
        self.assertIn(first, {"train", "holdout"})


if __name__ == "__main__":
    unittest.main()
