"""Prueba de los invariantes reproducibles del preparador de corpus."""
from __future__ import annotations

from prepare_corpus import belongs_to_validation, normalize_text


def main() -> None:
    digest = "b" * 64
    assert belongs_to_validation(digest, 0.5, 17) == belongs_to_validation(digest, 0.5, 17)
    assert not belongs_to_validation(digest, 0.0, 17)
    assert belongs_to_validation(digest, 1.0, 17)
    assert normalize_text("Correo: a@b.com", True) == "Correo: [EMAIL_REDACTED]"
    print('{"corpus_preparation_invariants_verified":true}')


if __name__ == "__main__":
    main()

