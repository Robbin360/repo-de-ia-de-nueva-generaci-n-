from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    wikimedia: bool
    gutenberg: bool
    olc: bool
    count: int
    size: int
    digest: str


def priority(candidate: Candidate):
    return (candidate.wikimedia, candidate.gutenberg, candidate.olc, candidate.count, candidate.size, candidate.digest)


old_olc = Candidate(False, False, True, 79, 327680, "old")
old_gutenberg_olc = Candidate(False, True, True, 79, 327680, "old-gutenberg")
new_wikimedia_gutenberg_olc = Candidate(True, True, True, 80, 340000, "new")
new_wikimedia_gutenberg_only = Candidate(True, True, False, 79, 330000, "new2")

ordered = sorted([old_olc, old_gutenberg_olc, new_wikimedia_gutenberg_olc, new_wikimedia_gutenberg_only], key=priority, reverse=True)
assert ordered[0] is new_wikimedia_gutenberg_olc
assert ordered[1] is new_wikimedia_gutenberg_only
assert ordered[2] is old_gutenberg_olc
assert ordered[3] is old_olc
print("OK: El respaldo Wikimedia y Project Gutenberg vencen a copias antiguas")
