from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    gutenberg: bool
    olc: bool
    count: int
    size: int
    digest: str


def priority(candidate: Candidate):
    return (candidate.gutenberg, candidate.olc, candidate.count, candidate.size, candidate.digest)


old_olc = Candidate(False, True, 79, 327680, "old")
new_gutenberg_olc = Candidate(True, True, 79, 327680, "new")
new_gutenberg_only = Candidate(True, False, 78, 317440, "new2")

ordered = sorted([old_olc, new_gutenberg_olc, new_gutenberg_only], key=priority, reverse=True)
assert ordered[0] is new_gutenberg_olc
assert ordered[1] is new_gutenberg_only
assert ordered[2] is old_olc
print("OK: Project Gutenberg tiene prioridad sobre copias antiguas con OLC")
