# Fixed attention excerpt rule

This change only selects existing recorded pixels for a legible appendix figure;
it runs no recognizer and does not recalculate attention or consult outcomes.

From the frozen contact-sheet manifest, sort geometry names lexicographically and
use the first two. Within each selected geometry, choose the lexicographically
first source-frame sheet (zero-padded board IDs). Use the eight diagonal square
queries in canonical order: a8, b7, c6, d5, e4, f3, g2, h1. The same query list is
used for both boards. Crops are exactly the recorded tile boundaries.

The figure has two blocks, each with two rows of four query tiles. Vector query
labels remain readable when printed on A4. The caption states both selection
rules and points to the full evidence directory. Preserve all original full
query sheets and hashes; the original three appendix sheets are not deleted.

Acceptance: the locked full manifest and all stored sheets verify; selection is
invariant to manifest order and unrelated outcomes; exactly two boards and eight
queries per board appear; Appendix E occupies one page with its caption at A4
print size. Regenerate the figure and build receipts, confirm other figures are
unchanged, run the fresh remote main/branch failing-ID comparison, and push the
branch with lease protection. Publication and final-test scoring stay disabled.
