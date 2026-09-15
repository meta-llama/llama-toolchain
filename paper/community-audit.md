# JOSS community metrics audit, 15 September 2026

The manuscript's development metrics use the end of 2 June 2026 in UTC as
their cutoff (`2026-06-03T00:00:00Z`, exclusive). This audit distinguishes
historical Git and release records from current GitHub account statistics.

## Retained historical figures

| Metric | Evidence | Manuscript wording |
| --- | --- | --- |
| Commits | 4,102 commits reachable from [`5a0e1010ccf6e777d24cb6eb25b43e2d0b550c6e`](https://github.com/ogx-ai/ogx/commit/5a0e1010ccf6e777d24cb6eb25b43e2d0b550c6e), the last first-parent main commit before the cutoff in the inspected history | Over 4,000 Git commits |
| Published releases | 68 non-draft GitHub release records with `published_at` before the cutoff; 67 stable releases and one prerelease, `v0.1.0rc12` | 68 GitHub releases, including one prerelease |
| Start of Git history | Root commit [`5d5acc8ed559ca3ba68f382be052adb1f55167a0`](https://github.com/ogx-ai/ogx/commit/5d5acc8ed559ca3ba68f382be052adb1f55167a0), committed on 23 July 2024 | Git history dating to July 2024 |

The commit count includes every reachable commit, including merges. It is
not a count of pull requests, first-parent commits, or commits on all branches.
The inspected main snapshot is
[`810d83f434c1abf237be7d5fdf8c2012e704f080`](https://github.com/ogx-ai/ogx/tree/810d83f434c1abf237be7d5fdf8c2012e704f080).
The local checkout is not shallow.

The release count uses the public [GitHub release records](https://api.github.com/repos/ogx-ai/ogx/releases)
retrieved on 15 September 2026. It uses publication timestamps, rather than
tag creation dates or the number of tags. The current inventory contains 86
published releases; filtering by the manuscript's cutoff yields 68.

These commands reproduce the retained counts from a full clone with `gh`
and `jq` installed:

```bash
git log 810d83f434c1abf237be7d5fdf8c2012e704f080 --first-parent \
  --before='2026-06-03T00:00:00Z' -1 --format='%H %cI'
git rev-list --count 5a0e1010ccf6e777d24cb6eb25b43e2d0b550c6e
git show -s --format='%H %cI' 5d5acc8ed559ca3ba68f382be052adb1f55167a0
gh api repos/ogx-ai/ogx/releases --paginate --slurp |
  jq '[.[][] | select(.draft == false and .published_at < "2026-06-03T00:00:00Z")] |
    {published: length, prereleases: ([.[] | select(.prerelease)] | length)}'
```

## Removed ambiguous figures

The original sentence also claimed over 8,400 stars and 242 contributors
as of June 2026. Both figures are removed from the manuscript:

- The [current repository metadata](https://api.github.com/repos/ogx-ai/ogx)
  reported 8,428 stars on 15 September 2026. That observation does not
  establish the star count on 2 June 2026. No contemporaneous snapshot was
  verified in this audit.
- At the pinned June Git snapshot, there are 244 distinct raw author names
  and 262 distinct raw author email addresses. Removing the two names ending
  in `[bot]` yields 242 names, but names and email addresses are not verified
  unique people or GitHub accounts. The current
  [GitHub contributors endpoint](https://api.github.com/repos/ogx-ai/ogx/contributors)
  returns 256 accounts, including three with type `Bot`; it does not provide
  the historical account count for the paper's date. The original sentence
  did not define its counting rule.

This change preserves the dated development evidence without substituting
current popularity or account statistics for historical measurements. The
separate deployment and research-impact statements are unchanged.
