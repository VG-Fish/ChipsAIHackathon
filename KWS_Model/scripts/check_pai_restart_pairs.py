"""Report whether each dendritic candidate in a run is still resumable.

A candidate's PAI restart state is two files bound by a digest: the native
blob and the KWS sidecar that attests it.  Checking that a sidecar's
``completed_epoch`` matches the seed's last logged epoch does *not* prove the
candidate can resume -- a torn pair passes that check and only fails at the
next launch, seconds after a supervisor has handed the slot to another seed.
This verifies the digest instead, which is the actual resume precondition, so
a torn pair is reported when it happens rather than discovered hours later.

A candidate written before the attested half moved off PAI's own ``latest.pt``
reads as torn for the 1-2 s of every epoch between PAI's write and the pair,
so a first unpaired reading is confirmed by a second one taken far enough
later to span an epoch: a live writer advances ``completed_epoch``, a dead one
does not.  Candidates written by the current code have no such window and are
confirmed immediately.

Intended for the run monitor:

    python scripts/check_pai_restart_pairs.py outputs/<experiment>/seed*

Exits 0 when every committed pair is intact, 1 when any is confirmed torn, so
it can be used directly as a cron predicate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from kws.optimize.dendritic import (
    PaiPairStatus,
    is_confirmed_torn,
    pai_restart_pair_status,
    pai_restart_pair_statuses,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root",
        nargs="+",
        type=Path,
        help="One or more run output directories (each holding manifest.yaml)",
    )
    parser.add_argument(
        "--confirm-after",
        type=float,
        default=30.0,
        help=(
            "Seconds to wait before re-reading an unpaired candidate. Must "
            "exceed one epoch of the run being watched; 0 disables the "
            "second reading and reports the first one as final."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only confirmed torn pairs",
    )
    args = parser.parse_args()

    suspects: list[tuple[Path, PaiPairStatus]] = []
    for run_root in args.run_root:
        statuses = pai_restart_pair_statuses(run_root)
        if not statuses and not args.quiet:
            print(f"{run_root}: no candidate has committed a restart pair yet")
        for status in statuses:
            if status.paired:
                if not args.quiet:
                    print(f"{run_root}: {_describe(status)} paired")
            elif status.repairable:
                # Unpaired on disk, but the sidecar carries the bytes it
                # attests, so the next resume rebuilds them.  Not an alert.
                if not args.quiet:
                    print(
                        f"{run_root}: {_describe(status)} unpaired on disk, "
                        "rebuildable from its sidecar"
                    )
            else:
                suspects.append((run_root, status))

    if not suspects:
        return 0
    if args.confirm_after > 0:
        time.sleep(args.confirm_after)

    torn = 0
    for run_root, status in suspects:
        later = pai_restart_pair_status(
            status.sidecar,
            run_root / "pai" / "candidates" / status.sidecar.parents[1].name,
        )
        if not is_confirmed_torn(status, later):
            if not args.quiet:
                print(
                    f"{run_root}: {_describe(later)} cleared on the second "
                    f"reading ({'paired' if later.paired else later.detail})"
                )
            continue
        torn += 1
        print(
            f"{run_root}: {_describe(later)} TORN -- {later.detail}",
            file=sys.stderr,
        )
    return 1 if torn else 0


def _describe(status: PaiPairStatus) -> str:
    return f"{status.sidecar.parents[1].name} epoch {status.completed_epoch}"


if __name__ == "__main__":
    raise SystemExit(main())
