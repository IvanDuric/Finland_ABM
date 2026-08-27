"""Build a de-identified Portugal Fruits deployment artefact.

The raw Firebase export stays outside version control.  The generated JSON
contains only the ABM catalogue, sequentially de-identified behavioural
profiles, calibration diagnostics, and aggregate study metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from portugal_fruits import build_portugal_fruit_config


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Private Firebase export")
    parser.add_argument("output", type=Path, help="De-identified JSON artefact")
    args = parser.parse_args()

    export = json.loads(args.source.read_text(encoding="utf-8"))
    config = build_portugal_fruit_config(export, pool_size=2000, n_archetypes=4)

    # The pipeline already replaces Firebase identifiers with sequential IDs.
    # Assert that invariant before creating an artefact intended for deployment.
    for profile in config["population"]:
        source_id = str(profile.get("source_id", ""))
        if not source_id.startswith("pt_preliminary_"):
            raise RuntimeError("A non-de-identified participant identifier remains")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, separators=(",", ":"), default=_json_value),
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} with {len(config['population'])} de-identified profiles "
        f"and {len(config['products'])} products"
    )


if __name__ == "__main__":
    main()
