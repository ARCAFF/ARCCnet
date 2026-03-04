"""Point-in-time (PIT) training wrapper for timeseries flare forecasting."""

from __future__ import annotations

import sys
import argparse
import subprocess


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run timeseries training in PIT mode (single timestep, optional no-temporal model) "
            "while forwarding all other args to arccnet.models.timeseries.train."
        )
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1,
        help="Number of timesteps to load for PIT mode (default: 1).",
    )
    parser.add_argument(
        "--timestep_selection",
        type=str,
        default="last",
        choices=["first", "last"],
        help="Whether selected timesteps come from the beginning or end of each sample (default: last).",
    )
    parser.add_argument(
        "--use_temporal_transformer",
        action="store_true",
        help="Enable temporal transformer even in PIT mode (disabled by default).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    pit_args, passthrough_args = parser.parse_known_args(argv)
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    if pit_args.num_timesteps < 1:
        parser.error("--num_timesteps must be >= 1")

    cmd = [
        sys.executable,
        "-m",
        "arccnet.models.timeseries.train",
        "--num_timesteps",
        str(int(pit_args.num_timesteps)),
        "--timestep_selection",
        str(pit_args.timestep_selection),
        "--use_temporal_transformer",
        "true" if pit_args.use_temporal_transformer else "false",
        *passthrough_args,
    ]
    print(
        "Launching PIT training with args: "
        f"--num_timesteps {pit_args.num_timesteps} "
        f"--timestep_selection {pit_args.timestep_selection} "
        f"--use_temporal_transformer {'true' if pit_args.use_temporal_transformer else 'false'}"
    )
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
