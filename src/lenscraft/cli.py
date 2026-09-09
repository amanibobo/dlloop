"""Command-line entry points.

    lenscraft simulate  --substructure vortex --n 100 --seed 0 --run-id run001 [--backend modal]
    lenscraft agent     "make 200 vortex images and 200 with no substructure" [--backend modal] [--yes]
    lenscraft runs      [--backend modal]
    lenscraft summarize run001 run002 [--backend modal]      (or local lensjsonl paths)
    lenscraft fetch     run001 --out data                     (copy a Modal run's records locally)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lenscraft.schema import LensCard, read_records, summarize_records
from lenscraft.schema.lensjsonl import dumps_summary, write_records


def _backend(args: argparse.Namespace):
    from lenscraft.compute import get_backend

    return get_backend(args.backend, data_dir=args.data_dir)


def _build_card(args: argparse.Namespace) -> LensCard:
    data = json.loads(Path(args.card).read_text(encoding="utf-8")) if args.card else {}
    overrides = {
        "n_images": args.n,
        "substructure": args.substructure,
        "halo_mass": args.halo_mass,
        "substructure_mass_fraction": args.mass_fraction,
        "axion_mass": args.axion_mass,
        "instrument": args.instrument,
        "image_size": args.image_size,
        "seed": args.seed,
    }
    data.update({k: v for k, v in overrides.items() if v is not None})
    return LensCard.model_validate(data)


def cmd_simulate(args: argparse.Namespace) -> int:
    card = _build_card(args)
    print(f"LensCard:\n{card.model_dump_json(indent=2)}", file=sys.stderr)
    if args.backend == "local" and args.progress:
        from lenscraft.sim import simulate_lens_batch

        result = simulate_lens_batch(card, args.data_dir, args.run_id, progress=True)
    else:
        backend = _backend(args)
        job_id = backend.submit(card, args.run_id)
        print(f"submitted {job_id} to {backend.name}", file=sys.stderr)
        status = backend.wait(job_id)
        if status.result is None:
            print(json.dumps({"status": status.state, "error": status.error}, indent=2))
            return 1
        result = status.result
    print(result.model_dump_json(indent=2))
    return 0 if result.status == "ok" else 1


def cmd_runs(args: argparse.Namespace) -> int:
    for run_id in _backend(args).list_runs():
        print(run_id)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    backend = None
    records = []
    for item in args.items:
        if Path(item).is_file():
            records.extend(read_records(item))
        else:
            backend = backend or _backend(args)
            records.extend(backend.read_records(item))
    print(dumps_summary(summarize_records(records)))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    backend = _backend(args)
    for run_id in args.run_ids:
        records = backend.read_records(run_id)
        path = Path(args.out) / run_id / "records.lensjsonl"
        n = write_records(path, records)
        if records:
            (path.parent / "card.json").write_text(records[0].lens_card.model_dump_json(indent=2), encoding="utf-8")
        print(f"{run_id}: {n} records -> {path}", file=sys.stderr)
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from lenscraft.agent import AgentDeps, auto_approver, build_agent, console_approver, run_with_approval

    agent = build_agent(args.model, core_only=args.core_only)
    deps = AgentDeps(backend=_backend(args))
    approver = auto_approver if args.yes else console_approver
    result = run_with_approval(agent, args.prompt, deps, approver)
    print(result.output)
    if deps.run_ids:
        print(f"\nruns created: {', '.join(deps.run_ids)}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lenscraft", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", choices=["local", "modal"], default="local", help="compute backend (default: local)")
    common.add_argument("--data-dir", default="data", dest="data_dir", help="local dataset root (local backend only)")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", parents=[common], help="run a LensCard on the compute backend")
    sim.add_argument("--card", help="JSON file with a LensCard; CLI flags override its fields")
    sim.add_argument("--n", type=int, help="n_images")
    sim.add_argument("--substructure", choices=["none", "subhalo", "vortex"])
    sim.add_argument("--halo-mass", type=float)
    sim.add_argument("--mass-fraction", type=float, dest="mass_fraction")
    sim.add_argument("--axion-mass", type=float, dest="axion_mass")
    sim.add_argument("--instrument", choices=["euclid", "custom"])
    sim.add_argument("--image-size", type=int, dest="image_size")
    sim.add_argument("--seed", type=int)
    sim.add_argument("--run-id", required=True, dest="run_id")
    sim.add_argument("--progress", action="store_true", help="per-image progress (local backend only)")
    sim.set_defaults(func=cmd_simulate)

    runs = sub.add_parser("runs", parents=[common], help="list existing runs")
    runs.set_defaults(func=cmd_runs)

    summ = sub.add_parser("summarize", parents=[common], help="composition summary of runs (ids on the backend, or lensjsonl paths)")
    summ.add_argument("items", nargs="+")
    summ.set_defaults(func=cmd_summarize)

    fetch = sub.add_parser("fetch", parents=[common], help="copy run records (not images) from the backend to a local directory")
    fetch.add_argument("run_ids", nargs="+")
    fetch.add_argument("--out", default="data")
    fetch.set_defaults(func=cmd_fetch)

    ag = sub.add_parser("agent", parents=[common], help="natural-language request -> proposed LensCard -> your approval -> run")
    ag.add_argument("prompt")
    ag.add_argument("--model", help="Pydantic AI model string (default: $LENSCRAFT_MODEL or anthropic:claude-opus-5)")
    ag.add_argument("--yes", action="store_true", help="auto-approve every proposed batch (scripts/benchmarks)")
    ag.add_argument("--core-only", action="store_true", dest="core_only", help="benchmark baseline: no domain tools")
    ag.set_defaults(func=cmd_agent)
    return parser


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()  # picks up FIREWORKS_API_KEY / ANTHROPIC_API_KEY / LENSCRAFT_MODEL from ./.env
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
