"""Command-line entry points.

    lenscraft simulate  --substructure vortex --n 100 --seed 0 --run-id run001 [--backend modal]
    lenscraft train     --kind classifier --arch resnet18 --model-id clf_a --runs r1 r2 r3 --epochs 10 [--backend modal]
    lenscraft evaluate  --model-id clf_a --runs r4 [--backend modal]
    lenscraft uncertainty --model-id clf_a --runs r4 [--backend modal]
    lenscraft loop      --model-id clf_a --train-runs r1 r2 r3 --test-runs r4 --rounds 2 [--yes] [--backend modal]
    lenscraft agent     "make 200 vortex images and 200 with no substructure" [--backend modal] [--yes]
    lenscraft runs / models  [--backend modal]
    lenscraft summarize run001 run002 [--backend modal]      (or local lensjsonl paths)
    lenscraft fetch     run001 --out data                     (copy a Modal run's records locally)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lenscraft.schema import EvalSpec, LensCard, TrainSpec, read_records, summarize_records
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


def _print_and_exit(result) -> int:
    print(result.model_dump_json(indent=2))
    return 0 if result.status == "ok" else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    card = _build_card(args)
    print(f"LensCard:\n{card.model_dump_json(indent=2)}", file=sys.stderr)
    if args.backend == "local" and args.progress:
        from lenscraft.sim import simulate_lens_batch

        return _print_and_exit(simulate_lens_batch(card, args.data_dir, args.run_id, progress=True))
    backend = _backend(args)
    job_id = backend.submit(card, args.run_id)
    print(f"submitted {job_id} to {backend.name}", file=sys.stderr)
    status = backend.wait(job_id)
    if status.result is None:
        print(json.dumps({"status": status.state, "error": status.error}, indent=2))
        return 1
    return _print_and_exit(status.result)


def cmd_train(args: argparse.Namespace) -> int:
    spec = TrainSpec(
        model_id=args.model_id, kind=args.kind, arch=args.arch, run_ids=args.runs, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr, input_size=args.input_size, seed=args.seed, max_images=args.max_images,
    )
    print(f"TrainSpec:\n{spec.model_dump_json(indent=2)}", file=sys.stderr)
    backend = _backend(args)
    job_id = backend.submit_train(spec)
    print(f"submitted {job_id} to {backend.name}", file=sys.stderr)
    status = backend.wait(job_id)
    if status.result is None:
        print(json.dumps({"status": status.state, "error": status.error}, indent=2))
        return 1
    return _print_and_exit(status.result)


def cmd_evaluate(args: argparse.Namespace) -> int:
    spec = EvalSpec(model_id=args.model_id, run_ids=args.runs, max_images=args.max_images)
    return _print_and_exit(_backend(args).evaluate(spec))


def cmd_uncertainty(args: argparse.Namespace) -> int:
    from lenscraft.ml.uncertainty import sample_uncertainty

    report = sample_uncertainty(_backend(args), args.model_id, args.runs, n_bins=args.n_bins, n_images_proposed=args.n_images)
    print(report.model_dump_json(indent=2))
    return 0 if report.status == "ok" else 1


def cmd_loop(args: argparse.Namespace) -> int:
    from lenscraft.agent import auto_approver, console_approver
    from lenscraft.loop import run_loop

    history = run_loop(
        _backend(args), model_id=args.model_id, train_runs=args.train_runs, test_runs=args.test_runs, rounds=args.rounds,
        approve=auto_approver if args.yes else console_approver, epochs=args.epochs, n_images=args.n_images, prefix=args.prefix,
        max_images=args.max_images, input_size=args.input_size,
    )
    for h in history:
        print(h.model_dump_json(indent=2))
    return 0 if history and history[-1].new_model_id else 1


def cmd_runs(args: argparse.Namespace) -> int:
    for run_id in _backend(args).list_runs():
        print(run_id)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    for model_id in _backend(args).list_models():
        print(model_id)
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
    if deps.model_ids:
        print(f"models trained: {', '.join(deps.model_ids)}", file=sys.stderr)
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

    tr = sub.add_parser("train", parents=[common], help="train a classifier or anomaly detector on existing runs")
    tr.add_argument("--model-id", required=True, dest="model_id")
    tr.add_argument("--kind", choices=["classifier", "anomaly"], required=True)
    tr.add_argument("--arch", required=True, help="resnet18|alexnet or dcae|vae|aae")
    tr.add_argument("--runs", nargs="+", required=True)
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--input-size", type=int, default=150, dest="input_size")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--max-images", type=int, dest="max_images")
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", parents=[common], help="evaluate a trained model on runs")
    ev.add_argument("--model-id", required=True, dest="model_id")
    ev.add_argument("--runs", nargs="+", required=True)
    ev.add_argument("--max-images", type=int, dest="max_images")
    ev.set_defaults(func=cmd_evaluate)

    unc = sub.add_parser("uncertainty", parents=[common], help="where is a model weakest? proposes the next LensCard")
    unc.add_argument("--model-id", required=True, dest="model_id")
    unc.add_argument("--runs", nargs="+", required=True, help="held-out runs to analyse")
    unc.add_argument("--n-bins", type=int, default=4, dest="n_bins")
    unc.add_argument("--n-images", type=int, default=2000, dest="n_images", help="size of the proposed batch")
    unc.set_defaults(func=cmd_uncertainty)

    lp = sub.add_parser("loop", parents=[common], help="active-learning rounds: uncertainty -> approve -> simulate -> retrain -> evaluate")
    lp.add_argument("--model-id", required=True, dest="model_id", help="starting trained model")
    lp.add_argument("--train-runs", nargs="+", required=True, dest="train_runs")
    lp.add_argument("--test-runs", nargs="+", required=True, dest="test_runs")
    lp.add_argument("--rounds", type=int, default=1)
    lp.add_argument("--epochs", type=int, default=10)
    lp.add_argument("--n-images", type=int, default=2000, dest="n_images")
    lp.add_argument("--prefix", default="al", help="run_id prefix for the new batches")
    lp.add_argument("--input-size", type=int, default=150, dest="input_size")
    lp.add_argument("--max-images", type=int, dest="max_images")
    lp.add_argument("--yes", action="store_true", help="auto-approve proposed batches")
    lp.set_defaults(func=cmd_loop)

    runs = sub.add_parser("runs", parents=[common], help="list existing runs")
    runs.set_defaults(func=cmd_runs)

    models = sub.add_parser("models", parents=[common], help="list trained models")
    models.set_defaults(func=cmd_models)

    summ = sub.add_parser("summarize", parents=[common], help="composition summary of runs (ids on the backend, or lensjsonl paths)")
    summ.add_argument("items", nargs="+")
    summ.set_defaults(func=cmd_summarize)

    fetch = sub.add_parser("fetch", parents=[common], help="copy run records (not images) from the backend to a local directory")
    fetch.add_argument("run_ids", nargs="+")
    fetch.add_argument("--out", default="data")
    fetch.set_defaults(func=cmd_fetch)

    ag = sub.add_parser("agent", parents=[common], help="natural-language request -> proposed tool calls -> your approval -> run")
    ag.add_argument("prompt")
    ag.add_argument("--model", help="Pydantic AI model string (default: $LENSCRAFT_MODEL or anthropic:claude-opus-5)")
    ag.add_argument("--yes", action="store_true", help="auto-approve every proposed batch/training job (scripts/benchmarks)")
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
