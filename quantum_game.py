#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re
from typing import Iterable, List

import numpy as np
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GATES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
}

INITIAL_STATE = np.array([1, 0], dtype=complex)
TARGET_STATES = {
    "0": np.array([1, 0], dtype=complex),
    "1": np.array([0, 1], dtype=complex),
    "gray": (1 / np.sqrt(2)) * np.array([1, 1], dtype=complex),
}
STATE_NAMES = {
    "0": "black",
    "1": "white",
    "gray": "gray",
}
TARGET_ALIASES = {
    "0": "0",
    "1": "1",
    "black": "0",
    "white": "1",
    "gray": "gray",
    "grey": "gray",
    "superposition": "gray",
    "b": "0",
    "w": "1",
}
BIT_COLORS = {
    0: "#000000",
    1: "#ffffff",
}
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "generated" / "player_circuit.qasm"
LEVELS_DIR = SCRIPT_DIR / "levels"
GATE_DESCRIPTIONS = {
    "I": "Identity: leave the current colour as it is.",
    "X": "Flip: swap black and white.",
    "H": "Hadamard: turn black into an even black/white mix that we call gray.",
    "Y": "Flip with an extra phase twist.",
    "Z": "Phase flip: changes the hidden phase of white.",
    "S": "Quarter-phase gate.",
    "T": "Eighth-phase gate.",
}


def parse_gate_input(raw_text: str):
    pgates = [gate.strip().upper() for gate in re.split(r"[\s,\/]+", raw_text) if gate.strip()]
    invalid = [pgate for pgate in pgates if pgate not in GATES]
    if invalid:
        supported = ", ".join(sorted(GATES))
        raise ValueError(f"Unsupported gate(s): {', '.join(invalid)}. Supported gates: {supported}.")
    return pgates


def normalize_target_choice(raw_text: str) -> str:
    target = raw_text.strip().lower()
    if target not in TARGET_ALIASES:
        raise ValueError("Please choose black, white, or gray.")
    return TARGET_ALIASES[target]


def normalize_level_choice(raw_text: str) -> str:
    level = raw_text.strip().lower()
    if level not in LEVEL_ALIASES:
        raise ValueError(f"Please choose one of: {', '.join(sorted(LEVEL_ALIASES))}.")
    return LEVEL_ALIASES[level]


def parse_level_file(level_path: Path) -> dict:
    config = {}
    for raw_line in level_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid line in {level_path.name}: {raw_line}")
        key, value = line.split(":", 1)
        config[key.strip().lower()] = value.strip()

    required_fields = {"id", "title", "description", "max_gates", "allowed_gates", "order", "aliases", "target"}
    missing = sorted(required_fields - config.keys())
    if missing:
        raise ValueError(f"Missing fields in {level_path.name}: {', '.join(missing)}")

    allowed_gates = [gate.strip().upper() for gate in config["allowed_gates"].split(",") if gate.strip()]
    invalid_gates = [gate for gate in allowed_gates if gate not in GATES]
    if invalid_gates:
        raise ValueError(f"Unsupported gates in {level_path.name}: {', '.join(invalid_gates)}")

    target_value = config["target"].strip().lower()
    target_label = None if target_value == "none" else normalize_target_choice(target_value)

    aliases = [alias.strip().lower() for alias in config["aliases"].split(",") if alias.strip()]
    if not aliases:
        raise ValueError(f"No aliases provided in {level_path.name}")

    return {
        "id": config["id"].strip().lower(),
        "title": config["title"].strip(),
        "description": config["description"].strip(),
        "max_gates": int(config["max_gates"]),
        "allowed_gates": allowed_gates,
        "order": int(config["order"]),
        "aliases": aliases,
        "target_label": target_label,
        "file_name": level_path.name,
    }


def load_levels():
    level_paths = sorted(LEVELS_DIR.glob("*.txt"))
    if not level_paths:
        raise FileNotFoundError(f"No level files found in {LEVELS_DIR}")

    levels = {}
    level_aliases = {}
    ordered_ids = []

    loaded_levels = [parse_level_file(level_path) for level_path in level_paths]
    for level in sorted(loaded_levels, key=lambda item: (item["order"], item["title"])):
        level_id = level["id"]
        levels[level_id] = level
        ordered_ids.append(level_id)
        for alias in level["aliases"]:
            level_aliases[alias] = level_id
        level_aliases[level_id] = level_id

    return levels, level_aliases, ordered_ids


LEVELS, LEVEL_ALIASES, LEVEL_ORDER = load_levels()


def validate_gate_sequence(gates: List[str], allowed_gates: List[str] | None = None, max_gates: int | None = None):
    if max_gates is not None and len(gates) > max_gates:
        raise ValueError(f"You can use at most {max_gates} gates in this mode.")

    if allowed_gates is not None:
        disallowed = [gate for gate in gates if gate not in allowed_gates]
        if disallowed:
            allowed_text = ", ".join(allowed_gates)
            raise ValueError(f"This mode only allows these gates: {allowed_text}.")


def apply_gates(gates: Iterable[str], start_state: np.ndarray | None = None) -> np.ndarray:
    state = np.array(start_state if start_state is not None else INITIAL_STATE, dtype=complex)
    for gate_name in gates:
        state = GATES[gate_name] @ state
    return state


def build_qasm(gates: Iterable[str]) -> str:
    lines = [
        "OPENQASM 2.0;",
        "qreg q[1];",
        "creg c[1];",
    ]
    for gate_name in gates:
        lines.append(f"{gate_name.lower()} q[0];")
    lines.append("measure q[0] -> c[0];")
    return "\n".join(lines) + "\n"


def write_qasm(program: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(program)
    return output_path


def states_match_up_to_global_phase(actual: np.ndarray, desired: np.ndarray, tolerance: float = 1e-9) -> bool:
    overlap = np.vdot(desired, actual)
    return np.isclose(abs(overlap), 1.0, atol=tolerance)


def normalize_measurements(raw_measurements) -> List[int]:
    if not isinstance(raw_measurements, list):
        raise ValueError(f"Unexpected Quokka result type: {type(raw_measurements).__name__}")

    shots: List[int] = []
    for measurement in raw_measurements:
        if isinstance(measurement, list) and measurement:
            shots.append(int(measurement[0]))
        elif isinstance(measurement, str) and measurement:
            shots.append(int(measurement[0]))
        else:
            shots.append(int(measurement))
    return shots


def sample_measurements_locally(state: np.ndarray, shots: int) -> List[int]:
    probabilities = np.abs(state) ** 2
    return list(np.random.choice([0, 1], size=shots, p=probabilities))


def send_to_quokka(program: str, shots: int, quokka_name: str = "quokka1") -> List[int]:
    request_http = f"http://{quokka_name}.quokkacomputing.com/qsim/qasm"
    payload = {
        "script": program,
        "count": shots,
    }
    response = requests.post(request_http, json=payload, verify=False, timeout=30)
    response.raise_for_status()
    json_obj = json.loads(response.content)
    return normalize_measurements(json_obj["result"]["c"])


def collect_measurements(program: str, state: np.ndarray, shots: int, quokka_name: str, allow_local_fallback: bool):
    try:
        return send_to_quokka(program, shots, quokka_name), "Quokka", None
    except Exception as exc:
        if not allow_local_fallback:
            raise RuntimeError(f"Quokka request failed: {exc}") from exc
        return (
            sample_measurements_locally(state, shots),
            "Local fallback",
            "Quokka was unavailable, so these shots were sampled from the analytic probabilities.",
        )


def measurement_counts(measurements: Iterable[int]) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    for bit in measurements:
        counts[str(int(bit))] += 1
    return counts


def state_name(bit_value: int | str) -> str:
    return STATE_NAMES[str(bit_value)]


def default_plot_path(qasm_path: Path) -> Path:
    return qasm_path.with_name(f"{qasm_path.stem}_measurements.png")


def plot_measurements(measurements: Iterable[int], plot_path: Path) -> Path:
    measurements = list(measurements)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    xs = rng.uniform(0.02, 0.98, size=len(measurements))
    ys = rng.uniform(0.02, 0.98, size=len(measurements))

    fig, ax = plt.subplots(figsize=(2, 2), dpi=180)
    fig.patch.set_facecolor("#d7d7d7")
    ax.set_facecolor("#d7d7d7")

    for bit in (0, 1):
        mask = np.array(measurements) == bit
        if not np.any(mask):
            continue
        ax.scatter(
            xs[mask],
            ys[mask],
            s=170,
            c=BIT_COLORS[bit],
            alpha=0.4,
            edgecolors="#000000",
            linewidths=0.8,
            label=f"Measured {state_name(bit)}",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(1.0)

    ax.set_title("Measured Colours", fontsize=12, color="#222222", pad=8)
    ax.text(
        0.02,
        -0.12,
        "Black and white dots overlap into gray",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
    )

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def build_dot_strip(measurements: Iterable[int], max_dots: int = 60) -> str:
    measurements = list(measurements)
    if not measurements:
        return ""
    if len(measurements) <= max_dots:
        sample = measurements
    else:
        step = len(measurements) / max_dots
        sample = [measurements[int(i * step)] for i in range(max_dots)]
    return "".join("." if bit == 0 else "o" for bit in sample)


def print_round_report(
    mode_name: str,
    target_label: str,
    gates: List[str],
    state: np.ndarray,
    measurements: List[int],
    source_label: str,
    qasm_path: Path,
    plot_path: Path,
    measurement_note: str | None,
):
    counts = measurement_counts(measurements)
    total = len(measurements)
    success = None if target_label is None else states_match_up_to_global_phase(state, TARGET_STATES[target_label])

    print()
    print("Round result")
    print(f"Mode: {mode_name}")
    print("Starting colour: black")
    if target_label is None:
        print("Target colour: none in playground mode")
    else:
        print(f"Target colour: {state_name(target_label)}")
    print(f"Gates used: {' '.join(gates) if gates else '(none)'}")
    print(f"QASM file: {qasm_path}")
    print(f"Measurement plot: {plot_path}")
    print()
    print(f"Measurements from {source_label}:")
    print(f"Black: {counts['0']:>4}/{total} = {counts['0'] / total:6.1%}")
    print(f"White: {counts['1']:>4}/{total} = {counts['1'] / total:6.1%}")
    if measurement_note:
        print(f"Note: {measurement_note}")
    print("Colour strip (. = black, o = white):")
    print(build_dot_strip(measurements))
    print()
    if success is None:
        print("Playground mode: no target check for this round.")
    else:
        print("YAY: you reached the target colour." if success else "NAY: you did not reach the target colour.")
    print()


def run_round(
    mode_name: str,
    target_label: str | None,
    gate_text: str,
    shots: int,
    quokka_name: str,
    qasm_path: Path,
    allow_local_fallback: bool,
    allowed_gates: List[str] | None = None,
    max_gates: int | None = None,
):
    gates = parse_gate_input(gate_text)
    validate_gate_sequence(gates, allowed_gates=allowed_gates, max_gates=max_gates)
    state = apply_gates(gates)
    program = build_qasm(gates)
    write_qasm(program, qasm_path)
    measurements, source_label, measurement_note = collect_measurements(
        program, state, shots, quokka_name, allow_local_fallback
    )
    plot_path = plot_measurements(measurements, default_plot_path(qasm_path))
    print_round_report(mode_name, target_label, gates, state, measurements, source_label, qasm_path, plot_path, measurement_note)


def print_gate_help(allowed_gates: List[str]):
    print("Available gates:")
    for gate in allowed_gates:
        print(f"  {gate}: {GATE_DESCRIPTIONS[gate]}")


def print_level_intro(level_key: str):
    level = LEVELS[level_key]
    print()
    print(level["title"])
    print(level["description"])
    print(f"Maximum gates: {level['max_gates']}")
    print_gate_help(level["allowed_gates"])
    print()


def prompt_level() -> str:
    while True:
        print("Choose a mode:")
        for level_id in LEVEL_ORDER:
            level = LEVELS[level_id]
            prompt_alias = level["aliases"][0]
            print(f"  {prompt_alias}: {level['title']}")
        try:
            return normalize_level_choice(input("Enter your choice: "))
        except ValueError as exc:
            print(exc)
            print()


def prompt_gate_text(allowed_gates: List[str], max_gates: int) -> str:
    while True:
        gate_text = input(
            f"Enter up to {max_gates} gates from {', '.join(allowed_gates)} "
            "(examples: X or H / X). Press Enter for no gate: "
        ).strip()
        try:
            gates = parse_gate_input(gate_text)
            validate_gate_sequence(gates, allowed_gates=allowed_gates, max_gates=max_gates)
            return gate_text
        except ValueError as exc:
            print(exc)


def prompt_next_action() -> str:
    while True:
        answer = input("Choose next action: replay level (r), level menu (m), or quit (q): ").strip().lower()
        if answer in {"r", "replay"}:
            return "replay"
        if answer in {"m", "menu"}:
            return "menu"
        if answer in {"q", "quit"}:
            return "quit"
        print("Please enter r, m, or q.")


def interactive_game(args):
    print("One-Qubit Quantum Game Prototype")
    print("Goal: start from black, choose a level, and use quantum gates to reach the target colour.")
    print()

    while True:
        level_key = prompt_level()
        level = LEVELS[level_key]

        while True:
            print_level_intro(level_key)
            gate_text = prompt_gate_text(level["allowed_gates"], level["max_gates"])
            run_round(
                mode_name=level["title"],
                target_label=level["target_label"],
                gate_text=gate_text,
                shots=args.shots,
                quokka_name=args.quokka,
                qasm_path=Path(args.output),
                allow_local_fallback=not args.no_local_fallback,
                allowed_gates=level["allowed_gates"],
                max_gates=level["max_gates"],
            )
            next_action = prompt_next_action()
            if next_action == "replay":
                continue
            if next_action == "menu":
                print()
                break
            return


def build_parser():
    parser = argparse.ArgumentParser(description="Simple one-qubit command line quantum game.")
    parser.add_argument("--target", help="Desired final colour for direct mode: black, white, or gray.")
    parser.add_argument("--gates", default="", help="Gate list, for example: 'X' or 'H / X'.")
    parser.add_argument("--shots", type=int, default=200, help="Number of measurements to request.")
    parser.add_argument("--quokka", default="quokka1", help="Quokka backend name, for example quokka1.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the generated QASM file.",
    )
    parser.add_argument(
        "--no-local-fallback",
        action="store_true",
        help="Fail instead of locally sampling when Quokka is unavailable.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.shots <= 0:
        parser.error("--shots must be a positive integer.")

    if args.target is None:
        interactive_game(args)
        return

    try:
        normalized_target = normalize_target_choice(args.target)
    except ValueError as exc:
        parser.error(str(exc))

    run_round(
        mode_name="Direct Mode",
        target_label=normalized_target,
        gate_text=args.gates,
        shots=args.shots,
        quokka_name=args.quokka,
        qasm_path=Path(args.output),
        allow_local_fallback=not args.no_local_fallback,
    )


if __name__ == "__main__":
    main()
