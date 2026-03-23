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
    "T": "Eighth-phase gate.",
}


def parse_gate_input(raw_text: str):
    pgates = [gate.strip().upper() for gate in re.split(r"[\s,\/]+", raw_text) if gate.strip()]
    invalid = [pgate for pgate in pgates if pgate not in GATES]
    if invalid:
        supported = ", ".join(sorted(GATES))
        raise ValueError(f"Unsupported gate(s): {', '.join(invalid)}. Supported gates: {supported}.")
    return pgates

def format_ratio_label(black_weight: float, white_weight: float) -> str:
    black_pct = black_weight * 100
    white_pct = white_weight * 100
    if np.isclose(black_pct, round(black_pct)) and np.isclose(white_pct, round(white_pct)):
        return f"{int(round(black_pct))}/{int(round(white_pct))}"
    return f"{black_pct:.1f}/{white_pct:.1f}"


def parse_colour_state(raw_text: str) -> tuple[np.ndarray, str]:
    raw_value = raw_text.strip()
    normalized = raw_value.lower()
    if normalized in TARGET_ALIASES:
        canonical = TARGET_ALIASES[normalized]
        return TARGET_STATES[canonical], STATE_NAMES[canonical]

    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)\s*", raw_value)
    percent_match = re.fullmatch(
        r"\s*([0-9]*\.?[0-9]+)\s*%?\s*black\s*[-/]\s*([0-9]*\.?[0-9]+)\s*%?\s*white\s*",
        normalized,
    )
    if match:
        black_weight = float(match.group(1))
        white_weight = float(match.group(2))
    elif percent_match:
        black_weight = float(percent_match.group(1))
        white_weight = float(percent_match.group(2))
    else:
        raise ValueError("Please choose black, white, gray, or a ratio like 70/30.")

    total = black_weight + white_weight
    if total <= 0:
        raise ValueError("The black/white ratio must add up to more than zero.")

    black_probability = black_weight / total
    white_probability = white_weight / total
    state = np.array([np.sqrt(black_probability), np.sqrt(white_probability)], dtype=complex)
    label = format_ratio_label(black_probability, white_probability)
    if np.isclose(black_probability, 0.5) and np.isclose(white_probability, 0.5):
        label = f"{label} (gray)"
    return state, label


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

    target_value = config["target"].strip()
    if target_value.lower() == "none":
        target_state = None
        target_label = None
    else:
        target_state, target_label = parse_colour_state(target_value)

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
        "target_state": target_state,
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


def build_qasm(gates: Iterable[str], start_state: np.ndarray | None = None) -> str:
    lines = [
        "OPENQASM 2.0;",
        "qreg q[1];",
        "creg c[1];",
    ]
    initial_state = np.array(start_state if start_state is not None else INITIAL_STATE, dtype=complex)
    if not np.allclose(initial_state, INITIAL_STATE):
        theta = 2 * np.arctan2(np.abs(initial_state[1]), np.abs(initial_state[0]))
        lines.append(f"ry({theta:.12f}) q[0];")
    for gate_name in gates:
        if gate_name == "I":
            continue
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
        if isinstance(measurement, int):
            if measurement in (0, 1):
                shots.append(measurement)
            else:
                raise ValueError(f"Unexpected measurement value: {measurement}")
        elif isinstance(measurement, list) and len(measurement) == 1:
            bit_value = measurement[0]
            if isinstance(bit_value, int) and bit_value in (0, 1):
                shots.append(bit_value)
            else:
                raise ValueError(f"Unexpected measurement value: {measurement}")
        else:
            raise ValueError(f"Unexpected measurement value: {measurement}")

    return shots


def extract_quokka_measurements(response_payload) -> List[int]:

    if isinstance(response_payload.get("result"), dict) and "c" in response_payload["result"]:
        return normalize_measurements(response_payload["result"]["c"])

    if "c" in response_payload:
        return normalize_measurements(response_payload["c"])

    if isinstance(response_payload.get("data"), dict) and "c" in response_payload["data"]:
        return normalize_measurements(response_payload["data"]["c"])

    raise ValueError(f"Unexpected Quokka response payload: {response_payload}")


def send_to_quokka(program: str, count: int, my_quokka: str = "quokka1") -> List[int]:
    request_http = f"http://{my_quokka}.quokkacomputing.com/qsim/qasm"
    data = {
        "script": program,
        "count": count,
    }
    response = requests.post(request_http, json=data, verify=False)
    response.raise_for_status()
    json_obj = response.json()
    obj = extract_quokka_measurements(json_obj)
    return obj

def collect_measurements(program: str, shots: int, quokka_name: str):
    try:
        return send_to_quokka(program, count = shots, my_quokka=quokka_name)
    except Exception as exc:
        raise RuntimeError(f"Quokka request failed: {exc}") from exc


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
    fig.set_facecolor("#d7d7d7")
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

    fig.tight_layout()
    fig.savefig(str(plot_path), bbox_inches="tight")
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
    start_label: str,
    target_label: str | None,
    target_state: np.ndarray | None,
    gates: List[str],
    state: np.ndarray,
    measurements: List[int],
    qasm_path: Path,
    plot_path: Path,
):
    counts = measurement_counts(measurements)
    total = len(measurements)
    success = None if target_state is None else states_match_up_to_global_phase(state, target_state)

    print()
    print("Round result")
    print(f"Mode: {mode_name}")
    print(f"Starting colour: {start_label}")
    if target_label is None:
        print("Target colour: none in playground mode")
    else:
        print(f"Target colour: {target_label}")
    print(f"Gates used: {' '.join(gates) if gates else '(none)'}")
    print(f"QASM file: {qasm_path}")
    print(f"Measurement plot: {plot_path}")
    print()
    # print(f"Black: {counts['0']:>4}/{total} = {counts['0'] / total:6.1%}")
    # print(f"White: {counts['1']:>4}/{total} = {counts['1'] / total:6.1%}")
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
    start_state: np.ndarray,
    start_label: str,
    target_state: np.ndarray | None,
    target_label: str | None,
    gate_text: str,
    shots: int,
    quokka_name: str,
    qasm_path: Path,
    allowed_gates: List[str] | None = None,
    max_gates: int | None = None,
):
    gates = parse_gate_input(gate_text)
    validate_gate_sequence(gates, allowed_gates=allowed_gates, max_gates=max_gates)
    state = apply_gates(gates, start_state=start_state)
    program = build_qasm(gates, start_state=start_state)
    write_qasm(program, qasm_path)
    measurements= collect_measurements(program, shots, quokka_name)
    plot_path = plot_measurements(measurements, default_plot_path(qasm_path))
    print_round_report(
        mode_name,
        start_label,
        target_label,
        target_state,
        gates,
        state,
        measurements,
        qasm_path,
        plot_path
    )


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


def play_level(level: dict, args):
    while True:
        print()
        print(level["title"])
        print(level["description"])
        print(f"Maximum gates: {level['max_gates']}")
        print_gate_help(level["allowed_gates"])
        print()

        gate_text = prompt_gate_text(level["allowed_gates"], level["max_gates"])
        run_round(
            mode_name=level["title"],
            start_state=INITIAL_STATE,
            start_label="black",
            target_state=level["target_state"],
            target_label=level["target_label"],
            gate_text=gate_text,
            shots=args.shots,
            quokka_name=args.quokka,
            qasm_path=Path(args.output),
            allowed_gates=level["allowed_gates"],
            max_gates=level["max_gates"],
        )
        next_action = prompt_next_action()
        if next_action == "replay":
            continue
        if next_action == "menu":
            print()
            return "menu"
        return "quit"


def interactive_game(args):
    print("Quantum Colour Mixer - Prototype")
    print("Choose a level and use quantum gates to reach the target colour.")
    print()

    while True:
        level_key = prompt_level()
        level = LEVELS[level_key]
        if play_level(level, args) == "quit":
            return


def build_parser():
    parser = argparse.ArgumentParser(description="Colour mixer the quantum way.")
    parser.add_argument("--level", help="Start the game using a level file path, for example: levels/lv3.txt.")
    parser.add_argument("--start", default="black", help="Starting colour for custom mode: black, white, gray, or a ratio like 70/30.")
    parser.add_argument("--target", help="Desired final colour for custom mode: black, white, gray, or a ratio like 70/30.")
    parser.add_argument("--gates", default="", help="Gate list to apply to your starting colour, for example: 'X' or 'H / X'.")
    parser.add_argument("--shots", type=int, default=500, help="Number of measurements to request.")
    parser.add_argument("--quokka", default="quokka1", help="Quokka name, for example quokka1.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the generated QASM file.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.shots <= 0:
        parser.error("--shots must be a positive integer.")

    if args.level is not None:
        if args.target is not None:
            parser.error("--level cannot be used together with --target.")
        if args.start != "black":
            parser.error("--level uses the level's own starting colour, so do not pass --start.")
        if args.gates:
            parser.error("--level starts the interactive level flow, so do not pass --gates.")

        try:
            level = parse_level_file(Path(args.level))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

        print("Quantum Colour Mixer - Prototype")
        print("Loaded level from file.")
        play_level(level, args)
        return

    if args.target is None:
        interactive_game(args)
        return

    try:
        start_state, start_label = parse_colour_state(args.start)
        target_state, target_label = parse_colour_state(args.target)
    except ValueError as exc:
        parser.error(str(exc))

    run_round(
        mode_name="Custom Mode",
        start_state=start_state,
        start_label=start_label,
        target_state=target_state,
        target_label=target_label,
        gate_text=args.gates,
        shots=args.shots,
        quokka_name=args.quokka,
        qasm_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
