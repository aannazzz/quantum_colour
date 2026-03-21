# One-Qubit Quantum Game Prototype

This prototype starts from the backend state `|0>`, presents that to the player as `black`, lets the player enter single-qubit gates, writes a QASM file, sends that circuit to Quokka for repeated measurements, and checks analytically whether the final state matches the target colour.

When the game starts, it asks the player which mode to enter:

- `Level 1`: start from black, target black, at most 3 gates, using `I` and `X`.
- `Level 2`: start from black, target white, at most 3 gates, using `I` and `X`.
- `Level 3`: start from black, target gray, at most 3 gates, using `I`, `X`, and `H`.
- `Playground`: no target check, up to 10 gates, using the full prototype gate set.

These modes are defined in separate text files under [levels](./levels), so adding a new level is just a matter of creating another `.txt` file with the same fields.

## Run the game

Interactive mode:

```bash
python3 quantum_game.py
```

Direct single round:

```bash
python3 quantum_game.py --target white --gates "X" --shots 200
```

Direct single round with a superposition target:

```bash
python3 quantum_game.py --target gray --gates "H" --shots 200
```

## What it prints

- The target colour and the gates the player chose.
- The path to the generated QASM file.
- The path to a measurement plot PNG.
- A measurement distribution over `black` and `white`.
- A colour strip where `.` means `black` and `o` means `white`.
- A final `YAY` or `NAY` based on the hidden analytic check, except in playground mode.

## Notes

- Supported gates: `I`, `X`, `Y`, `Z`, `H`, `S`, `T`.
- QASM is written to the prototype's `generated/player_circuit.qasm` by default.
- The measurement cloud is written to the prototype's `generated/player_circuit_measurements.png` by default.
- Black dots represent the black result, white dots represent the white result, and alpha blending lets overlaps appear gray.
- If Quokka is unavailable, the script can fall back to local sampling from the analytic probabilities unless you pass `--no-local-fallback`.
