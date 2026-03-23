# Colour Mixer, a quantum game

This is a game designed to introduce basic quantum operations and measurements. Players are given a set of basic colours (aka the bases set), a starting colour (initial state), and a handful of mixer magics (quatum gates) that allow them to perform state flips, rotations, superpositionsm and entanglements. The aim is to generate the target colour using any combination of magics.

## AI disclaimer

OpenAI codex was used to write part of the functions, and was involved in debugging. All code is reviewed by human.

The game mechanism and the levels are purely done by human.

## Prototype

This prototype is a one-qubit model with two basis states, `|0>` representing black and `|1>` representing white. Three levels are designed to introduce the three most basic magics.

When the game starts, it asks the player which mode to enter:

- `Level 1`: start from black, target black, at most 3 gates, using `I` and `X`.
- `Level 2`: start from black, target white, at most 3 gates, using `I` and `X`.
- `Level 3`: start from black, target gray, at most 3 gates, using `I`, `X`, and `H`.

The levels are defined in [levels](./levels).

In each level, player will have access to ristricted magics, and can only input no more than a certain number of magics.

There is no "standard answer" - use any combination of magics within the constraints to create target colour and you will get a YAY!

## Playground

There's no target here. Default starting colour is black. Apply any magic available to create your own superposition! Maximum 10 magics.

## Play the game

Interactive levels:

```bash
python quantum_game.py
```

Custom mode single round:

```bash
python quantum_game.py --start black --target white --gates "X" --shots 500
```

Custom mode with a superposition target:

```bash
python quantum_game.py --start black --target 50/50 --gates "H" --shots 500
```

Custom mode with a custom black/white superposition:

```bash
python quantum_game.py --start 70/30 --target 30/70 --gates "X" --shots 500
```
## Write your own level

Each level is just a .txt file. You can create your own level by editing the existing levels file.

Run the game from a file path to the level file:

```bash
python quantum_game.py --level levels/lv3.txt
```


## Notes

- Supported gates: `I`, `X`, `Y`, `Z`, `H`, `T`. (More gates coming in full version)
- Custom mode accepts `black`, `white`, `gray`, or a ratio like `70/30` for both `--start` and `--target`.
- Quokka access is required for measurements.
