# Colour Mixer, a quantum game

This is a game designed to introduce basic quantum operations and measurements. Players are given a set of basic colours (aka the bases set), a starting colour (initial state), and a handful of mixer magics (quatum gates) that allow them to perform state flips, rotations, superpositions and entanglements. The aim is to generate the target colour using any combination of magics.

## Play on my webapp

colour.tonina.cc

## AI disclaimer

OpenAI codex was used to write part (all) of the functions, and was involved in debugging. All (no) code is reviewed by human.

The game mechanism and the levels are purely done by human.

## Playing around CMYK colours

Don't panic. We are using two qubits to give you any combination of four basis colours:
- `|00>`: cyan
- `|01>`: magenta
- `|10>`: yellow
- `|11>`: black

When the game starts, you can choose from the preset levels, or playground:

- `Level 1 - Mix Cyan`: a 50-50 cyan/magenta mix from one Hadamard.
- `Level 2 - Blendy Blend`: an even four-colour mix from independent superpositions.
- `Level 3 - Forbidden Colours`: a cyan/black blend that introduces entanglement.
- `Level 4 - Weighted Purple`: an approximate 75/25 cyan/magenta blend using `RY`.
- `Level 5 - Swap Around`: a weighted cyan/yellow blend that teaches `SWAP`.
- `Playground`: build your own circuit with the available single-qubit and two-qubit gates.

The levels are defined as JSON files in [levels](./levels).

In each level, you will have access to ristricted magics, and can only input no more than a certain number of magics.

There is no "standard answer" - use any combination of magics within the constraints to create target colour and you will get a YAY!

## Magics

Magics are single-qubit gates and two-qubit gates. Here we provide a simple yet powerful set of "universal gates": you can theoretically reach any colour mix with these gates.

### Single-qubit gates:
- `X`: state flip
- `Y`: phase flip
- `Z`: state & phase flip
- `RX`, `RY`, `RZ`: rotations around `x`, `y`, or `z` axis

Single-qubit gates default to qubit 0. For rotation gates with two arguments, the first argument must be the qubit number and the second must be the angle, for example `RX(1, pi/2)`, `RY(1, pi/3)`, or `RZ(1, -pi/4)`

### Two-qubit gates:
- `CNOT`: controlled-NOT
- `SWAP`: ...swap.
- `CPHASE`: controlled phase flip

Use syntax like `CNOT(0,1)`, `SWAP(0,1)`, and `CPHASE(0,1)`.

## Playground

There's no target here. Default starting colour is black. Apply any magic available to create your own superposition! Maximum 10 magics. Playground supports single-qubit gates and the two-qubit gates `CNOT`, `SWAP`, and `CPHASE`.

## Level source

Level definitions live in [levels](./levels) as one JSON file per level. The webapp imports those files through [src/levels.ts](./src/levels.ts).

## Example gate sequences

```bash
H(1)
```

```bash
H(0) / CNOT(0,1)
```

```bash
RY(1, pi/3)
```

```bash
RY(1, pi/3) / SWAP(0,1)
```

## Notes

- Supported gates: `I` as a game-level no-op, plus Quokka-backed `X`, `Y`, `Z`, `H`, `RX`, `RY`, `RZ`, `CNOT`, `SWAP`, and `CPHASE`.
- Level success is checked against the target colour probabilities, not exact quantum phase. That keeps colour-mixing levels intuitive and allows some tolerance where appropriate.
- Measurement plots draw every shot as a semi-transparent coloured circle, so overlapping circles visually mimic a mixed colour.
- Quokka access is required for measurements.

## DIY the webapp

The primary version is now a React + Vite webapp. It runs the game UI and local target checking in the browser, then sends generated OpenQASM to Quokka through a same-origin `/api/quokka` Cloudflare Pages Function. The proxy is required because the Quokka endpoint is not browser-CORS-enabled.

Install dependencies:

```bash
pnpm install
```

Run locally with Cloudflare Wrangler so `/api/quokka` is available:

```bash
# Build the app and run Cloudflare Pages locally
pnpm dev
```

Run locally with hot-reloading for development
```bash
# In separate terminals

# Rebuild on file changes
pnpm build:watch

# Run Cloudflare Pages locally
pnpm wrangler:dev
```

For frontend-only development without Quokka calls:

```bash
pnpm dev:vite
```

Build and test:

```bash
pnpm build
pnpm test
```

Architecture notes live in [architecture](./architecture).

## Hosting On Cloudflare Pages

The default Cloudflare Pages project name is `quantum-colour`, configured in [wrangler.jsonc](./wrangler.jsonc). The build output directory is `dist`, and the Quokka proxy is the Pages Function at [functions/api/quokka.ts](./functions/api/quokka.ts).

If you deploy your own copy and need a different Pages project name, update `name` in `wrangler.jsonc` before creating or deploying the project:

```jsonc
{
  "name": "your-pages-project-name",
  "pages_build_output_dir": "dist"
}
```

One-time Cloudflare setup:

1. Log in to Cloudflare from the CLI:

```bash
pnpm wrangler login
```

2. Create the Pages project if it does not already exist:

```bash
pnpm wrangler pages project create quantum-colour --production-branch cmyk-web-app
```

3. Deploy the app:

```bash
pnpm deploy:app
```

The deploy script runs `pnpm build` and then `wrangler pages deploy dist`. If you need to pass a branch explicitly, use:

```bash
pnpm build
pnpm wrangler pages deploy dist --branch main
```

Custom domain setup:

1. In Cloudflare, open Workers & Pages, then the `quantum-colour` Pages project.
2. Go to Custom domains and add `quantum-color.example.com`.
3. If `example.com` DNS is managed by Cloudflare, accept the DNS record Cloudflare creates.
4. If DNS is managed elsewhere, manually add the CNAME record Cloudflare shows for `quantum-color.example.com`.
5. Wait for Cloudflare to issue the certificate and mark the domain active.
