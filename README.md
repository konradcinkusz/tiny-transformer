<a name="readme-top"></a>

# TinyTransformer

[![Ask Me Anything](https://flat.badgen.net/static/Ask%20me/anything?icon=github&color=black&scale=1.01)](https://github.com/konradcinkusz "Ask me anything")
[![GitHub license](https://flat.badgen.net/github/license/konradcinkusz/tiny-transformer?icon=github&color=black&scale=1.01)](https://github.com/konradcinkusz/tiny-transformer/blob/master/LICENSE.txt "GitHub license")
[![Maintained](https://flat.badgen.net/static/Maintained/yes?icon=github&color=black&scale=1.01)](https://github.com/konradcinkusz/tiny-transformer/commits/master "Maintained")
[![GitHub issues](https://flat.badgen.net/github/issues/konradcinkusz/tiny-transformer?icon=github&color=black&scale=1.01)](https://github.com/konradcinkusz/tiny-transformer/issues "GitHub issues")
[![GitHub pull requests](https://flat.badgen.net/github/prs/konradcinkusz/tiny-transformer?icon=github&color=black&scale=1.01)](https://github.com/konradcinkusz/tiny-transformer/pulls "GitHub pull requests")
[![CI](https://github.com/konradcinkusz/tiny-transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/konradcinkusz/tiny-transformer/actions/workflows/ci.yml "CI")

TinyTransformer is a transformer encoder block implemented from scratch in C#, with no
ML/tensor framework anywhere in the dependency graph. It exists to make the mechanics of
"Attention Is All You Need" legible - every matrix multiply, softmax and residual
connection is code you can read top to bottom - and it ships as a live, runnable demo:
an ASP.NET Core API wraps the model, and a small browser UI runs your text through it and
visualizes what comes out at each stage.

![TinyTransformer demo: text input, tokens, and a token-embedding heatmap](docs/screenshot.png)

## What this is (and is not)

**This is:** a hand-written encoder block - embedding lookup, sinusoidal positional
encoding, single-head self-attention, residual connections, LayerNorm, and a
feed-forward sublayer - runnable from the command line, a REST API, or a browser, with
a unit test for every layer.

**This is not:** a trained language model. There is no backpropagation, no optimizer,
and no training loop anywhere in this repository - every run initializes its weights
from a random seed and performs a single forward pass. The output is real and correctly
computed, but it reflects what an *untrained* transformer computes, not learned
language behavior. Two smaller, related honesty notes: tokenization is a simple
character-level scheme (not BPE/subword - there's no pretrained vocabulary to load),
and attention is single-head, not multi-head. See
[`docs/architecture/DECISIONS.md`](docs/architecture/DECISIONS.md) for the reasoning
behind these and other scope choices, including two design diagrams under `docs/` that
sketch a possible future autograd/training path that was never built - they're kept for
historical context but are **not** a description of current behavior.

## Quick start

Zero cloud accounts, zero API keys, zero unwritten prerequisites - pick one:

### Option A - Docker (recommended, matches production)

```bash
git clone https://github.com/konradcinkusz/tiny-transformer.git
cd tiny-transformer
docker compose up --build
```

Open **http://localhost:8080**.

### Option B - .NET SDK directly

Requires the [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).

```bash
git clone https://github.com/konradcinkusz/tiny-transformer.git
cd tiny-transformer
dotnet run --project TinyTransformer.Api
```

Open **http://localhost:8080** (the port is fixed in `Properties/launchSettings.json`
so it matches the Docker path above).

### Run the tests

```bash
dotnet test
```

### Run the console demo

A minimal, non-HTTP walkthrough of the same pipeline - useful for stepping through in a
debugger:

```bash
dotnet run --project TinyTransformer.ConsoleApp
```

## How it works

Type a sentence, hit "Run encoder", and the page sends it to `POST /api/encode`, which:

1. **Tokenizes** the text character-by-character (`CharTokenizer`) - each distinct
   character seen gets the next free token id, in order of first appearance.
2. **Looks up embeddings** for each token id (`Embedding`) - a random vector per id,
   since nothing here is trained.
3. **Adds positional encoding** (`PositionalEncoding`, the classic sinusoidal scheme) -
   without this step, self-attention is provably permutation-*equivariant* (see
   `SelfAttentionTests`), meaning token order carries no information at all.
4. **Runs one `TransformerEncoderBlock`**: self-attention → residual + LayerNorm →
   feed-forward → residual + LayerNorm.
5. Returns the embeddings, the positional encoding table, the attention weights, and the
   final output - all rendered as heatmaps in the browser.

The same seed always reproduces the same "model" (weights) and therefore the same
output, so a response is replayable by resending it with its own `seed` value.

## API reference

Interactive docs (Swagger UI) are served at **`/swagger`** whenever the app is running.

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Liveness check. |
| `/api/encode` | `POST` | Run text through the encoder block. See below. |

<details>
<summary><code>POST /api/encode</code> request/response</summary>

Every field except `text` is optional:

```json
{
  "text": "the cat sat on the mat",
  "dModel": 16,
  "dK": 16,
  "ffHidden": 32,
  "numHeads": 1,
  "numLayers": 1,
  "seed": 42
}
```

| Field | Range | Default |
|---|---|---|
| `text` | 1-64 characters | *(required)* |
| `dModel` | 4-64 | 16 |
| `dK` | 2-64 | 16 |
| `ffHidden` | 4-256 | 32 |
| `numHeads` | 1-8 | 1 |
| `numLayers` | 1-6 | 1 |
| `seed` | any integer | a fresh random value, returned in the response |

```json
{
  "tokens": ["t", "h", "e", "␣", "c", "a", "t"],
  "tokenIds": [0, 1, 2, 3, 4, 5, 0],
  "config": { "dModel": 16, "dK": 16, "ffHidden": 32, "numHeads": 1, "numLayers": 1, "seed": 42, "sequenceLength": 7, "vocabSize": 6 },
  "embeddings": [[...]],
  "positionalEncoding": [[...]],
  "attentionWeights": [[...]],
  "attentionWeightsPerLayer": [[[[...]]]],
  "encoderOutput": [[...]]
}
```

`attentionWeights` is always `[sequenceLength x sequenceLength]` regardless of
`numHeads`/`numLayers`: it's the last encoder block's attention, averaged
across heads if there is more than one (see
`TransformerEncoderBlock.ForwardWithAttention`). `attentionWeightsPerLayer` is
the full, unaveraged detail behind it - indexed
`[layer][head][sequenceLength][sequenceLength]` - for clients (like the
frontend's layer/head selector) that want to inspect an individual layer or
head; `attentionWeights` is exactly the average of `attentionWeightsPerLayer`'s
last entry.

Invalid input returns `400` with an RFC 9110 validation-problem body
(`{ "errors": { "text": ["Text is required."] } }`); exceeding the rate limit (30
requests/minute per client) returns `429` with `{ "error": "...", "retryAfter": <seconds> }`.

</details>

## Project layout

```
TinyTransformer.Core/          the model - Embedding, PositionalEncoding, SelfAttention,
                                LayerNorm, FeedForwardAuto, TransformerEncoderBlock, MathOps
TinyTransformer.Api/           ASP.NET Core minimal API + static frontend (wwwroot/)
TinyTransformer.ConsoleApp/    command-line walkthrough of the same pipeline
TinyTransformer.Tests/         unit tests for TinyTransformer.Core
TinyTransformer.Api.Tests/     integration tests for TinyTransformer.Api (WebApplicationFactory)
docs/architecture/             design decisions and scope notes
```

## Architecture & design decisions

This repository was reviewed against
[`konradcinkusz/architecture-standards`](https://github.com/konradcinkusz/architecture-standards),
a cross-repo reference architecture. Most of that architecture targets multi-service
systems with user accounts and cloud deployment, which doesn't describe this project -
[`docs/architecture/DECISIONS.md`](docs/architecture/DECISIONS.md) records exactly what
was applied here, what was deliberately left out, and why, rather than either
cargo-culting the whole thing or applying nothing and staying silent about it.

## Contributing

Issues and PRs are welcome - see the templates under `.github/`. CI (`dotnet test`,
Docker build, CodeQL, and a secret scan) runs on every PR.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
