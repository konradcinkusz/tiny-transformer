# Contributing to TinyTransformer

This is a guide for working *on* the codebase - if you just want to *use* the demo,
see the [README](README.md)'s Quick start instead. This file assumes you've read
[`docs/architecture/DECISIONS.md`](docs/architecture/DECISIONS.md), which explains why
the codebase is shaped the way it is; this file is about how to work within that shape.

## Local dev setup

You need the [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) - nothing
else (no Docker required for development, though `docker compose up --build` still
works if you'd rather not install the SDK locally).

```bash
git clone https://github.com/konradcinkusz/tiny-transformer.git
cd tiny-transformer
dotnet build
```

The solution (`TinyTransformer.sln`) has five projects:

| Project | What it is |
|---|---|
| `TinyTransformer.Core` | The actual transformer: layers, math, training, persistence. No HTTP, no I/O beyond save/load. |
| `TinyTransformer.Tests` | Unit tests for `Core` (`dotnet test`'s largest suite). |
| `TinyTransformer.Api` | The ASP.NET Core API + static frontend (`wwwroot/`) that wrap `Core` for the live demo. |
| `TinyTransformer.Api.Tests` | Integration tests for `Api`, booting the app in-process via `WebApplicationFactory`. |
| `TinyTransformer.ConsoleApp` | A minimal non-HTTP walkthrough of `Core`, useful for stepping through in a debugger. |

## Running the test suite

```bash
dotnet test
```

Before opening a PR, make sure this passes locally - CI runs the same command (plus a
Docker build, CodeQL, and a secret scan; see `.github/workflows/`) on every PR, and a
red build blocks review. If you touched `TinyTransformer.Api/wwwroot/`, also start the
app (`dotnet run --project TinyTransformer.Api`) and click through the change in a
browser - the test suite covers the API contract, not what the page looks like.

## Adding a new layer

`ILayer` (`TinyTransformer.Core/Interfaces/ILayer.cs`) is the one extension point for
Core layers - a single `Forward(float[,] X) -> float[,]` method. New layers implement
this interface directly and get composed into whatever needs them (see
`docs/architecture/DECISIONS.md`'s note on "interface + registration, not
inheritance" - there is no base class to derive from). If the layer needs to be
trainable, it should also implement `IDifferentiableLayer` (adds
`Backward(float[,] dOut) -> float[,]`) and, if it has learnable parameters,
`IHasParameterGradients` (adds `ApplyGradients(float learningRate)`).

The existing `ReLU` layer (`TinyTransformer.Core/Layers/ReLU.cs`) is the simplest
complete example to copy the pattern from - it has no learnable parameters, so it only
implements `ILayer` and `IDifferentiableLayer`. Walking through what a new layer needs,
using `ReLU` as the reference:

1. **Implement `Forward`.** Cache whatever `Backward` will need (`ReLU` caches its
   input, since the gradient depends on which entries were positive).
2. **If it should be trainable, implement `Backward`.** Derive the gradient by hand and
   sanity-check it against a matrix-calculus reference before trusting it.
3. **Add a unit test** in `TinyTransformer.Tests/LayersTests/`, following
   `ReLUTests.cs`'s pattern:
   - A `Forward` correctness test (compare against a known-correct reference
     computation, e.g. `MathOps` if one already exists there, or hand-computed
     expected values for a small fixed input).
   - A `Backward_MatchesNumericalGradient` test using `TestsBase.NumericalGradient`
     (central finite differences) - this is the project's standard way of catching a
     wrong-but-plausible-looking derivative; see any `*BackwardTests.cs` file for the
     pattern (e.g. `LayerNormBackwardTests.cs`, `LinearBackwardTests.cs`).
   - Edge cases specific to the layer (see `ReLUTests.Backward_ZerosOutGradientWherever_InputWasNonPositive`
     for an example of pinning down a specific numerical edge case, not just the
     general gradient-check).
4. **Wire it in wherever it's meant to be used** - e.g. composed into
   `FeedForwardAuto` or `TransformerEncoderBlock` if it's a building block for those,
   or added to `TinyTransformer.ConsoleApp/Program.cs`'s demo if it's meant to be
   independently visible.
5. If the layer has learnable parameters, add a deterministic constructor
   (`Linear(float[,] W, float[] b)` and `LayerNorm(float[] gamma, float[] beta)` are
   the existing examples) alongside the random-initializing one, and read-only
   accessors to its parameters (`Linear.Weights`/`Linear.Bias`) - this is what makes a
   layer usable from `TinyTransformer.Core.Models.TinyTransformerModel`'s save/load
   format, which every trainable layer needs to support (see
   `TinyTransformerModel.cs`'s `To*State`/`From*State` conversions for the pattern).

## Branches and PRs

Branch names are `<kind>/<short-description>` (`feature/...`, `fix/...`, `docs/...`,
`chore/...` - see recent history for examples). Reference the issue a PR closes with
`Closes #N` in the PR description so it closes automatically on merge. Keep PRs to one
logical change - this codebase's history is almost entirely one PR per issue, which
keeps review and `git blame` both readable.

See the README's [Contributing](README.md#contributing) section for the issue/PR
template pointers and what CI runs.
