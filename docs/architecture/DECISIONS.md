# Architecture decisions

TinyTransformer was reviewed against
[`architecture-standards`](https://github.com/konradcinkusz/architecture-standards)
(the 15-principle reference architecture, plus the SERVICE-API-PATTERNS, FRONTEND-BFF,
REPO-BASELINE, TESTING-STRATEGY, README-BADGES and OPEN-SOURCE-RELEASE guides). This
document records what was applied, what was deliberately not, and why - per that
repo's own P14 ("a document that says we considered X and rejected it because Y is
worth more than a document that lists commands") and its REPO-BASELINE §7 carve-out for
repos that "genuinely aren't meant to conform to this constitution."

## Why the estate architecture doesn't apply wholesale

`architecture-standards`' reference architecture (P1-P15) was extracted from two
production, multi-service .NET Aspire systems running on Fly.io, with user accounts,
billing, and cross-service auth. TinyTransformer is a single-service educational demo
with no users, no accounts, and nothing to persist. Applying the estate's architecture
wholesale would be cargo-culting: an Aspire AppHost to orchestrate one container,
JWT/JWKS for an API nobody logs into, a Next.js BFF to keep a token away from a browser
that is never issued one. For that reason this repo does **not** declare
`architecture-core` adoption in `.claude/settings.json` (REPO-BASELINE §7) - it is
exactly the kind of repo that guide's carve-out describes.

## What was applied anyway, adapted to scale

| Principle / guide | Applied as |
|---|---|
| P6 - container per service | One multi-stage `Dockerfile`, listens on `:8080`, non-root `$APP_UID`, runtime image major version matches the `net8.0` TFM |
| P8 - optional deps degrade / zero-cloud-creds start | No external integrations exist, but the same spirit holds: `git clone && dotnet run` (or `docker compose up --build`) works with zero configuration |
| P9 - `Program.cs` is a manifest | `TinyTransformer.Api/Program.cs` wires services, rate limiting and endpoints in ~15 lines; the use-case lives in `Services/EncoderDemoService.cs`, endpoint binding in `Endpoints/TransformerEndpoints.cs` |
| P10 - interface + registration, not inheritance | `ILayer` stays the one extension point for Core layers; `TransformerEncoderBlock` composes concrete layers rather than deriving from a base class |
| P13 / TESTING-STRATEGY | A unit test file per Core layer, including the two that had none before this change (`TransformerEncoderBlock`, plus the new `PositionalEncoding` / `CharTokenizer`); `WebApplicationFactory`-based integration tests for every API endpoint, including validation and rate-limit edge cases |
| SERVICE-API-PATTERNS §1 - rate limiting | Fixed-window, partitioned by client IP (the guide's documented fallback when there's no authenticated user id to partition by), uniform `{ error, retryAfter }` 429 body |
| SERVICE-API-PATTERNS §3 - validation | Every numeric knob and the input text length are clamped server-side with field-level errors; the bound exists because attention is O(n²) per request, not as a style choice |
| SERVICE-API-PATTERNS §2 - endpoint organization | One `/api` route group, not the public/auth/admin triad - there is exactly one trust level here (see below) |
| REPO-BASELINE | `.editorconfig`, central package versions (`Directory.Packages.props`), exclusion-based `.dockerignore`, `CODEOWNERS`, Dependabot, PR/issue templates, CI (build+test, Docker build, CodeQL, gitleaks) |
| README-BADGES | Header badge row (license / maintained / issues / PRs / CI); no footer block - this isn't a showcase repo soliciting sponsorship or follows |
| OPEN-SOURCE-RELEASE | `LICENSE.txt` fixed from a `[year] [fullname]` placeholder before anything else (§3); README opens with what/why in two sentences and a quick start that runs end to end with zero unwritten prerequisites (§4) |
| P14 - record reasoning | This document |

## Specific deviations, and why

**No authentication, no JWT/JWKS (P5).** There is nothing to protect: no accounts, no
user data, no persistence. Adding auth here would mean inventing a login just to have
something to secure, which cuts against P5's actual intent. If this ever grows a
"save your run" feature, that's the trigger to revisit it - not before.

**No Aspire AppHost, no Fly.io (P1, P7).** Both exist to orchestrate *multiple*
services and a platform topology. TinyTransformer is one container; `dotnet run` and
`docker compose up --build` already satisfy P1's "one command" test without an
orchestrator to maintain. The Dockerfile is the portable artifact (P6) - any container
platform can run it unchanged, so picking one isn't this repo's decision to make.

**No Next.js/BFF frontend ([`FRONTEND-BFF.md`](https://github.com/konradcinkusz/architecture-standards/blob/main/docs/guides/FRONTEND-BFF.md)).**
That guide's entire model exists to keep a bearer token off the browser. There is no
token here: the frontend is static HTML/CSS/vanilla JS served from the API's own
`wwwroot`, same origin, no cookies, no proxy layer, no build step. Introducing Next.js
would add a second language and package manager, plus a BFF proxy, to guard a secret
that doesn't exist.

**No database, no migrations (P3, P4).** Every request is stateless: text in, one
forward pass through freshly-initialized (untrained) weights, JSON out. Nothing to
persist between requests.

**Single-head attention, no training loop.** Pre-existing properties of the Core
library, not decisions made for this change - see the README's "What this is not."
`SelfAttention` implements one attention head; there is no backprop/optimizer, so every
run uses random, untrained weights. Multi-head attention and autodiff are both
substantial, separate undertakings - out of scope for "wire up an API and frontend
around what already exists here."

**`docs/diagram.png` and `docs/diagram_solution.png` predate this change and describe
a planned, not-implemented direction** - `diagram_solution.png` in particular sketches
an `Autograd` package (`Tensor`, `TensorOps`, `Backward()`, cross-entropy) and
`*Auto`-suffixed layer classes that do not exist in the codebase; only `FeedForwardAuto`
was ever actually built, and it has no autograd behavior despite the name. These are
left in place as historical design sketches (deleting a contributor's design thinking
isn't this change's call to make) but are now labeled in the README as a **concept
sketch for a possible future training path, not current behavior** - the corollary
to P14 is that undocumented-as-aspirational diagrams are exactly as misleading as a
stale README describing a system that no longer exists.

**FluentAssertions 8.x licensing.** The test projects use FluentAssertions under the
Xceed Community License (free for non-commercial use, which this project is). This
predates this change - the original repository already pinned FluentAssertions 8.6.0.
Noted here because a downstream commercial fork of the test projects would need to
either obtain a license or replace it (e.g. with `Assert`/Shouldly).
