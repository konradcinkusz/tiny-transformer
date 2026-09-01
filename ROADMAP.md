# Roadmap

This is the source of truth for TinyTransformer's next phase of work: what's next, in
what order, and why. It exists because GitHub Milestones could not be created from the
session that wrote this document (no `gh` CLI and no milestone-creation tool were
available — see "A note on tooling" at the end). Each phase below stands in for a
milestone; each issue references its phase by name in its body.

## Where the project stands (2026-09-01)

All four phases below are complete except one item explicitly blocked on tooling (see
Phase 4). TinyTransformer is a from-scratch, dependency-free (beyond xUnit/
FluentAssertions/Swashbuckle) transformer **encoder** in C#, with multi-head attention,
multi-layer stacking, a real (if deliberately toy) hand-derived backward pass and SGD
training loop, a save/load format for trained weights, an ASP.NET Core API and browser
demo exposing all of it (including a "Weights: Random / Trained" toggle), tests, CI,
and docs (see `README.md` and `docs/architecture/DECISIONS.md`) - kept up to date as
each phase landed, not just written once at the start.

**The one item this roadmap could not finish:** the repository's `description` and
`topics` fields still don't match the project's actual scope (they claim
"encoder–decoder", which was never built - see `docs/architecture/DECISIONS.md`'s
"Encoder-only by design" entry). Updating them needs the repo-settings API, which no
tool available across this roadmap's execution could reach (no `gh` CLI, no
repo-admin-scoped API tool) - see Phase 4 and the comment on its tracking issue for the
exact values to apply.

<details>
<summary>Original "where the project stands" snapshot (2026-08-31, before any phase below started)</summary>

TinyTransformer is a from-scratch, dependency-free (beyond xUnit/FluentAssertions/
Swashbuckle) transformer **encoder** in C#, with an ASP.NET Core API, a browser demo,
tests, CI, and docs (see `README.md` and `docs/architecture/DECISIONS.md`). It is a
forward-pass-only implementation: there is no training loop, no backpropagation, and
every run initializes random, untrained weights. Self-attention is single-head, and
exactly one encoder block is run per request — there is no multi-layer stack.

**A concrete gap worth naming up front:** the repository's own description says
*"encoder–decoder, multi-head self-attention... "*. The actual code is encoder-only
with single-head attention. Phase 1 below closes part of that gap (multi-head,
multi-layer); the rest is a documentation/description decision, not a build task — see
Phase 1's third issue.

</details>

## Phases

### Phase 1 — Model Fidelity ✅ done (2026-08-31, target was 2026-09-21)

Close the gap between what the project claims to be and what it does, on the parts
worth building rather than just rewording.

- [x] Implement multi-head self-attention in `TinyTransformer.Core` (#18)
- [x] Support stacking multiple `TransformerEncoderBlock`s (configurable depth) (#19)
- [x] Reconcile the repo description/README with actual model scope, once the above
  land (decide and document: encoder-only by design, no decoder — update the GitHub
  repo description/topics and README accordingly) (#20 - the README/DECISIONS.md half
  landed; the GitHub repo description/topics half is still blocked, see Phase 4)

### Phase 2 — Learning & Training ✅ done (2026-08-31, target was 2026-10-12)

Give the project an actual training story. This is the biggest lift on the roadmap —
hand-written backprop through attention is real numerical work, not a weekend task —
and it's split into the smallest pieces that still ship independently, per issue.

- [x] Add gradient-carrying infrastructure to Core (what a layer's `Backward()` needs,
  and where gradients accumulate) (#21, #22)
- [x] Implement backward pass + gradient tests for `Linear` and `LayerNorm` (#22)
- [x] Implement backward pass + gradient tests for Softmax / cross-entropy loss (#23)
- [x] Implement backward pass + gradient tests for `SelfAttention` (#24)
- [x] Wire a full training loop (forward + backward + SGD) with a toy overfitting
  example in `TinyTransformer.ConsoleApp` (#25)
- [x] Add model weight save/load (JSON) so a trained model can be persisted and
  reloaded instead of always starting from random weights (#26)

### Phase 3 — Demo & API Polish ✅ done (2026-08-31, target was 2026-11-02)

Surface Phases 1 and 2 through the actual product surface (the API and the browser
demo), not just the console app.

- [x] Expose multi-head/multi-layer configuration through `POST /api/encode` (#27)
- [x] Update the frontend to visualize multiple layers and attention heads (not just
  one block's output) (#28)
- [x] Add a "train a tiny model" demo path connecting Phase 2's training loop to the UI
  (#29)

### Phase 4 — Community & Release Readiness (target: 2026-11-16)

- [x] Add `CONTRIBUTING.md`: dev setup and how to add a new `ILayer` implementation
  (#30)
- [x] Update README/`DECISIONS.md` sections left stale by Phases 2-3 (#45 - found
  during this phase's own release-readiness pass, not originally listed above)
- [ ] Sync the GitHub repo description and topics with the project's actual scope -
  **blocked**: needs a human with repo admin access or a `gh`-equipped session, neither
  of which any session executing this roadmap had; see the comment on #31 for the
  exact description/topics to apply

## Sequencing notes

- Within a phase, issues are unordered unless one explicitly says otherwise, except
  that Phase 2's issues are listed in dependency order (each one builds on the last).
- Phase 3 depends on Phase 1 (multi-head/multi-layer) and Phase 2 (a trained model to
  demo) both being done — don't start Phase 3 issues early.
- Due dates above are placeholders, not commitments: this project's actual velocity so
  far has been bursty and AI-assisted (most of the existing feature work landed in a
  single session), not a steady weekly cadence. Adjust freely.

## A note on tooling

The session that wrote this roadmap had GitHub issue read/write access but no way to
create Milestone objects (no `gh` CLI, no milestone-creation tool, no generic REST/GraphQL
passthrough). Rather than silently skip milestones or fake having created them, it wrote
this file instead and named each issue's phase in its body. If you'd like real GitHub
Milestones, creating the four above (with these due dates) and re-assigning the existing
issues to them is a five-minute manual task, or a session with `gh` access can do it via
`gh api repos/{owner}/{repo}/milestones`.
