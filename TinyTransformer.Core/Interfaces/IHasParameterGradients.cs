namespace TinyTransformer.Core.Interfaces;

// A layer with learnable parameters whose gradients accumulate during
// Backward() and can be applied by an optimizer. Kept separate from
// IDifferentiableLayer since not every differentiable layer has parameters
// (ReLU doesn't).
public interface IHasParameterGradients
{
    // Applies one plain SGD step (parameter -= learningRate * gradient) to
    // every learnable parameter, using the gradients accumulated by the most
    // recent Backward() call, then clears them so a stale gradient can't be
    // silently reapplied. There is deliberately no separate "read gradients"
    // step, and no support for any optimizer beyond plain SGD -
    // ROADMAP.md Phase 2 commits only to "forward + backward pass + SGD
    // update," not a general optimizer abstraction nobody has asked for yet.
    void ApplyGradients(float learningRate);
}
