namespace TinyTransformer.Core.Interfaces;

// A layer that can run backpropagation: given the gradient of the loss with
// respect to this layer's output, compute the gradient with respect to its
// input. This is the foundation ROADMAP.md Phase 2 builds a training loop on
// top of.
//
// Layers with learnable parameters (Linear, LayerNorm, ...) will also need a
// way to expose their own parameter gradients (dW, db, ...) so an optimizer
// can update them - deliberately not designed here. Speculatively shaping
// that now, before a real implementer exists to prove the shape right, is
// exactly the kind of premature abstraction this codebase avoids elsewhere;
// the issue that adds the first such layer is where that decision belongs.
public interface IDifferentiableLayer : ILayer
{
    // Must be called after Forward(X), with the gradient of the loss with
    // respect to that call's output, and returns the gradient with respect
    // to X. Implementations that need state from Forward (e.g. which inputs
    // were positive) must cache it themselves.
    float[,] Backward(float[,] dOut);
}
