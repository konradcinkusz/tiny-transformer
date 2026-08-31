namespace TinyTransformer.Core.Layers;

public class FeedForwardAuto : IDifferentiableLayer, IHasParameterGradients
{
    private readonly Linear _l1;
    private readonly ReLU _relu;
    private readonly Linear _l2;

    public FeedForwardAuto(int dModel, int hidden, Random rnd)
    {
        _l1 = new Linear(dModel, hidden, rnd);
        _relu = new ReLU();
        _l2 = new Linear(hidden, dModel, rnd);
    }

    // Deterministic - for reconstructing a previously-trained feed-forward
    // block (see TinyTransformer.Core.Models.TinyTransformerModel).
    public FeedForwardAuto(Linear l1, Linear l2)
    {
        _l1 = l1;
        _relu = new ReLU();
        _l2 = l2;
    }

    // Read-only accessors for persistence.
    public Linear L1 => _l1;
    public Linear L2 => _l2;

    public float[,] Forward(float[,] X)
    {
        var h = _l1.Forward(X);
        h = _relu.Forward(h); //non-linearity
        return _l2.Forward(h);
    }

    // Reverse of Forward: _l2 -> _relu -> _l1. _l1/_l2 accumulate their own
    // dW/db as a side effect of their own Backward call, same as everywhere
    // else this pattern is used (SelfAttention's Wq/Wk/Wv/Wo).
    public float[,] Backward(float[,] dOut)
    {
        var dHiddenAfterRelu = _l2.Backward(dOut);
        var dHiddenBeforeRelu = _relu.Backward(dHiddenAfterRelu);
        return _l1.Backward(dHiddenBeforeRelu);
    }

    public void ApplyGradients(float learningRate)
    {
        _l1.ApplyGradients(learningRate);
        _l2.ApplyGradients(learningRate);
    }
}
