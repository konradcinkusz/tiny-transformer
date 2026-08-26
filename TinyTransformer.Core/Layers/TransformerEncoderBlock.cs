namespace TinyTransformer.Core.Layers;

public class TransformerEncoderBlock : ILayer
{
    // Concrete type (not ILayer) because ForwardWithAttention below needs the
    // attention weights SelfAttention computes internally, which the plain
    // ILayer.Forward contract does not expose.
    private readonly SelfAttention _selfAttention;
    private readonly ILayer _feedForward;
    private readonly ILayer _ln1;
    private readonly ILayer _ln2;

    public TransformerEncoderBlock(int dModel, int dK, int ffHidden, Random rnd)
    {
        _selfAttention = new SelfAttention(dModel, dK, rnd);
        _feedForward = new FeedForwardAuto(dModel, ffHidden, rnd);
        _ln1 = new LayerNorm(dModel);
        _ln2 = new LayerNorm(dModel);
    }

    public float[,] Forward(float[,] X) => ForwardWithAttention(X).Output;

    // Same computation as Forward, but also returns the self-attention
    // weights - useful for callers that want to visualize what the block did.
    public (float[,] Output, float[,] AttentionWeights) ForwardWithAttention(float[,] X)
    {
        var (attentionOutput, attentionWeights) = _selfAttention.ForwardWithAttention(X);
        var x1 = _ln1.Forward(MathOps.Add(X, attentionOutput));
        var ffOutput = _feedForward.Forward(x1);
        var output = _ln2.Forward(MathOps.Add(x1, ffOutput));
        return (output, attentionWeights);
    }
}
