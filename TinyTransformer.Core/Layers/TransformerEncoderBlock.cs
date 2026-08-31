namespace TinyTransformer.Core.Layers;

public class TransformerEncoderBlock : IDifferentiableLayer, IHasParameterGradients
{
    // Concrete types (not ILayer) because ForwardWithAttention needs the
    // attention weights MultiHeadSelfAttention computes internally, and
    // Backward/ApplyGradients need every sub-layer's own gradient support,
    // none of which the plain ILayer.Forward contract exposes.
    private readonly MultiHeadSelfAttention _selfAttention;
    private readonly FeedForwardAuto _feedForward;
    private readonly LayerNorm _ln1;
    private readonly LayerNorm _ln2;

    // numHeads defaults to 1 so every existing caller (TinyTransformer.Api,
    // TinyTransformer.ConsoleApp, and the tests that predate multi-head
    // support) keeps compiling and behaving identically - MultiHeadSelfAttention
    // with numHeads = 1 is bit-for-bit the same computation SelfAttention did.
    public TransformerEncoderBlock(int dModel, int dK, int ffHidden, Random rnd, int numHeads = 1)
    {
        _selfAttention = new MultiHeadSelfAttention(dModel, dK, numHeads, rnd);
        _feedForward = new FeedForwardAuto(dModel, ffHidden, rnd);
        _ln1 = new LayerNorm(dModel);
        _ln2 = new LayerNorm(dModel);
    }

    // Deterministic - for reconstructing a previously-trained block (see
    // TinyTransformer.Core.Models.TinyTransformerModel).
    public TransformerEncoderBlock(MultiHeadSelfAttention selfAttention, FeedForwardAuto feedForward, LayerNorm ln1, LayerNorm ln2)
    {
        _selfAttention = selfAttention;
        _feedForward = feedForward;
        _ln1 = ln1;
        _ln2 = ln2;
    }

    // Read-only accessors for persistence.
    public MultiHeadSelfAttention SelfAttention => _selfAttention;
    public FeedForwardAuto FeedForward => _feedForward;
    public LayerNorm Ln1 => _ln1;
    public LayerNorm Ln2 => _ln2;

    public float[,] Forward(float[,] X) => ForwardWithAttention(X).Output;

    // Same computation as Forward, but also returns the self-attention
    // weights - useful for callers that want to visualize what the block did.
    //
    // AttentionWeights is the *average* across heads when numHeads > 1, for
    // callers that only know how to render one attention heatmap (e.g.
    // TinyTransformer.ConsoleApp). Callers that want the full per-head
    // breakdown (e.g. the API/frontend, see ROADMAP.md Phase 3) should use
    // ForwardWithPerHeadAttention instead.
    public (float[,] Output, float[,] AttentionWeights) ForwardWithAttention(float[,] X)
    {
        var (output, attentionWeightsPerHead) = ForwardWithPerHeadAttention(X);
        return (output, MathOps.AverageAcrossHeads(attentionWeightsPerHead));
    }

    // Same computation as Forward, but also returns every head's own
    // attention weights, unaveraged - one [T x T] matrix per head, in head
    // order (mirrors MultiHeadSelfAttention.ForwardWithAttention).
    public (float[,] Output, float[][,] PerHeadAttentionWeights) ForwardWithPerHeadAttention(float[,] X)
    {
        var (attentionOutput, attentionWeightsPerHead) = _selfAttention.ForwardWithAttention(X);
        var x1 = _ln1.Forward(MathOps.Add(X, attentionOutput));
        var ffOutput = _feedForward.Forward(x1);
        var output = _ln2.Forward(MathOps.Add(x1, ffOutput));
        return (output, attentionWeightsPerHead);
    }

    // Reverse of ForwardWithAttention. Each residual sum (X + attentionOutput,
    // x1 + ffOutput) sends the *same* upstream gradient to both of its inputs
    // - d(a+b)/da = d(a+b)/db = 1 - so x1 and X each accumulate a gradient
    // contribution from two different paths, which get summed.
    public float[,] Backward(float[,] dOutput)
    {
        var dSumTwo = _ln2.Backward(dOutput); // gradient w.r.t. (x1 + ffOutput)
        var dX1FromResidual = dSumTwo;
        var dFfOutput = dSumTwo;

        var dX1FromFeedForward = _feedForward.Backward(dFfOutput);
        var dX1 = MathOps.Add(dX1FromResidual, dX1FromFeedForward);

        var dSumOne = _ln1.Backward(dX1); // gradient w.r.t. (X + attentionOutput)
        var dXFromResidual = dSumOne;
        var dAttentionOutput = dSumOne;

        var dXFromAttention = _selfAttention.Backward(dAttentionOutput);
        return MathOps.Add(dXFromResidual, dXFromAttention);
    }

    public void ApplyGradients(float learningRate)
    {
        _selfAttention.ApplyGradients(learningRate);
        _feedForward.ApplyGradients(learningRate);
        _ln1.ApplyGradients(learningRate);
        _ln2.ApplyGradients(learningRate);
    }

}
