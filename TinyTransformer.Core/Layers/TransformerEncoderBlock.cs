namespace TinyTransformer.Core.Layers;

public class TransformerEncoderBlock : ILayer
{
    // Concrete type (not ILayer) because ForwardWithAttention below needs the
    // attention weights MultiHeadSelfAttention computes internally, which the
    // plain ILayer.Forward contract does not expose.
    private readonly MultiHeadSelfAttention _selfAttention;
    private readonly ILayer _feedForward;
    private readonly ILayer _ln1;
    private readonly ILayer _ln2;

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

    public float[,] Forward(float[,] X) => ForwardWithAttention(X).Output;

    // Same computation as Forward, but also returns the self-attention
    // weights - useful for callers that want to visualize what the block did.
    //
    // AttentionWeights is the *average* across heads when numHeads > 1: this
    // keeps the return shape [T x T] so existing callers (the API/frontend,
    // which only know how to render one attention heatmap) don't need to
    // change yet - see ROADMAP.md Phase 3 for exposing per-head detail
    // properly. Callers that want the full per-head breakdown today can use
    // MultiHeadSelfAttention directly instead of going through this class.
    public (float[,] Output, float[,] AttentionWeights) ForwardWithAttention(float[,] X)
    {
        var (attentionOutput, attentionWeightsPerHead) = _selfAttention.ForwardWithAttention(X);
        var x1 = _ln1.Forward(MathOps.Add(X, attentionOutput));
        var ffOutput = _feedForward.Forward(x1);
        var output = _ln2.Forward(MathOps.Add(x1, ffOutput));
        return (output, AverageAcrossHeads(attentionWeightsPerHead));
    }

    private static float[,] AverageAcrossHeads(float[][,] perHead)
    {
        int rows = perHead[0].GetLength(0);
        int cols = perHead[0].GetLength(1);
        var sum = new float[rows, cols];

        foreach (var head in perHead)
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    sum[i, j] += head[i, j];

        float inv = 1f / perHead.Length;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                sum[i, j] *= inv;

        return sum;
    }
}
