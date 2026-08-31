namespace TinyTransformer.Core.Layers;

// Stacks numLayers TransformerEncoderBlocks sequentially, each block's output
// feeding the next - real transformers run N blocks, not one, and until now
// every caller in this codebase only ever ran a single block per request.
public class TransformerEncoderStack : IDifferentiableLayer, IHasParameterGradients
{
    private readonly TransformerEncoderBlock[] _blocks;

    public TransformerEncoderStack(int dModel, int dK, int ffHidden, int numLayers, Random rnd, int numHeads = 1)
    {
        if (numLayers <= 0)
            throw new ArgumentException("numLayers must be positive", nameof(numLayers));

        _blocks = new TransformerEncoderBlock[numLayers];
        for (int i = 0; i < numLayers; i++)
            _blocks[i] = new TransformerEncoderBlock(dModel, dK, ffHidden, rnd, numHeads);
    }

    // Deterministic - for reconstructing a previously-trained stack (see
    // TinyTransformer.Core.Models.TinyTransformerModel).
    public TransformerEncoderStack(TransformerEncoderBlock[] blocks)
    {
        if (blocks is null || blocks.Length == 0)
            throw new ArgumentException("Provide at least one block", nameof(blocks));

        _blocks = blocks;
    }

    // Read-only accessor for persistence.
    public IReadOnlyList<TransformerEncoderBlock> Blocks => _blocks;

    public float[,] Forward(float[,] X) => ForwardWithAttention(X).Output;

    // AttentionWeights is the *last* block's attention matrix, averaged
    // across heads - the one closest to the output, and the one most callers
    // mean when they ask "what did the model attend to." With numLayers = 1
    // and numHeads = 1 this is exactly that one block's attention, so the
    // single-block behavior every existing caller relies on is unchanged.
    // Callers that want every layer's and every head's attention (e.g. the
    // API/frontend, see ROADMAP.md Phase 3) should use ForwardWithAllAttention.
    public (float[,] Output, float[,] AttentionWeights) ForwardWithAttention(float[,] X)
    {
        var (output, perLayerPerHead) = ForwardWithAllAttention(X);
        return (output, MathOps.AverageAcrossHeads(perLayerPerHead[^1]));
    }

    // Same computation as Forward, but also returns every layer's and every
    // head's own attention weights, unaveraged. PerLayerPerHeadAttentionWeights
    // is indexed [layer][head] -> one [T x T] matrix, in layer order (block 0
    // first) then head order (mirrors MultiHeadSelfAttention.ForwardWithAttention).
    public (float[,] Output, float[][][,] PerLayerPerHeadAttentionWeights) ForwardWithAllAttention(float[,] X)
    {
        float[,] current = X;
        var perLayer = new float[_blocks.Length][][,];

        for (int i = 0; i < _blocks.Length; i++)
            (current, perLayer[i]) = _blocks[i].ForwardWithPerHeadAttention(current);

        return (current, perLayer);
    }

    // Reverse of Forward: walk the blocks back-to-front, each one's Backward
    // producing the gradient w.r.t. its own input - which is exactly the
    // upstream gradient the previous block's Backward needs.
    public float[,] Backward(float[,] dOutput)
    {
        float[,] dCurrent = dOutput;

        for (int i = _blocks.Length - 1; i >= 0; i--)
            dCurrent = _blocks[i].Backward(dCurrent);

        return dCurrent;
    }

    public void ApplyGradients(float learningRate)
    {
        foreach (var block in _blocks)
            block.ApplyGradients(learningRate);
    }
}
