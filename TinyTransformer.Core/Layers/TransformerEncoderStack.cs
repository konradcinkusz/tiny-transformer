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

    public float[,] Forward(float[,] X) => ForwardWithAttention(X).Output;

    // AttentionWeights is the *last* block's attention matrix (itself already
    // averaged across heads by TransformerEncoderBlock) - the one closest to
    // the output, and the one most callers mean when they ask "what did the
    // model attend to." With numLayers = 1 this is exactly that one block's
    // attention, so the single-block behavior every existing caller relies on
    // is unchanged.
    public (float[,] Output, float[,] AttentionWeights) ForwardWithAttention(float[,] X)
    {
        float[,] current = X;
        float[,]? lastAttention = null;

        foreach (var block in _blocks)
            (current, lastAttention) = block.ForwardWithAttention(current);

        // The constructor guard (numLayers > 0) guarantees the loop above ran
        // at least once, so lastAttention is always assigned here.
        return (current, lastAttention!);
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
