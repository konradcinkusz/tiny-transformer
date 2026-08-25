namespace TinyTransformer.Core.Layers;

// Classic sinusoidal positional encoding from "Attention Is All You Need".
// Token embeddings alone carry no order information (SelfAttention is
// permutation-equivariant - see SelfAttentionTests), so this is added to the
// embeddings before they reach the encoder block.
public class PositionalEncoding : ILayer
{
    private readonly float[,] _table; // [maxLen x dModel], precomputed once

    public PositionalEncoding(int dModel, int maxLen = 512)
    {
        if (dModel <= 0) throw new ArgumentException("dModel must be positive", nameof(dModel));
        if (maxLen <= 0) throw new ArgumentException("maxLen must be positive", nameof(maxLen));

        _table = Build(maxLen, dModel);
    }

    // PE(pos, 2i)   = sin(pos / 10000^(2i/dModel))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel))
    public static float[,] Build(int seqLen, int dModel)
    {
        var PE = new float[seqLen, dModel];

        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int i = 0; i < dModel; i += 2)
            {
                double angle = pos / Math.Pow(10000.0, (double)i / dModel);
                PE[pos, i] = (float)Math.Sin(angle);
                if (i + 1 < dModel)
                    PE[pos, i + 1] = (float)Math.Cos(angle);
            }
        }

        return PE;
    }

    public float[,] Forward(float[,] X)
    {
        int T = X.GetLength(0);
        int d = X.GetLength(1);

        if (T > _table.GetLength(0))
            throw new ArgumentException($"Sequence length {T} exceeds the precomputed maxLen {_table.GetLength(0)}.");
        if (d != _table.GetLength(1))
            throw new ArgumentException("Input dModel does not match the encoding's dModel.");

        var Y = new float[T, d];
        for (int i = 0; i < T; i++)
            for (int j = 0; j < d; j++)
                Y[i, j] = X[i, j] + _table[i, j];

        return Y;
    }
}
