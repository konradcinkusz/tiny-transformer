namespace TinyTransformer.Core.Layers;

// Generalizes SelfAttention to multiple heads ("Attention Is All You Need" §3.2.2):
// each head gets its own Wq/Wk/Wv projecting dModel -> dK, attention is computed
// independently per head, and the concatenated per-head outputs (width
// numHeads * dK) are projected back to dModel by one shared Wo.
//
// dK stays an independent parameter here rather than being derived as
// dModel / numHeads (the canonical paper's shape): this codebase's SelfAttention
// already treats dK as free of dModel, the API contract exposes it that way to
// clients, and deriving it would make numHeads = 1 stop matching SelfAttention
// whenever a caller picks dK != dModel (several tests do). Wo's input width
// simply grows with numHeads instead.
//
// With numHeads = 1 this is bit-for-bit identical to SelfAttention: construction
// order (Wq, Wk, Wv, then Wo) and every matrix shape match exactly - see
// MultiHeadSelfAttentionTests.
public class MultiHeadSelfAttention : ILayer
{
    private readonly int _dK;
    private readonly int _numHeads;
    private readonly Linear[] _Wq;
    private readonly Linear[] _Wk;
    private readonly Linear[] _Wv;
    private readonly Linear _Wo;

    public MultiHeadSelfAttention(int dModel, int dK, int numHeads, Random rnd)
    {
        if (numHeads <= 0)
            throw new ArgumentException("numHeads must be positive", nameof(numHeads));

        _dK = dK;
        _numHeads = numHeads;
        _Wq = new Linear[numHeads];
        _Wk = new Linear[numHeads];
        _Wv = new Linear[numHeads];

        for (int h = 0; h < numHeads; h++)
        {
            _Wq[h] = new Linear(dModel, dK, rnd);
            _Wk[h] = new Linear(dModel, dK, rnd);
            _Wv[h] = new Linear(dModel, dK, rnd);
        }

        _Wo = new Linear(dK * numHeads, dModel, rnd);
    }

    public float[,] Forward(float[,] X) => ForwardWithAttention(X).Output;

    // AttentionWeights holds one [T x T] matrix per head, in head order.
    public (float[,] Output, float[][,] AttentionWeights) ForwardWithAttention(float[,] X)
    {
        int T = X.GetLength(0);
        var attentionPerHead = new float[_numHeads][,];
        var contextPerHead = new float[_numHeads][,];
        float scale = 1f / (float)Math.Sqrt(_dK);

        for (int h = 0; h < _numHeads; h++)
        {
            var Q = _Wq[h].Forward(X);
            var K = _Wk[h].Forward(X);
            var V = _Wv[h].Forward(X);

            var scores = MathOps.MatMul(Q, MathOps.Transpose(K));
            scores = MathOps.ScalarMatrixMultiplication(scores, scale);
            var attention = MathOps.SoftmaxRows(scores);
            var context = MathOps.MatMul(attention, V);

            attentionPerHead[h] = attention;
            contextPerHead[h] = context;
        }

        var concatenated = ConcatColumns(contextPerHead, T);
        var output = _Wo.Forward(concatenated);

        return (output, attentionPerHead);
    }

    private static float[,] ConcatColumns(float[][,] perHead, int rows)
    {
        int totalCols = 0;
        foreach (var m in perHead)
            totalCols += m.GetLength(1);

        var result = new float[rows, totalCols];
        int colOffset = 0;
        foreach (var m in perHead)
        {
            int cols = m.GetLength(1);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, colOffset + j] = m[i, j];
            colOffset += cols;
        }

        return result;
    }
}
