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
public class MultiHeadSelfAttention : IDifferentiableLayer, IHasParameterGradients
{
    private readonly int _dK;
    private readonly int _numHeads;
    private readonly Linear[] _Wq;
    private readonly Linear[] _Wk;
    private readonly Linear[] _Wv;
    private readonly Linear _Wo;

    // Cached from Forward, needed by Backward: each head's own Q/K/V and
    // attention weights (the Wq/Wk/Wv/Wo Linear instances also cache their
    // own inputs internally, but we need Q/K/V/attention themselves here to
    // backprop through the attention math between the projections).
    private float[][,]? _lastQ;
    private float[][,]? _lastK;
    private float[][,]? _lastV;
    private float[][,]? _lastAttention;
    private int? _lastInputWidth;

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

    // Deterministic - mirrors Linear's (W, b) constructor, for reconstructing
    // a previously-trained attention block (see
    // TinyTransformer.Core.Models.TinyTransformerModel). dK isn't derivable
    // from the Linear shapes alone (Wq[h] is dModel x dK, but so is every
    // other per-head projection), so it's passed explicitly.
    public MultiHeadSelfAttention(int dK, Linear[] wq, Linear[] wk, Linear[] wv, Linear wo)
    {
        if (wq.Length != wk.Length || wq.Length != wv.Length)
            throw new ArgumentException("Wq, Wk, and Wv must have the same number of heads");

        _dK = dK;
        _numHeads = wq.Length;
        _Wq = wq;
        _Wk = wk;
        _Wv = wv;
        _Wo = wo;
    }

    // Read-only accessors for persistence.
    public int DK => _dK;
    public int NumHeads => _numHeads;
    public IReadOnlyList<Linear> Wq => _Wq;
    public IReadOnlyList<Linear> Wk => _Wk;
    public IReadOnlyList<Linear> Wv => _Wv;
    public Linear Wo => _Wo;

    public float[,] Forward(float[,] X) => ForwardWithAttention(X).Output;

    // AttentionWeights holds one [T x T] matrix per head, in head order.
    public (float[,] Output, float[][,] AttentionWeights) ForwardWithAttention(float[,] X)
    {
        int T = X.GetLength(0);
        var attentionPerHead = new float[_numHeads][,];
        var contextPerHead = new float[_numHeads][,];
        var qPerHead = new float[_numHeads][,];
        var kPerHead = new float[_numHeads][,];
        var vPerHead = new float[_numHeads][,];
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

            qPerHead[h] = Q;
            kPerHead[h] = K;
            vPerHead[h] = V;
            attentionPerHead[h] = attention;
            contextPerHead[h] = context;
        }

        _lastQ = qPerHead;
        _lastK = kPerHead;
        _lastV = vPerHead;
        _lastAttention = attentionPerHead;
        _lastInputWidth = X.GetLength(1);

        var concatenated = ConcatColumns(contextPerHead, T);
        var output = _Wo.Forward(concatenated);

        return (output, attentionPerHead);
    }

    // Backprops through, in reverse: Wo -> per-head (context = attention @ V)
    // -> softmax -> scale -> (rawScores = Q @ K^T) -> Wq/Wk/Wv. Each Linear
    // (Wq[h], Wk[h], Wv[h], Wo) accumulates its own dW/db as a side effect of
    // its own Backward call, exactly as it would standalone - this class
    // does not duplicate that math, only the attention-specific parts
    // between the projections.
    public float[,] Backward(float[,] dOutput)
    {
        if (_lastQ is null || _lastK is null || _lastV is null || _lastAttention is null || _lastInputWidth is null)
            throw new InvalidOperationException($"{nameof(Backward)} was called before {nameof(Forward)}.");

        int T = dOutput.GetLength(0);
        int dModel = _lastInputWidth.Value;
        float scale = 1f / (float)Math.Sqrt(_dK);

        var dConcatenated = _Wo.Backward(dOutput);

        var dX = new float[T, dModel];
        int colOffset = 0;

        for (int h = 0; h < _numHeads; h++)
        {
            var Q = _lastQ[h];
            var K = _lastK[h];
            var V = _lastV[h];
            var attention = _lastAttention[h];
            int headWidth = V.GetLength(1);

            var dContext = SliceColumns(dConcatenated, colOffset, headWidth);
            colOffset += headWidth;

            // context = attention @ V
            var dAttention = MathOps.MatMul(dContext, MathOps.Transpose(V));
            var dV = MathOps.MatMul(MathOps.Transpose(attention), dContext);

            // attention = softmax_rows(scaledScores)
            var dScaledScores = SoftmaxBackward(attention, dAttention);

            // scaledScores = (Q @ K^T) * scale
            var dRawScores = MathOps.ScalarMatrixMultiplication(dScaledScores, scale);
            var dQ = MathOps.MatMul(dRawScores, K);
            var dK = MathOps.MatMul(MathOps.Transpose(dRawScores), Q);

            var dXq = _Wq[h].Backward(dQ);
            var dXk = _Wk[h].Backward(dK);
            var dXv = _Wv[h].Backward(dV);

            for (int i = 0; i < T; i++)
                for (int j = 0; j < dModel; j++)
                    dX[i, j] += dXq[i, j] + dXk[i, j] + dXv[i, j];
        }

        return dX;
    }

    public void ApplyGradients(float learningRate)
    {
        for (int h = 0; h < _numHeads; h++)
        {
            _Wq[h].ApplyGradients(learningRate);
            _Wk[h].ApplyGradients(learningRate);
            _Wv[h].ApplyGradients(learningRate);
        }

        _Wo.ApplyGradients(learningRate);
    }

    // Backward of softmax_rows: for row i, if a = softmax(scores_i) and dA is
    // the upstream gradient w.r.t. that row's softmax output, then
    // dScores_i,k = a_k * (dA_i,k - sum_j(dA_i,j * a_j)) - the standard
    // Jacobian-vector product for softmax, applied per row since each row is
    // an independent softmax.
    private static float[,] SoftmaxBackward(float[,] softmaxOutput, float[,] dSoftmaxOutput)
    {
        int rows = softmaxOutput.GetLength(0);
        int cols = softmaxOutput.GetLength(1);
        var dScores = new float[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            float dot = 0f;
            for (int j = 0; j < cols; j++)
                dot += dSoftmaxOutput[i, j] * softmaxOutput[i, j];

            for (int k = 0; k < cols; k++)
                dScores[i, k] = softmaxOutput[i, k] * (dSoftmaxOutput[i, k] - dot);
        }

        return dScores;
    }

    private static float[,] SliceColumns(float[,] matrix, int colStart, int width)
    {
        int rows = matrix.GetLength(0);
        var result = new float[rows, width];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < width; j++)
                result[i, j] = matrix[i, colStart + j];
        return result;
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
