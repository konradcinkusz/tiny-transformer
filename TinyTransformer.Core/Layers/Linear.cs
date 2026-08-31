namespace TinyTransformer.Core.Layers;

//Linear layer: Y = X W + b
//core building block
//fully connected / dense
//Why we need a Linear class? -> Encapsulates a learnable affine transformation
public class Linear : IDifferentiableLayer, IHasParameterGradients
{
    private readonly float[,] _W;
    private readonly float[] _b;

    private float[,]? _lastInput;
    private float[,]? _dW;
    private float[]? _db;

    /// <summary>
    ///
    /// </summary>
    /// <param name="din">input dimensionality -> number of features per input sample</param>
    /// <param name="dout">output dimensionality -> number of neurons / outputs</param>
    /// <param name="rnd">random number generator to initialize weights</param>
    public Linear(int din, int dout, Random rnd)
    {
        float scale = (float)Math.Sqrt(2.0 / (din + dout));//helps keep activtions from exploding
        _W = MathOps.InitMatrix(din, dout, rnd, scale);
        _b = MathOps.InitVector(dout);
    }

    //deterministic
    public Linear(float[,] W, float[] b)
    {
        _W = W ?? throw new ArgumentNullException(nameof(W));
        _b = b ?? throw new ArgumentNullException(nameof(b));

        if (W.GetLength(1) != b.Length)
            throw new ArgumentException("W (din x dout) and b (dout) must agree");
    }

    // Read-only copies for persistence (see TinyTransformer.Core.Models.TinyTransformerModel) -
    // callers get a snapshot, not a handle that could bypass ApplyGradients.
    public float[,] Weights => (float[,])_W.Clone();
    public float[] Bias => (float[])_b.Clone();

    //how do I compute my outputs given my inputs and my current parameters
    public float[,] Forward(float[,] X)
    {
        _lastInput = X;
        var Y = MathOps.MatMul(X, _W);
        return MathOps.AddBias(Y, _b);
    }

    // Y = XW + b, so:
    //   dX = dY @ W^T   (chain rule through the matmul)
    //   dW = X^T @ dY
    //   db = column-sums of dY (b broadcasts identically into every row of Y)
    public float[,] Backward(float[,] dOut)
    {
        if (_lastInput is null)
            throw new InvalidOperationException($"{nameof(Backward)} was called before {nameof(Forward)}.");

        int rows = _lastInput.GetLength(0);
        int dout = _b.Length;

        _dW = MathOps.MatMul(MathOps.Transpose(_lastInput), dOut);

        _db = new float[dout];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < dout; j++)
                _db[j] += dOut[i, j];

        return MathOps.MatMul(dOut, MathOps.Transpose(_W));
    }

    public void ApplyGradients(float learningRate)
    {
        if (_dW is null || _db is null)
            throw new InvalidOperationException($"{nameof(ApplyGradients)} was called before {nameof(Backward)}.");

        int din = _W.GetLength(0);
        int dout = _W.GetLength(1);

        for (int i = 0; i < din; i++)
            for (int j = 0; j < dout; j++)
                _W[i, j] -= learningRate * _dW[i, j];

        for (int j = 0; j < dout; j++)
            _b[j] -= learningRate * _db[j];

        _dW = null;
        _db = null;
    }
}
