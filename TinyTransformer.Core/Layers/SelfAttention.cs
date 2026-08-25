namespace TinyTransformer.Core.Layers;

// A class implementing a classic attention mechanism (one head)
// followe by linear projection back to dModel
// Computes attention
public class SelfAttention : ILayer
{
    //model dimension of input/output
    private readonly int _dModel;
    //attention head dimension
    private readonly int _dk;
    //layer to produce queries
    private readonly Linear _Wq;
    //keys
    private readonly Linear _Wk;
    //values
    private readonly Linear _Wv;
    //layer projecting head output back to model dModel
    private readonly Linear _Wo;

    public SelfAttention(int dModel, int dK, Random rnd)
    {
        _dModel = dModel;
        _dk = dK;
        _Wq = new Linear(dModel, dK, rnd);
        _Wk = new Linear(dModel, dK, rnd);
        _Wv = new Linear(dModel, dK, rnd);
        _Wo = new Linear(dK, dModel, rnd);
    }

    public float[,] Forward(float[,] X) => ForwardWithAttention(X).Output;

    // Same computation as Forward, but also returns the attention weight
    // matrix (post-softmax, one row per query token) - useful for callers
    // that want to inspect or visualize what each token attended to.
    public (float[,] Output, float[,] AttentionWeights) ForwardWithAttention(float[,] X)
    {
        //project input into Q, K, V
        var Q = _Wq.Forward(X);
        var K = _Wk.Forward(X);
        var V = _Wv.Forward(X);

        //comput raw attention score (scores = Q*K^T)
        var scores = MathOps.MatMul(Q, MathOps.Transpose(K));
        float scale = 1f / (float)Math.Sqrt(_dk);
        //lets scale to keep softmax in a numerically friendly range
        scores = MathOps.ScalarMatrixMultiplication(scores, scale);

        //turns each row of scores into a probability distribution
        var attention = MathOps.SoftmaxRows(scores);

        //weighted sum of values contex = attention * V
        var context = MathOps.MatMul(attention, V);

        //project back to model width
        var outProject = _Wo.Forward(context);

        return (outProject, attention);
    }
}
