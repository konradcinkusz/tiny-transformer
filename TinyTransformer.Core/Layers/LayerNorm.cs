namespace TinyTransformer.Core.Layers;

//It keeps transformers block's activations and gradients stable
//LayerNorm compute mean and variance, normalize the row then scale and shift it with
//learned vector gamma and beta
public class LayerNorm : IDifferentiableLayer, IHasParameterGradients
{
    private readonly int _d; // feature dimension (length of the vector that represents a single token)
    //in Transformer it's the size of each token's emvedding or hidden state
    private readonly float[] _gamma, _beta;
    private const float Eps = 1e-5f;

    // Cached from Forward, needed by Backward: the normalized value (before
    // gamma/beta) and 1/std per row.
    private float[,]? _lastNorm;
    private float[]? _lastInv;

    private float[]? _dGamma;
    private float[]? _dBeta;

    public LayerNorm(int d)
    {
        _d = d;
        _gamma = Enumerable.Repeat(1f, d).ToArray();
        _beta = Enumerable.Repeat(0f, d).ToArray();
    }

    // Deterministic - mirrors Linear's (W, b) constructor, for tests and
    // reproducible construction with specific (e.g. previously-trained) parameters.
    public LayerNorm(float[] gamma, float[] beta)
    {
        _gamma = gamma ?? throw new ArgumentNullException(nameof(gamma));
        _beta = beta ?? throw new ArgumentNullException(nameof(beta));

        if (gamma.Length != beta.Length)
            throw new ArgumentException("gamma and beta must have the same length");

        _d = gamma.Length;
    }

    public float[,] Forward(float[,] X)
    {
        int n = X.GetLength(0); // n = number of tokens (rows)
        var Y = new float[n, _d]; //output buffer, same shape as X
        var norm = new float[n, _d];
        var invs = new float[n];

        for (int i = 0; i < n; i++) //process each token independently
        {
            float mean = MathOps.Mean(X, i, _d);
            float variance = MathOps.Variance(X, i, _d, mean);
            //the core math of layer normalization

            //CORE LOGIC
            //1. compute the inverse standard deviation
            float inv = 1f / (float)Math.Sqrt(variance + Eps);
            invs[i] = inv;

            //2. normalize each feature, then scale and shift - loop
            for(int j = 0; j < _d; j++)
            {
                float n_ij = (X[i,j] - mean) * inv;
                norm[i, j] = n_ij;
                Y[i,j] = _gamma[j]* n_ij + _beta[j];
            }
        }

        _lastNorm = norm;
        _lastInv = invs;
        return Y;
    }

    // Standard LayerNorm backward (see e.g. the original Layer Normalization
    // paper's backward derivation). For row i, with N = _d features,
    // dnorm = dOut .* gamma, norm the cached normalized value, and
    // inv = 1/std for that row:
    //
    //   dX_j = inv * (dnorm_j - mean(dnorm) - norm_j * mean(dnorm .* norm))
    //
    // dGamma/dBeta accumulate across every row, mirroring how gamma/beta are
    // shared across every token.
    public float[,] Backward(float[,] dOut)
    {
        if (_lastNorm is null || _lastInv is null)
            throw new InvalidOperationException($"{nameof(Backward)} was called before {nameof(Forward)}.");

        int n = dOut.GetLength(0);
        var dX = new float[n, _d];
        _dGamma = new float[_d];
        _dBeta = new float[_d];
        var dNormRow = new float[_d];

        for (int i = 0; i < n; i++)
        {
            float meanDNorm = 0f;
            float meanDNormNorm = 0f;

            for (int j = 0; j < _d; j++)
            {
                float dNorm = dOut[i, j] * _gamma[j];
                dNormRow[j] = dNorm;
                meanDNorm += dNorm;
                meanDNormNorm += dNorm * _lastNorm[i, j];

                _dGamma[j] += dOut[i, j] * _lastNorm[i, j];
                _dBeta[j] += dOut[i, j];
            }
            meanDNorm /= _d;
            meanDNormNorm /= _d;

            float inv = _lastInv[i];
            for (int j = 0; j < _d; j++)
                dX[i, j] = inv * (dNormRow[j] - meanDNorm - _lastNorm[i, j] * meanDNormNorm);
        }

        return dX;
    }

    public void ApplyGradients(float learningRate)
    {
        if (_dGamma is null || _dBeta is null)
            throw new InvalidOperationException($"{nameof(ApplyGradients)} was called before {nameof(Backward)}.");

        for (int j = 0; j < _d; j++)
        {
            _gamma[j] -= learningRate * _dGamma[j];
            _beta[j] -= learningRate * _dBeta[j];
        }

        _dGamma = null;
        _dBeta = null;
    }
}
