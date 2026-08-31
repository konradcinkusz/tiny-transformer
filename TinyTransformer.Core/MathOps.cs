namespace TinyTransformer.Core;

public static class MathOps
{
    //Matrix multiplication
    public static float[,] MatMul(float[,] A, float[,] B)
    {
        int n = A.GetLength(0);
        int m = A.GetLength(1);
        int p = B.GetLength(1);

        if (B.GetLength(0) != m) throw new ArgumentException("Incompatible size");

        var C = new float[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < m; k++)
            {
                var aik = A[i, k];
                for (int j = 0; j < p; j++)
                {
                    C[i, j] += aik * B[k, j];
                }
            }
        }

        return C;
    }

    //Transpose martrix
    public static float[,] Transpose(float[,] A)
    {
        int n = A.GetLength(0);
        int m = A.GetLength(1);

        var T = new float[m, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                T[j, i] = A[i, j];
            }
        }

        return T;
    }

    //ADD matrix (takes two 2D float arrays and returns
    //their element-wise sum
    //A and B must have the same shape (n x m)
    public static float[,] Add(float[,] A, float[,] B)
    {
        int n = A.GetLength(0);
        int m = A.GetLength(1);

        if (B.GetLength(0) != n || B.GetLength(1) != m)
        {
            throw new ArgumentException("Shape mismatch");
        }

        var C = new float[n, m];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                C[i, j] = A[i, j] + B[i, j];
            }
        }

        return C;
    }

    //adding a bias vector to each row of a matrix
    //(a common step in neural networks)
    public static float[,] AddBias(float[,] A, float[] b)
    {
        int n = A.GetLength(0);
        int m = A.GetLength(1);

        if (b.Length != m)
            throw new ArgumentException("Bias length mismatch.");

        var C = new float[n, m];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                C[i, j] = A[i, j] + b[j];
            }
        }

        return C;
    }

    //scalar–matrix multiplication method
    public static float[,] ScalarMatrixMultiplication(float[,] A, float s)
    {
        int
            n = A.GetLength(0);
        int m = A.GetLength(1);

        var C = new float[n, m];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
            {
                C[i, j] = A[i, j] * s;
            }

        return C;
    }

    //turning logits into probabilities for each row
    // https://en.wikipedia.org/wiki/Softmax_function
    public static float[,] SoftmaxRows(float[,] A)
    {
        int n = A.GetLength(0); //rows
        int m = A.GetLength(1); //cols

        var S = new float[n, m];

        for (int i = 0; i < n; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < m; j++)
                if (A[i, j] > max)
                    max = A[i, j];

            float sum = 0f;

            for (int j = 0; j < m; j++)
            {
                S[i, j] = (float)Math.Exp(A[i, j] - max);
                sum += S[i, j];
            }

            float inv = 1f / sum;

            for (int j = 0; j < m; j++)
            {
                S[i, j] *= inv;
            }
        }

        return S;
    }

    //ReLU
    //https://en.wikipedia.org/wiki/Rectified_linear_unit
    public static float[,] ReLU(float[,] A)
    {
        int n = A.GetLength(0);
        int m = A.GetLength(1);

        var R = new float[n, m];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                R[i, j] = Math.Max(0f, A[i, j]);

        return R;
    }

    public static float[,] InitMatrix(int rows, int cols, Random rnd, float scale = 0.02f)
    {
        var M = new float[rows, cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                M[i, j] = (float)(rnd.NextDouble() * 2 - 1) * scale;//-scale to scale [-scale, scale), by deafult it will be [-0.02, 0.02)


        return M;
    }
    public static float[,] InitMatrix(params float[][] rows)
    {
        if (rows == null || rows.Length == 0)
            throw new ArgumentException("Provide at least one row", nameof(InitMatrix));

        int r = rows.Length;
        int c = rows[0].Length;

        var M = new float[r, c];

        for (int i = 0; i < r; i++)
        {
            if (rows[i].Length != c)
                throw new ArgumentException("All rows must be the same lenght");

            for (int j = 0; j < c; j++)
                M[i, j] = rows[i][j];
        }

        return M;
    }

    public static float[] InitVector(int n, float value = 0f)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
            v[i] = value;
        return v;
    }
    public static float[] InitVector(params float[] values)
    {
        var v = new float[values.Length];
        Array.Copy(values, v, values.Length);
        return v;
    }
    // Element-wise mean of one [T x T] attention matrix per head, collapsing
    // multi-head attention into the single matrix single-head callers (and
    // the existing API/UI heatmap) expect. Shared by TransformerEncoderBlock
    // and TransformerEncoderStack.
    public static float[,] AverageAcrossHeads(float[][,] perHead)
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

    public static float Mean(float[,] X, int row, int dim)
    {
        float mean = 0f;
        for (int j = 0; j < dim; j++)
            mean += X[row, j];
        return mean /= dim;
    }

    public static float Variance(float[,] X, int row, int dim, float mean)
    {
        float variance = 0f;
        for (int j = 0; j < dim; j++)
        {
            float u = X[row, j] - mean;
            variance += u * u;
        }
        return variance /= dim;
    }

    public static float[,] ReverseRow(float[,] X)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);
        var Y = new float[n, m];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                Y[n - 1 - i, j] = X[i, j];

        return Y;
    }

}
