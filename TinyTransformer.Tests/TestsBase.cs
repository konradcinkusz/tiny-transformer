namespace TinyTransformer.Tests;

public class TestsBase
{
    protected static void MatricesShouldBeApproximatelyEqual(float[,] actual, float[,] expected, float tol = 1e-5f)
    {
        actual.GetLength(0).Should().Be(expected.GetLength(0));
        actual.GetLength(1).Should().Be(expected.GetLength(1));
        for (int i = 0; i < actual.GetLength(0); i++)
            for (int j = 0; j < actual.GetLength(1); j++)
                actual[i, j].Should().BeApproximately(expected[i, j], tol);
    }
    protected static void RowsShouldBeApproximatelyEqaul(float[,] A, float[,]B, float tolerance = 1e-5f)
    {
        A.GetLength(0).Should().Be(B.GetLength(0));
        A.GetLength(1).Should().Be(B.GetLength(1));
        for(int i=0; i<A.GetLength(0);i++)
            for(int j=0; j<A.GetLength(1);j++)
                A[i, j].Should().BeApproximately(B[i,j], tolerance);
    }
    protected static float[] RandomVector(int d, Random rnd)
    {
        var v = new float[d];
        for (int i = 0; i < d; i++)
            v[i] = (float)rnd.NextDouble() - 0.5f;
        return v;
    }
    protected static float[,] TakeRow(float[] row, int nCopies)
    {
        var X = new float[nCopies, row.Length];
        for (int i = 0; i < nCopies; i++)
            for (int j = 0; j < row.Length; j++)
                X[i, j] = row[j];
        return X;
    }

    // Standard gradient-check recipe: for a given dOut, the "loss" being
    // differentiated is L(X) = sum(forward(X) .* dOut) - the analytical
    // gradient of L w.r.t. X is exactly what Backward(dOut) should compute,
    // and its numerical gradient (central finite differences) is computed
    // here independently, so comparing the two catches a wrong derivation
    // that still "looks plausible" and compiles fine. Mutates and restores X
    // in place rather than cloning per-element, since X is large enough in
    // some tests (e.g. attention weight matrices) that per-element cloning
    // would be needlessly slow.
    protected static float[,] NumericalGradient(Func<float[,], float[,]> forward, float[,] dOut, float[,] X, float eps = 1e-3f)
    {
        int rows = X.GetLength(0);
        int cols = X.GetLength(1);
        var grad = new float[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float original = X[i, j];

                X[i, j] = original + eps;
                float lossPlus = DotProduct(forward(X), dOut);

                X[i, j] = original - eps;
                float lossMinus = DotProduct(forward(X), dOut);

                X[i, j] = original;
                grad[i, j] = (lossPlus - lossMinus) / (2f * eps);
            }
        }

        return grad;
    }

    protected static float DotProduct(float[,] A, float[,] B)
    {
        float sum = 0f;
        for (int i = 0; i < A.GetLength(0); i++)
            for (int j = 0; j < A.GetLength(1); j++)
                sum += A[i, j] * B[i, j];
        return sum;
    }
}
