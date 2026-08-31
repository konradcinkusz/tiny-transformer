namespace TinyTransformer.Tests.LayersTests;

public class LinearBackwardTests : TestsBase
{
    [Fact]
    public void Backward_InputGradientMatchesNumericalGradient()
    {
        var W = MathOps.InitMatrix(4, 3, new Random(1), scale: 1f);
        var b = MathOps.InitVector(3, 0.1f);
        var X = MathOps.InitMatrix(5, 4, new Random(2), scale: 1f);
        var dOut = MathOps.InitMatrix(5, 3, new Random(3), scale: 1f);

        var linear = new Linear(W, b);
        linear.Forward(X);
        var analytical = linear.Backward(dOut);

        var numerical = NumericalGradient(x => new Linear(W, b).Forward(x), dOut, X);

        MatricesShouldBeApproximatelyEqual(analytical, numerical, 1e-2f);
    }

    [Fact]
    public void Backward_WeightGradientMatchesNumericalGradient()
    {
        var W = MathOps.InitMatrix(4, 3, new Random(1), scale: 1f);
        var b = MathOps.InitVector(3, 0.1f);
        var X = MathOps.InitMatrix(5, 4, new Random(2), scale: 1f);
        var dOut = MathOps.InitMatrix(5, 3, new Random(3), scale: 1f);

        var linear = new Linear(W, b);
        linear.Forward(X);
        linear.Backward(dOut);
        // ApplyGradients is the only way Backward's dW is observable from
        // outside, so read it indirectly: apply a tiny known learning rate
        // and infer dW from how much W moved.
        var wBefore = (float[,])W.Clone();
        linear.ApplyGradients(learningRate: 1f);
        var analyticalDW = new float[W.GetLength(0), W.GetLength(1)];
        for (int i = 0; i < W.GetLength(0); i++)
            for (int j = 0; j < W.GetLength(1); j++)
                analyticalDW[i, j] = wBefore[i, j] - W[i, j]; // W -= lr * dW, lr = 1

        var numericalDW = NumericalGradientForMatrix(wBefore, (i, j, value) =>
        {
            var perturbedW = (float[,])wBefore.Clone();
            perturbedW[i, j] = value;
            return DotProduct(new Linear(perturbedW, b).Forward(X), dOut);
        });

        MatricesShouldBeApproximatelyEqual(analyticalDW, numericalDW, 1e-2f);
    }

    [Fact]
    public void Backward_BiasGradientMatchesNumericalGradient()
    {
        var W = MathOps.InitMatrix(4, 3, new Random(1), scale: 1f);
        var b = MathOps.InitVector(3, 0.1f);
        var X = MathOps.InitMatrix(5, 4, new Random(2), scale: 1f);
        var dOut = MathOps.InitMatrix(5, 3, new Random(3), scale: 1f);

        var linear = new Linear(W, b);
        linear.Forward(X);
        linear.Backward(dOut);
        var bBefore = (float[])b.Clone();
        linear.ApplyGradients(learningRate: 1f);

        for (int j = 0; j < b.Length; j++)
        {
            float analyticalDb = bBefore[j] - b[j]; // b -= lr * db, lr = 1

            float original = bBefore[j];
            var bPlus = (float[])bBefore.Clone();
            bPlus[j] = original + 1e-3f;
            float lossPlus = DotProduct(new Linear(W, bPlus).Forward(X), dOut);

            var bMinus = (float[])bBefore.Clone();
            bMinus[j] = original - 1e-3f;
            float lossMinus = DotProduct(new Linear(W, bMinus).Forward(X), dOut);

            float numericalDb = (lossPlus - lossMinus) / (2e-3f);
            analyticalDb.Should().BeApproximately(numericalDb, 1e-2f);
        }
    }

    [Fact]
    public void ApplyGradients_ThrowsIfCalledBeforeBackward()
    {
        var linear = new Linear(din: 3, dout: 2, new Random(1));

        var act = () => linear.ApplyGradients(0.1f);

        act.Should().Throw<InvalidOperationException>();
    }

    private static float[,] NumericalGradientForMatrix(float[,] original, Func<int, int, float, float> lossWithElementSetTo, float eps = 1e-3f)
    {
        int rows = original.GetLength(0);
        int cols = original.GetLength(1);
        var grad = new float[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float lossPlus = lossWithElementSetTo(i, j, original[i, j] + eps);
                float lossMinus = lossWithElementSetTo(i, j, original[i, j] - eps);
                grad[i, j] = (lossPlus - lossMinus) / (2f * eps);
            }
        }

        return grad;
    }
}
