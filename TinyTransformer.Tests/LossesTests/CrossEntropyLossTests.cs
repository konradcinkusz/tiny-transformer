namespace TinyTransformer.Tests.LossesTests;

public class CrossEntropyLossTests : TestsBase
{
    [Fact]
    public void Forward_UniformLogits_EqualsLogOfClassCount()
    {
        // softmax(all-zero logits) is uniform 1/C, so -log(1/C) = log(C) -
        // a closed-form sanity check independent of the softmax/log code path.
        int C = 4;
        var logits = new float[3, C]; // all zero
        int[] targets = [0, 1, 2];
        var loss = new CrossEntropyLoss();

        float actual = loss.Forward(logits, targets);

        actual.Should().BeApproximately((float)Math.Log(C), 1e-4f);
    }

    [Fact]
    public void Forward_StronglyCorrectPrediction_ProducesNearZeroLoss()
    {
        var logits = new float[,] { { 20f, -20f, -20f } }; // overwhelmingly predicts class 0
        int[] targets = [0];
        var loss = new CrossEntropyLoss();

        float actual = loss.Forward(logits, targets);

        actual.Should().BeLessThan(1e-6f);
    }

    [Fact]
    public void Forward_TargetOutOfRange_Throws()
    {
        var logits = new float[,] { { 1f, 2f, 3f } };
        int[] targets = [3]; // only 0..2 are valid
        var loss = new CrossEntropyLoss();

        var act = () => loss.Forward(logits, targets);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Forward_TargetsLengthMismatch_Throws()
    {
        var logits = new float[2, 3];
        int[] targets = [0]; // only one target for two rows

        var loss = new CrossEntropyLoss();
        var act = () => loss.Forward(logits, targets);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Backward_ThrowsIfCalledBeforeForward()
    {
        var loss = new CrossEntropyLoss();

        var act = () => loss.Backward();

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Backward_MatchesNumericalGradient()
    {
        var logits = MathOps.InitMatrix(4, 3, new Random(1), scale: 2f);
        int[] targets = [0, 2, 1, 1];
        var loss = new CrossEntropyLoss();
        loss.Forward(logits, targets);

        var analytical = loss.Backward();
        var numerical = NumericalGradientOfScalar(logits, x => new CrossEntropyLoss().Forward(x, targets));

        MatricesShouldBeApproximatelyEqual(analytical, numerical, 1e-2f);
    }

    // CrossEntropyLoss.Forward already returns a scalar loss directly, so
    // (unlike TestsBase.NumericalGradient, built for Backward(dOut) -> dX
    // layers where the scalar loss is synthesized via a dot product with a
    // caller-supplied dOut) the check here just perturbs each logit and
    // reads Forward's own loss back.
    private static float[,] NumericalGradientOfScalar(float[,] X, Func<float[,], float> scalarLoss, float eps = 1e-3f)
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
                float lossPlus = scalarLoss(X);

                X[i, j] = original - eps;
                float lossMinus = scalarLoss(X);

                X[i, j] = original;
                grad[i, j] = (lossPlus - lossMinus) / (2f * eps);
            }
        }

        return grad;
    }
}
