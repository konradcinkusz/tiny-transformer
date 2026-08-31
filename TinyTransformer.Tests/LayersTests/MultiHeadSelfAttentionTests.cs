namespace TinyTransformer.Tests.LayersTests;

public class MultiHeadSelfAttentionTests : TestsBase
{
    [Fact]
    public void WithOneHead_MatchesSelfAttentionExactly()
    {
        // Same construction order (Wq, Wk, Wv, then Wo) and the same shapes,
        // so the same seed must produce bit-for-bit identical weights and
        // therefore identical output.
        int T = 5, dModel = 8, dK = 4;
        var X = MathOps.InitMatrix(T, dModel, new Random(11));

        var original = new SelfAttention(dModel, dK, new Random(7));
        var multiHead = new MultiHeadSelfAttention(dModel, dK, numHeads: 1, new Random(7));

        var expected = original.Forward(X);
        var actual = multiHead.Forward(X);

        MatricesShouldBeApproximatelyEqual(actual, expected, 1e-6f);
    }

    [Fact]
    public void Forward_OutputShapeIsIndependentOfHeadCount()
    {
        int T = 6, dModel = 12, dK = 5;
        var X = MathOps.InitMatrix(T, dModel, new Random(3));

        foreach (int numHeads in new[] { 1, 2, 3 })
        {
            var attention = new MultiHeadSelfAttention(dModel, dK, numHeads, new Random(3));
            var Y = attention.Forward(X);

            Y.GetLength(0).Should().Be(T);
            Y.GetLength(1).Should().Be(dModel);
        }
    }

    [Fact]
    public void ForwardWithAttention_ReturnsOneRowStochasticMatrixPerHead()
    {
        int T = 5, dModel = 12, dK = 4, numHeads = 3;
        var X = MathOps.InitMatrix(T, dModel, new Random(9));
        var attention = new MultiHeadSelfAttention(dModel, dK, numHeads, new Random(9));

        var (_, attentionPerHead) = attention.ForwardWithAttention(X);

        attentionPerHead.Length.Should().Be(numHeads);
        foreach (var headWeights in attentionPerHead)
        {
            headWeights.GetLength(0).Should().Be(T);
            headWeights.GetLength(1).Should().Be(T);
            for (int i = 0; i < T; i++)
            {
                float rowSum = 0f;
                for (int j = 0; j < T; j++)
                {
                    headWeights[i, j].Should().BeGreaterThanOrEqualTo(0f);
                    rowSum += headWeights[i, j];
                }
                rowSum.Should().BeApproximately(1f, 1e-4f);
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_ThrowsForNonPositiveHeadCount(int numHeads)
    {
        var act = () => new MultiHeadSelfAttention(dModel: 8, dK: 4, numHeads, new Random(1));

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Backward_ThrowsIfCalledBeforeForward()
    {
        var attention = new MultiHeadSelfAttention(dModel: 8, dK: 4, numHeads: 2, new Random(1));
        var dOutput = MathOps.InitMatrix(3, 8, new Random(1));

        var act = () => attention.Backward(dOutput);

        act.Should().Throw<InvalidOperationException>();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void Backward_MatchesNumericalGradient(int numHeads)
    {
        int T = 4, dModel = 6, dK = 3;
        var X = MathOps.InitMatrix(T, dModel, new Random(5), scale: 1f);
        var dOutput = MathOps.InitMatrix(T, dModel, new Random(6), scale: 1f);

        var attention = new MultiHeadSelfAttention(dModel, dK, numHeads, new Random(42));
        attention.Forward(X);
        var analytical = attention.Backward(dOutput);

        var numerical = NumericalGradient(
            x => new MultiHeadSelfAttention(dModel, dK, numHeads, new Random(42)).Forward(x),
            dOutput,
            X);

        MatricesShouldBeApproximatelyEqual(analytical, numerical, 2e-2f);
    }

    [Fact]
    public void ApplyGradients_ChangesSubsequentOutputForTheSameInput()
    {
        // Confirms ApplyGradients isn't a no-op: it must actually reach every
        // internal Wq/Wk/Wv/Wo Linear across every head, not just some of them.
        int T = 4, dModel = 6, dK = 3, numHeads = 2;
        var X = MathOps.InitMatrix(T, dModel, new Random(1), scale: 1f);
        var dOutput = MathOps.InitMatrix(T, dModel, new Random(2), scale: 1f);
        var attention = new MultiHeadSelfAttention(dModel, dK, numHeads, new Random(3));

        var before = attention.Forward(X);
        attention.Backward(dOutput);
        attention.ApplyGradients(learningRate: 0.1f);
        var after = attention.Forward(X);

        var act = () => MatricesShouldBeApproximatelyEqual(after, before, 1e-9f);
        act.Should().Throw<Exception>("parameters should have moved after a gradient step");
    }
}
