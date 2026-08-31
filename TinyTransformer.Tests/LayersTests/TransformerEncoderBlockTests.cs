namespace TinyTransformer.Tests.LayersTests;

public class TransformerEncoderBlockTests : TestsBase
{
    [Fact]
    public void Forward_PreservesInputShape()
    {
        int T = 5, dModel = 8, dK = 4, ffHidden = 16;
        var X = MathOps.InitMatrix(T, dModel, new Random(42));
        var block = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(42));

        var Y = block.Forward(X);

        Y.GetLength(0).Should().Be(T);
        Y.GetLength(1).Should().Be(dModel);
    }

    [Fact]
    public void Forward_IsDeterministicForTheSameSeed()
    {
        int T = 4, dModel = 6, dK = 3, ffHidden = 12;
        var X = MathOps.InitMatrix(T, dModel, new Random(5));

        var block1 = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(99));
        var block2 = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(99));

        var Y1 = block1.Forward(X);
        var Y2 = block2.Forward(X);

        MatricesShouldBeApproximatelyEqual(Y1, Y2, 1e-6f);
    }

    [Fact]
    public void ForwardWithAttention_OutputMatchesForward()
    {
        int T = 4, dModel = 6, dK = 3, ffHidden = 12;
        var X = MathOps.InitMatrix(T, dModel, new Random(21));
        var block = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(21));

        var forwardOnly = block.Forward(X);
        var (withAttentionOutput, _) = block.ForwardWithAttention(X);

        MatricesShouldBeApproximatelyEqual(forwardOnly, withAttentionOutput, 1e-6f);
    }

    [Fact]
    public void ForwardWithAttention_ReturnsOneAttentionRowPerToken()
    {
        int T = 5, dModel = 6, dK = 3, ffHidden = 12;
        var X = MathOps.InitMatrix(T, dModel, new Random(2));
        var block = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(2));

        var (_, attentionWeights) = block.ForwardWithAttention(X);

        attentionWeights.GetLength(0).Should().Be(T);
        attentionWeights.GetLength(1).Should().Be(T);
        for (int i = 0; i < T; i++)
        {
            float rowSum = 0f;
            for (int j = 0; j < T; j++)
                rowSum += attentionWeights[i, j];
            rowSum.Should().BeApproximately(1f, 1e-4f);
        }
    }

    [Fact]
    public void Forward_DefaultHeadCountMatchesExplicitSingleHead()
    {
        // numHeads defaults to 1 so every caller written before multi-head
        // support existed keeps compiling and behaving identically.
        int T = 4, dModel = 6, dK = 3, ffHidden = 12;
        var X = MathOps.InitMatrix(T, dModel, new Random(8));

        var defaultBlock = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(8));
        var explicitSingleHead = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(8), numHeads: 1);

        MatricesShouldBeApproximatelyEqual(defaultBlock.Forward(X), explicitSingleHead.Forward(X), 1e-6f);
    }

    [Fact]
    public void ForwardWithAttention_WithMultipleHeads_PreservesShapeAndRowStochasticity()
    {
        int T = 5, dModel = 12, dK = 4, ffHidden = 16, numHeads = 3;
        var X = MathOps.InitMatrix(T, dModel, new Random(14));
        var block = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(14), numHeads);

        var (output, attentionWeights) = block.ForwardWithAttention(X);

        output.GetLength(0).Should().Be(T);
        output.GetLength(1).Should().Be(dModel);

        // Averaging row-stochastic matrices (each row sums to 1) preserves
        // row-stochasticity, so this holds for the multi-head average too.
        attentionWeights.GetLength(0).Should().Be(T);
        attentionWeights.GetLength(1).Should().Be(T);
        for (int i = 0; i < T; i++)
        {
            float rowSum = 0f;
            for (int j = 0; j < T; j++)
                rowSum += attentionWeights[i, j];
            rowSum.Should().BeApproximately(1f, 1e-4f);
        }
    }

    [Fact]
    public void ForwardWithPerHeadAttention_AveragingItMatchesForwardWithAttention()
    {
        int T = 5, dModel = 12, dK = 4, ffHidden = 16, numHeads = 3;
        var X = MathOps.InitMatrix(T, dModel, new Random(14));
        var block = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(14), numHeads);

        var (output, perHead) = block.ForwardWithPerHeadAttention(X);
        var (averagedOutput, averagedAttention) = block.ForwardWithAttention(X);

        MatricesShouldBeApproximatelyEqual(output, averagedOutput, 1e-6f);
        perHead.Length.Should().Be(numHeads);
        foreach (var head in perHead)
        {
            head.GetLength(0).Should().Be(T);
            head.GetLength(1).Should().Be(T);
        }
        MatricesShouldBeApproximatelyEqual(MathOps.AverageAcrossHeads(perHead), averagedAttention, 1e-6f);
    }

    [Fact]
    public void Backward_MatchesNumericalGradient()
    {
        int T = 4, dModel = 6, dK = 3, ffHidden = 12, numHeads = 2;
        var X = MathOps.InitMatrix(T, dModel, new Random(30), scale: 1f);
        var dOutput = MathOps.InitMatrix(T, dModel, new Random(31), scale: 1f);

        var block = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(99), numHeads);
        block.Forward(X);
        var analytical = block.Backward(dOutput);

        var numerical = NumericalGradient(
            x => new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(99), numHeads).Forward(x),
            dOutput,
            X);

        MatricesShouldBeApproximatelyEqual(analytical, numerical, 3e-2f);
    }

    [Fact]
    public void ApplyGradients_ChangesSubsequentOutputForTheSameInput()
    {
        int T = 4, dModel = 6, dK = 3, ffHidden = 12;
        var X = MathOps.InitMatrix(T, dModel, new Random(1), scale: 1f);
        var dOutput = MathOps.InitMatrix(T, dModel, new Random(2), scale: 1f);
        var block = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(3));

        var before = block.Forward(X);
        block.Backward(dOutput);
        block.ApplyGradients(learningRate: 0.1f);
        var after = block.Forward(X);

        var act = () => MatricesShouldBeApproximatelyEqual(after, before, 1e-9f);
        act.Should().Throw<Exception>("every sub-layer's parameters should have moved after a gradient step");
    }

    [Fact]
    public void Forward_OutputIsLayerNormalizedPerToken()
    {
        // The block's last step is LayerNorm, so every output row should have
        // (close to) zero mean and unit variance - this is what pins the
        // encoder block to actually applying LN2 last, not just compiling.
        int T = 4, dModel = 10, dK = 5, ffHidden = 20;
        var X = MathOps.InitMatrix(T, dModel, new Random(17));
        var block = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(17));

        var Y = block.Forward(X);

        for (int i = 0; i < T; i++)
        {
            float mean = MathOps.Mean(Y, i, dModel);
            float variance = MathOps.Variance(Y, i, dModel, mean);
            mean.Should().BeApproximately(0f, 1e-3f);
            variance.Should().BeApproximately(1f, 1e-2f);
        }
    }
}
