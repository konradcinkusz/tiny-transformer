namespace TinyTransformer.Tests.LayersTests;

public class TransformerEncoderStackTests : TestsBase
{
    [Fact]
    public void WithOneLayer_MatchesASingleBlockExactly()
    {
        int T = 5, dModel = 8, dK = 4, ffHidden = 16;
        var X = MathOps.InitMatrix(T, dModel, new Random(2));

        var singleBlock = new TransformerEncoderBlock(dModel, dK, ffHidden, new Random(6));
        var stack = new TransformerEncoderStack(dModel, dK, ffHidden, numLayers: 1, new Random(6));

        MatricesShouldBeApproximatelyEqual(stack.Forward(X), singleBlock.Forward(X), 1e-6f);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void Forward_OutputShapeIsIndependentOfDepth(int numLayers)
    {
        int T = 5, dModel = 8, dK = 4, ffHidden = 16;
        var X = MathOps.InitMatrix(T, dModel, new Random(1));
        var stack = new TransformerEncoderStack(dModel, dK, ffHidden, numLayers, new Random(1));

        var Y = stack.Forward(X);

        Y.GetLength(0).Should().Be(T);
        Y.GetLength(1).Should().Be(dModel);
    }

    [Fact]
    public void Forward_IsDeterministicForTheSameSeed()
    {
        int T = 4, dModel = 6, dK = 3, ffHidden = 12, numLayers = 3;
        var X = MathOps.InitMatrix(T, dModel, new Random(5));

        var stack1 = new TransformerEncoderStack(dModel, dK, ffHidden, numLayers, new Random(77));
        var stack2 = new TransformerEncoderStack(dModel, dK, ffHidden, numLayers, new Random(77));

        MatricesShouldBeApproximatelyEqual(stack1.Forward(X), stack2.Forward(X), 1e-6f);
    }

    [Fact]
    public void ForwardWithAttention_ReturnsTheLastBlocksRowStochasticAttention()
    {
        int T = 5, dModel = 8, dK = 4, ffHidden = 16, numLayers = 3;
        var X = MathOps.InitMatrix(T, dModel, new Random(19));
        var stack = new TransformerEncoderStack(dModel, dK, ffHidden, numLayers, new Random(19));

        var (_, attentionWeights) = stack.ForwardWithAttention(X);

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

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Constructor_ThrowsForNonPositiveLayerCount(int numLayers)
    {
        var act = () => new TransformerEncoderStack(dModel: 8, dK: 4, ffHidden: 16, numLayers, new Random(1));

        act.Should().Throw<ArgumentException>();
    }
}
