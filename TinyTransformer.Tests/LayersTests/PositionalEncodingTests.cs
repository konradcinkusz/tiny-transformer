namespace TinyTransformer.Tests.LayersTests;

public class PositionalEncodingTests : TestsBase
{
    [Fact]
    public void Build_Position0_IsAllSinZeroCosOne()
    {
        // sin(0) = 0, cos(0) = 1 for every frequency, regardless of dModel
        var PE = PositionalEncoding.Build(seqLen: 3, dModel: 4);

        PE[0, 0].Should().BeApproximately(0f, 1e-6f); // sin
        PE[0, 1].Should().BeApproximately(1f, 1e-6f); // cos
        PE[0, 2].Should().BeApproximately(0f, 1e-6f); // sin
        PE[0, 3].Should().BeApproximately(1f, 1e-6f); // cos
    }

    [Fact]
    public void Build_MatchesClosedFormForKnownPosition()
    {
        var PE = PositionalEncoding.Build(seqLen: 2, dModel: 4);

        // pos=1: PE(1,0)=sin(1/10000^0)=sin(1); PE(1,1)=cos(1)
        // PE(1,2)=sin(1/10000^(2/4)); PE(1,3)=cos(1/10000^(2/4))
        PE[1, 0].Should().BeApproximately((float)Math.Sin(1.0), 1e-5f);
        PE[1, 1].Should().BeApproximately((float)Math.Cos(1.0), 1e-5f);
        PE[1, 2].Should().BeApproximately((float)Math.Sin(1.0 / Math.Sqrt(10000.0)), 1e-5f);
        PE[1, 3].Should().BeApproximately((float)Math.Cos(1.0 / Math.Sqrt(10000.0)), 1e-5f);
    }

    [Fact]
    public void Forward_AddsPositionalEncodingToInput()
    {
        int T = 4, dModel = 8;
        var rnd = new Random(7);
        var X = MathOps.InitMatrix(T, dModel, rnd);
        var encoder = new PositionalEncoding(dModel, maxLen: 16);

        var Y = encoder.Forward(X);
        var expectedPE = PositionalEncoding.Build(T, dModel);

        for (int i = 0; i < T; i++)
            for (int j = 0; j < dModel; j++)
                Y[i, j].Should().BeApproximately(X[i, j] + expectedPE[i, j], 1e-5f);
    }

    [Fact]
    public void Forward_ThrowsWhenSequenceExceedsMaxLen()
    {
        var encoder = new PositionalEncoding(dModel: 4, maxLen: 2);
        var X = MathOps.InitMatrix(3, 4, new Random(1));

        var act = () => encoder.Forward(X);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Forward_ThrowsOnDModelMismatch()
    {
        var encoder = new PositionalEncoding(dModel: 8);
        var X = MathOps.InitMatrix(2, 4, new Random(1)); // wrong width

        var act = () => encoder.Forward(X);

        act.Should().Throw<ArgumentException>();
    }
}
