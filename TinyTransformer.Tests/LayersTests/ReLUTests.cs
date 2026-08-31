namespace TinyTransformer.Tests.LayersTests;

public class ReLUTests : TestsBase
{
    [Fact]
    public void Forward_MatchesMathOpsReLU()
    {
        var X = MathOps.InitMatrix(4, 5, new Random(1), scale: 2f);
        var relu = new ReLU();

        var actual = relu.Forward(X);
        var expected = MathOps.ReLU(X);

        MatricesShouldBeApproximatelyEqual(actual, expected, 1e-6f);
    }

    [Fact]
    public void Backward_MatchesNumericalGradient()
    {
        var X = MathOps.InitMatrix(4, 5, new Random(3), scale: 2f);
        var dOut = MathOps.InitMatrix(4, 5, new Random(4), scale: 2f);
        var relu = new ReLU();
        relu.Forward(X);

        var analytical = relu.Backward(dOut);
        var numerical = NumericalGradient(MathOps.ReLU, dOut, X);

        MatricesShouldBeApproximatelyEqual(analytical, numerical, 1e-2f);
    }

    [Fact]
    public void Backward_ZerosOutGradientWherever_InputWasNonPositive()
    {
        var X = MathOps.InitMatrix([[-1f, 2f, 0f, -3f]]);
        var dOut = MathOps.InitMatrix([[5f, 7f, 9f, 11f]]);
        var relu = new ReLU();
        relu.Forward(X);

        var dX = relu.Backward(dOut);

        dX[0, 0].Should().Be(0f);  // input was negative
        dX[0, 1].Should().Be(7f); // input was positive: gradient passes through unchanged
        dX[0, 2].Should().Be(0f);  // input was exactly zero (the ReLU kink): treated as 0
        dX[0, 3].Should().Be(0f);  // input was negative
    }

    [Fact]
    public void Backward_ThrowsIfCalledBeforeForward()
    {
        var relu = new ReLU();
        var dOut = MathOps.InitMatrix(2, 2, new Random(1));

        var act = () => relu.Backward(dOut);

        act.Should().Throw<InvalidOperationException>();
    }
}
