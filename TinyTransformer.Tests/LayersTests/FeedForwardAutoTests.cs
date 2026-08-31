namespace TinyTransformer.Tests.LayersTests;

public class FeedForwardAutoTests : TestsBase
{
    //1. positionwise property - if two input rows are identical, the coresponding
    // output rows must also be identical

    [Fact]
    public void FeedForwardAuto_IsPositionwise_RowsWithTheSameInputProducesSameOutput()
    {
        //Arrange
        int dModel = 4;
        int hidden = 8;
        int seed = 123;
        var rnd = new Random(seed);

        var ff = new FeedForwardAuto(dModel, hidden, rnd);

        //Build an input with duplicated rows
        var rowA = MathOps.InitVector(0.2f, -0.1f, 0.5f, 0.0f);
        var rowB = MathOps.InitVector(0.4f, -0.31f, 0.9f, -0.40f);
        var rowC = MathOps.InitVector(0.5f, -0.15f, -0.1f, 0.9f);

        var X = MathOps.InitMatrix(rowA, rowA, rowB, rowC);

        //Act 
        var Y = ff.Forward(X);

        //Assert - row 0 and 1 shold be nearly equal
        for (int i = 0; i < dModel; i++)
            Y[0, i].Should().BeApproximately(Y[1, i], 1e-5f); //compare each column value of 0 row with 1 row
    }

    [Fact]
    public void Backward_MatchesNumericalGradient()
    {
        int dModel = 5, hidden = 8, T = 4;
        var X = MathOps.InitMatrix(T, dModel, new Random(1), scale: 1f);
        var dOutput = MathOps.InitMatrix(T, dModel, new Random(2), scale: 1f);

        var ff = new FeedForwardAuto(dModel, hidden, new Random(9));
        ff.Forward(X);
        var analytical = ff.Backward(dOutput);

        var numerical = NumericalGradient(
            x => new FeedForwardAuto(dModel, hidden, new Random(9)).Forward(x),
            dOutput,
            X);

        MatricesShouldBeApproximatelyEqual(analytical, numerical, 2e-2f);
    }

    [Fact]
    public void ApplyGradients_ChangesSubsequentOutputForTheSameInput()
    {
        int dModel = 5, hidden = 8, T = 4;
        var X = MathOps.InitMatrix(T, dModel, new Random(1), scale: 1f);
        var dOutput = MathOps.InitMatrix(T, dModel, new Random(2), scale: 1f);
        var ff = new FeedForwardAuto(dModel, hidden, new Random(3));

        var before = ff.Forward(X);
        ff.Backward(dOutput);
        ff.ApplyGradients(learningRate: 0.1f);
        var after = ff.Forward(X);

        var act = () => MatricesShouldBeApproximatelyEqual(after, before, 1e-9f);
        act.Should().Throw<Exception>("both Linear layers' parameters should have moved after a gradient step");
    }
}
