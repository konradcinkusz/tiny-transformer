namespace TinyTransformer.Tests.LayersTests;

public class SelfAttentionTests : TestsBase
{
    [Fact]
    public void SelfAttention_WithIdenticalTokens_ProducesIdenticalOutputPerRow()
    {
        //Arrange
        int T = 5;
        int dModel = 8;
        int dK = 4;
        var rnd = new Random(123);

        //creates a single random vector of lenght dModel
        //one token embedding [x1,x2,....,x_dModel
        var row = RandomVector(dModel, rnd);
        var X = TakeRow(row, T); //[T x dModel] all rows equal

        var attention = new SelfAttention(dModel, dK, rnd);

        //Act
        var Y = attention.Forward(X); //[T x dModel]

        //Assert: every row of Y should be (approximately) identical
        for(int i  = 1; i< T; i++)
        {
            int m = Y.GetLength(1);
            for (int j = 0; j < m; j++)
                Y[i, j].Should().BeApproximately(Y[0, j], 1e-5f);
        }
    }


    [Fact]
    public void SelfAttention_IsPermutationEquivariant_OverTokens()
    {
        //if you reorder (permute) the input token
        //the output gets reordered in exactly the sameway

        //Arrange 
        int T = 7, dModel = 10, dK = 5, seed = 123;
        var rnd = new Random(seed);

        var X = MathOps.InitMatrix(T, dModel, rnd);
        
        //reverse rows of X
        //definition of simple permuatation
        var revX = MathOps.ReverseRow(X);
        var attention = new SelfAttention(dModel, dK, new Random(seed)); //same to keeps test deterministic

        //Act
        var Y = attention.Forward(X);
        var revY = attention.Forward(revX);

        var Y_exepcted = MathOps.ReverseRow(Y);
        RowsShouldBeApproximatelyEqaul(revY, Y_exepcted, 1e-5f);
    }

    [Fact]
    public void ForwardWithAttention_OutputMatchesForward()
    {
        int T = 4, dModel = 6, dK = 3;
        var X = MathOps.InitMatrix(T, dModel, new Random(11));
        var attention = new SelfAttention(dModel, dK, new Random(11));

        var forwardOnly = attention.Forward(X);
        var (withAttentionOutput, _) = attention.ForwardWithAttention(X);

        RowsShouldBeApproximatelyEqaul(forwardOnly, withAttentionOutput, 1e-6f);
    }

    [Fact]
    public void ForwardWithAttention_WeightsFormARowStochasticMatrix()
    {
        // softmax output: every row is a probability distribution over
        // "attends-to" tokens - non-negative and summing to 1.
        int T = 5, dModel = 6, dK = 4;
        var X = MathOps.InitMatrix(T, dModel, new Random(3));
        var attention = new SelfAttention(dModel, dK, new Random(3));

        var (_, weights) = attention.ForwardWithAttention(X);

        weights.GetLength(0).Should().Be(T);
        weights.GetLength(1).Should().Be(T);
        for (int i = 0; i < T; i++)
        {
            float rowSum = 0f;
            for (int j = 0; j < T; j++)
            {
                weights[i, j].Should().BeGreaterThanOrEqualTo(0f);
                rowSum += weights[i, j];
            }
            rowSum.Should().BeApproximately(1f, 1e-4f);
        }
    }
}
