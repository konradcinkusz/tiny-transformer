namespace TinyTransformer.Tests.LayersTests;

public class LayerNormBackwardTests : TestsBase
{
    [Fact]
    public void Backward_InputGradientMatchesNumericalGradient()
    {
        int d = 5;
        var gamma = MathOps.InitVector(d, 1f);
        var beta = MathOps.InitVector(d, 0f);
        var X = MathOps.InitMatrix(4, d, new Random(1), scale: 2f);
        var dOut = MathOps.InitMatrix(4, d, new Random(2), scale: 1f);

        var layerNorm = new LayerNorm(gamma, beta);
        layerNorm.Forward(X);
        var analytical = layerNorm.Backward(dOut);

        var numerical = NumericalGradient(x => new LayerNorm(gamma, beta).Forward(x), dOut, X);

        MatricesShouldBeApproximatelyEqual(analytical, numerical, 1e-2f);
    }

    [Fact]
    public void Backward_GammaGradientMatchesNumericalGradient()
    {
        int d = 5;
        // Passed directly (not cloned) into the LayerNorm under test so that
        // ApplyGradients - which mutates its constructor arrays in place,
        // same as Linear's W/b - lets these outer variables observe the
        // post-update values directly, with no getter needed on LayerNorm.
        var gamma = MathOps.InitVector(d, 1f);
        var beta = MathOps.InitVector(d, 0f);
        var gammaBefore = (float[])gamma.Clone();
        var betaBefore = (float[])beta.Clone(); // beta is also mutated in place by ApplyGradients below
        var X = MathOps.InitMatrix(4, d, new Random(1), scale: 2f);
        var dOut = MathOps.InitMatrix(4, d, new Random(2), scale: 1f);

        var layerNorm = new LayerNorm(gamma, beta);
        layerNorm.Forward(X);
        layerNorm.Backward(dOut);
        layerNorm.ApplyGradients(learningRate: 1f);

        for (int j = 0; j < d; j++)
        {
            float analyticalDGamma = gammaBefore[j] - gamma[j]; // gamma -= lr * dGamma, lr = 1

            var gammaPlus = (float[])gammaBefore.Clone();
            gammaPlus[j] += 1e-3f;
            float lossPlus = DotProduct(new LayerNorm(gammaPlus, betaBefore).Forward(X), dOut);

            var gammaMinus = (float[])gammaBefore.Clone();
            gammaMinus[j] -= 1e-3f;
            float lossMinus = DotProduct(new LayerNorm(gammaMinus, betaBefore).Forward(X), dOut);

            float numericalDGamma = (lossPlus - lossMinus) / 2e-3f;
            analyticalDGamma.Should().BeApproximately(numericalDGamma, 1e-2f);
        }
    }

    [Fact]
    public void Backward_BetaGradientMatchesNumericalGradient()
    {
        int d = 5;
        var gamma = MathOps.InitVector(d, 1f);
        var beta = MathOps.InitVector(d, 0f);
        var gammaBefore = (float[])gamma.Clone(); // gamma is also mutated in place by ApplyGradients below
        var betaBefore = (float[])beta.Clone();
        var X = MathOps.InitMatrix(4, d, new Random(1), scale: 2f);
        var dOut = MathOps.InitMatrix(4, d, new Random(2), scale: 1f);

        var layerNorm = new LayerNorm(gamma, beta);
        layerNorm.Forward(X);
        layerNorm.Backward(dOut);
        layerNorm.ApplyGradients(learningRate: 1f);

        for (int j = 0; j < d; j++)
        {
            float analyticalDBeta = betaBefore[j] - beta[j]; // beta -= lr * dBeta, lr = 1

            var betaPlus = (float[])betaBefore.Clone();
            betaPlus[j] += 1e-3f;
            float lossPlus = DotProduct(new LayerNorm(gammaBefore, betaPlus).Forward(X), dOut);

            var betaMinus = (float[])betaBefore.Clone();
            betaMinus[j] -= 1e-3f;
            float lossMinus = DotProduct(new LayerNorm(gammaBefore, betaMinus).Forward(X), dOut);

            float numericalDBeta = (lossPlus - lossMinus) / 2e-3f;
            analyticalDBeta.Should().BeApproximately(numericalDBeta, 1e-2f);
        }
    }

    [Fact]
    public void ApplyGradients_ThrowsIfCalledBeforeBackward()
    {
        var layerNorm = new LayerNorm(4);

        var act = () => layerNorm.ApplyGradients(0.1f);

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void DeterministicConstructor_ThrowsOnLengthMismatch()
    {
        var act = () => new LayerNorm(gamma: [1f, 1f, 1f], beta: [0f, 0f]);

        act.Should().Throw<ArgumentException>();
    }
}
