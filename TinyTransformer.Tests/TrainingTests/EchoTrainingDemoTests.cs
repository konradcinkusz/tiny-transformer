namespace TinyTransformer.Tests.TrainingTests;

public class EchoTrainingDemoTests : TestsBase
{
    private static EchoTrainingDemo BuildDemo() => new(
        tokens: [3, 1, 4, 1, 5],
        vocabSize: 6,
        dModel: 8,
        dK: 4,
        ffHidden: 16,
        numHeads: 2,
        numLayers: 1,
        rnd: new Random(0));

    [Fact]
    public void TrainStep_ReducesLossSubstantiallyOverIterations()
    {
        var demo = BuildDemo();
        float initialLoss = demo.EvaluateLoss();

        float finalLoss = initialLoss;
        for (int i = 0; i < 200; i++)
            finalLoss = demo.TrainStep(learningRate: 0.1f);

        // A model with real capacity should comfortably overfit 5 fixed
        // tokens in 200 steps - this threshold is deliberately generous
        // (not "loss ~= 0") so the test isn't sensitive to the exact
        // learning rate/iteration count, only to the mechanics actually
        // working: forward, loss, backward, and SGD update all wired
        // correctly end to end.
        finalLoss.Should().BeLessThan(initialLoss * 0.1f);
    }

    [Fact]
    public void TrainStep_IsDeterministicForTheSameSeed()
    {
        var demo1 = BuildDemo();
        var demo2 = BuildDemo();

        for (int i = 0; i < 10; i++)
        {
            float loss1 = demo1.TrainStep(0.1f);
            float loss2 = demo2.TrainStep(0.1f);
            loss1.Should().BeApproximately(loss2, 1e-5f);
        }
    }

    [Fact]
    public void EvaluateLoss_DoesNotChangeParameters()
    {
        // Forward-only calls must not mutate anything a subsequent TrainStep
        // depends on - two consecutive EvaluateLoss calls should agree.
        var demo = BuildDemo();

        float first = demo.EvaluateLoss();
        float second = demo.EvaluateLoss();

        first.Should().BeApproximately(second, 1e-6f);
    }
}
