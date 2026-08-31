namespace TinyTransformer.Tests.ModelsTests;

public class TinyTransformerModelTests : TestsBase
{
    private static TinyTransformerModel BuildModel(Random rnd) => new(
        vocabSize: 6, dModel: 8, dK: 4, ffHidden: 16, numHeads: 2, numLayers: 2, rnd);

    [Fact]
    public void SaveThenLoad_ProducesIdenticalOutputOnTheSameInput()
    {
        int[] tokens = [3, 1, 4, 1, 5];
        var model = BuildModel(new Random(0));
        var before = model.Forward(tokens);

        string path = Path.Combine(Path.GetTempPath(), $"tiny-transformer-model-{Guid.NewGuid():N}.json");
        try
        {
            model.Save(path);
            var loaded = TinyTransformerModel.Load(path);
            var after = loaded.Forward(tokens);

            MatricesShouldBeApproximatelyEqual(after, before, 1e-6f);
            loaded.VocabSize.Should().Be(model.VocabSize);
            loaded.DModel.Should().Be(model.DModel);
            loaded.DK.Should().Be(model.DK);
            loaded.FfHidden.Should().Be(model.FfHidden);
            loaded.NumHeads.Should().Be(model.NumHeads);
            loaded.NumLayers.Should().Be(model.NumLayers);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_RejectsAnUnsupportedFormatVersion()
    {
        string path = Path.Combine(Path.GetTempPath(), $"tiny-transformer-model-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, """{"FormatVersion": 999}""");

        try
        {
            var act = () => TinyTransformerModel.Load(path);
            act.Should().Throw<NotSupportedException>();
        }
        finally
        {
            File.Delete(path);
        }
    }
}
