namespace TinyTransformer.Tests.TokenizationTests;

public class CharTokenizerTests
{
    [Fact]
    public void Encode_AssignsSequentialIdsInOrderOfFirstAppearance()
    {
        var tokenizer = new CharTokenizer();

        var ids = tokenizer.Encode("aba");

        ids.Should().Equal(0, 1, 0); // 'a' -> 0, 'b' -> 1, repeated 'a' -> 0
        tokenizer.VocabSize.Should().Be(2);
    }

    [Fact]
    public void Encode_EmptyString_ProducesNoTokensAndEmptyVocab()
    {
        var tokenizer = new CharTokenizer();

        var ids = tokenizer.Encode(string.Empty);

        ids.Should().BeEmpty();
        tokenizer.VocabSize.Should().Be(0);
    }

    [Fact]
    public void Encode_ReusesTheSameIdForARepeatedCharacterAcrossCalls()
    {
        var tokenizer = new CharTokenizer();
        tokenizer.Encode("cat");

        var secondIds = tokenizer.Encode("act");

        // 'a' -> 1, 'c' -> 0, 't' -> 2 from the first call; vocab does not reset between calls
        secondIds.Should().Equal(1, 0, 2);
        tokenizer.VocabSize.Should().Be(3);
    }

    [Fact]
    public void TokenText_ReturnsTheOriginalCharacter()
    {
        var tokenizer = new CharTokenizer();
        tokenizer.Encode("hi");

        tokenizer.TokenText(0).Should().Be("h");
        tokenizer.TokenText(1).Should().Be("i");
    }

    [Fact]
    public void TokenText_RendersSpaceAsAVisibleGlyph()
    {
        var tokenizer = new CharTokenizer();
        tokenizer.Encode("a b");

        tokenizer.TokenText(1).Should().Be("␣");
    }

    [Fact]
    public void TokenText_ThrowsForAnUnknownId()
    {
        var tokenizer = new CharTokenizer();
        tokenizer.Encode("hi");

        var act = () => tokenizer.TokenText(5);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
