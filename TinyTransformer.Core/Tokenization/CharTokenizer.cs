namespace TinyTransformer.Core.Tokenization;

// A minimal, dependency-free tokenizer for demo purposes: it assigns each
// distinct character in the input the next free id, in order of first
// appearance. It is NOT a real subword/BPE vocabulary - there is no
// pretrained vocab file to load, and none of these weights are trained -
// but it is deterministic, has no out-of-vocabulary case, and is enough to
// turn arbitrary text into token ids for the Embedding layer to look up.
public sealed class CharTokenizer
{
    private readonly Dictionary<char, int> _charToId = new();
    private readonly List<char> _idToChar = new();

    public int VocabSize => _idToChar.Count;

    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var ids = new int[text.Length];
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (!_charToId.TryGetValue(c, out int id))
            {
                id = _idToChar.Count;
                _charToId[c] = id;
                _idToChar.Add(c);
            }
            ids[i] = id;
        }

        return ids;
    }

    public string TokenText(int id)
    {
        if (id < 0 || id >= _idToChar.Count)
            throw new ArgumentOutOfRangeException(nameof(id));

        char c = _idToChar[id];
        return c == ' ' ? "␣" : c.ToString(); // render space as a visible "open box" glyph
    }
}
