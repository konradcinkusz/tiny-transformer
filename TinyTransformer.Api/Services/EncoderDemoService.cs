using TinyTransformer.Api.Contracts;
using TinyTransformer.Core.Layers;
using TinyTransformer.Core.Tokenization;

namespace TinyTransformer.Api.Services;

// The use-case behind POST /api/encode: text in, one transformer encoder
// block's internals out. Kept out of the endpoint delegate (P9 - transport
// stays thin) and out of Core (Core has no HTTP/validation concerns).
public sealed class EncoderDemoService
{
    public const int MaxTextLength = 64;
    public const int MinDModel = 4, MaxDModel = 64;
    public const int MinDK = 2, MaxDK = 64;
    public const int MinFfHidden = 4, MaxFfHidden = 256;

    // Bounds exist because every request runs live, synchronous, unauthenticated
    // matrix math (attention is O(sequenceLength^2 x dModel)) - this is the
    // same "clamp untrusted size inputs" discipline SERVICE-API-PATTERNS.md
    // applies to list-endpoint paging, applied here to compute cost instead.
    public IDictionary<string, string[]> Validate(ResolvedEncodeRequest request)
    {
        var errors = new Dictionary<string, List<string>>();

        void AddError(string field, string message)
        {
            if (!errors.TryGetValue(field, out var list))
                errors[field] = list = [];
            list.Add(message);
        }

        // Field names match the JSON wire casing (see EncodeContracts.cs), not
        // the C# property names, so a client can map errors[field] straight
        // back onto the request it sent without a translation table.
        if (string.IsNullOrWhiteSpace(request.Text))
            AddError("text", "Text is required.");
        else if (request.Text.Length > MaxTextLength)
            AddError("text", $"Text must be at most {MaxTextLength} characters.");

        if (request.DModel is < MinDModel or > MaxDModel)
            AddError("dModel", $"dModel must be between {MinDModel} and {MaxDModel}.");

        if (request.DK is < MinDK or > MaxDK)
            AddError("dK", $"dK must be between {MinDK} and {MaxDK}.");

        if (request.FfHidden is < MinFfHidden or > MaxFfHidden)
            AddError("ffHidden", $"ffHidden must be between {MinFfHidden} and {MaxFfHidden}.");

        return errors.ToDictionary(kv => kv.Key, kv => kv.Value.ToArray());
    }

    // Assumes Validate(request) returned no errors.
    public EncodeResponse Encode(ResolvedEncodeRequest request)
    {
        var tokenizer = new CharTokenizer();
        var tokenIds = tokenizer.Encode(request.Text);
        var tokens = tokenIds.Select(tokenizer.TokenText).ToArray();

        // One shared RNG seeds every layer's random weights deterministically:
        // the same request (including the same seed) always reproduces the
        // same "model", which is what makes the seed worth returning to the
        // client for replay.
        var rnd = new Random(request.Seed);
        var embedding = new Embedding(tokenizer.VocabSize, request.DModel, rnd);
        var encoderBlock = new TransformerEncoderBlock(request.DModel, request.DK, request.FfHidden, rnd);

        var rawEmbeddings = embedding.Lookup(tokenIds);
        var positionalEncodingTable = Core.Layers.PositionalEncoding.Build(tokenIds.Length, request.DModel);
        var withPosition = new PositionalEncoding(request.DModel, Math.Max(tokenIds.Length, 1)).Forward(rawEmbeddings);
        var (encoderOutput, attentionWeights) = encoderBlock.ForwardWithAttention(withPosition);

        var config = new EncodeConfig(
            request.DModel,
            request.DK,
            request.FfHidden,
            request.Seed,
            tokenIds.Length,
            tokenizer.VocabSize);

        return new EncodeResponse(
            tokens,
            tokenIds,
            config,
            rawEmbeddings.ToJagged(),
            positionalEncodingTable.ToJagged(),
            attentionWeights.ToJagged(),
            encoderOutput.ToJagged());
    }
}
