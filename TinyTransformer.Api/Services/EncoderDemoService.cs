using TinyTransformer.Api.Contracts;
using TinyTransformer.Core;
using TinyTransformer.Core.Layers;
using TinyTransformer.Core.Models;
using TinyTransformer.Core.Tokenization;

namespace TinyTransformer.Api.Services;

// The use-case behind POST /api/encode: text in, a stack of transformer
// encoder blocks' internals out (one block by default, matching every
// request's behavior before numHeads/numLayers were configurable). Kept out
// of the endpoint delegate (P9 - transport stays thin) and out of Core
// (Core has no HTTP/validation concerns).
public sealed class EncoderDemoService
{
    // The fixed demo sequence TrainedModelFactory trained this on - "tokens"
    // for the trained-model response, since a pretrained model's own vocab
    // doesn't come from CharTokenizer (see EncodeRequest's comment on why).
    private static readonly string[] TrainedModelTokenLabels =
        TrainedModelFactory.DemoTokens.Select(id => id.ToString()).ToArray();

    private readonly TinyTransformerModel _trainedModel;

    public EncoderDemoService(TinyTransformerModel trainedModel)
    {
        _trainedModel = trainedModel;
    }

    public const int MaxTextLength = 64;
    public const int MinDModel = 4, MaxDModel = 64;
    public const int MinDK = 2, MaxDK = 64;
    public const int MinFfHidden = 4, MaxFfHidden = 256;
    public const int MinNumHeads = 1, MaxNumHeads = 8;
    public const int MinNumLayers = 1, MaxNumLayers = 6;

    // Bounds exist because every request runs live, synchronous, unauthenticated
    // matrix math (attention is O(sequenceLength^2 x dModel)) - this is the
    // same "clamp untrusted size inputs" discipline SERVICE-API-PATTERNS.md
    // applies to list-endpoint paging, applied here to compute cost instead.
    // NumHeads/NumLayers get tighter ceilings than the other knobs because
    // they multiply the base cost directly (numHeads x numLayers, on top of
    // dModel/dK/ffHidden already being large) rather than being one factor
    // in it.
    public IDictionary<string, string[]> Validate(ResolvedEncodeRequest request)
    {
        var errors = new Dictionary<string, List<string>>();

        // The trained model ignores every other field (see EncodeRequest's
        // comment on why), so none of them are validated in this mode - the
        // simplest valid trained-model request is just { "useTrainedModel": true }.
        if (request.UseTrainedModel)
            return errors.ToDictionary(kv => kv.Key, kv => kv.Value.ToArray());

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

        if (request.NumHeads is < MinNumHeads or > MaxNumHeads)
            AddError("numHeads", $"numHeads must be between {MinNumHeads} and {MaxNumHeads}.");

        if (request.NumLayers is < MinNumLayers or > MaxNumLayers)
            AddError("numLayers", $"numLayers must be between {MinNumLayers} and {MaxNumLayers}.");

        return errors.ToDictionary(kv => kv.Key, kv => kv.Value.ToArray());
    }

    // Assumes Validate(request) returned no errors.
    public EncodeResponse Encode(ResolvedEncodeRequest request) =>
        request.UseTrainedModel ? EncodeWithTrainedModel() : EncodeWithRandomWeights(request);

    private EncodeResponse EncodeWithTrainedModel()
    {
        var tokenIds = TrainedModelFactory.DemoTokens;
        var (embeddings, positionalEncodingTable, encoderOutput, attentionWeightsPerLayer) =
            _trainedModel.ForwardWithAllAttention(tokenIds);
        var attentionWeights = MathOps.AverageAcrossHeads(attentionWeightsPerLayer[^1]);

        var config = new EncodeConfig(
            _trainedModel.DModel,
            _trainedModel.DK,
            _trainedModel.FfHidden,
            _trainedModel.NumHeads,
            _trainedModel.NumLayers,
            Seed: 0,
            tokenIds.Length,
            _trainedModel.VocabSize,
            UsedTrainedModel: true);

        return new EncodeResponse(
            TrainedModelTokenLabels,
            tokenIds,
            config,
            embeddings.ToJagged(),
            positionalEncodingTable.ToJagged(),
            attentionWeights.ToJagged(),
            attentionWeightsPerLayer.Select(perHead => perHead.Select(m => m.ToJagged()).ToArray()).ToArray(),
            encoderOutput.ToJagged());
    }

    private EncodeResponse EncodeWithRandomWeights(ResolvedEncodeRequest request)
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
        var encoderStack = new TransformerEncoderStack(request.DModel, request.DK, request.FfHidden, request.NumLayers, rnd, request.NumHeads);

        var rawEmbeddings = embedding.Lookup(tokenIds);
        var positionalEncodingTable = Core.Layers.PositionalEncoding.Build(tokenIds.Length, request.DModel);
        var withPosition = new PositionalEncoding(request.DModel, Math.Max(tokenIds.Length, 1)).Forward(rawEmbeddings);
        var (encoderOutput, attentionWeightsPerLayer) = encoderStack.ForwardWithAllAttention(withPosition);
        var attentionWeights = MathOps.AverageAcrossHeads(attentionWeightsPerLayer[^1]);

        var config = new EncodeConfig(
            request.DModel,
            request.DK,
            request.FfHidden,
            request.NumHeads,
            request.NumLayers,
            request.Seed,
            tokenIds.Length,
            tokenizer.VocabSize,
            UsedTrainedModel: false);

        return new EncodeResponse(
            tokens,
            tokenIds,
            config,
            rawEmbeddings.ToJagged(),
            positionalEncodingTable.ToJagged(),
            attentionWeights.ToJagged(),
            attentionWeightsPerLayer.Select(perHead => perHead.Select(m => m.ToJagged()).ToArray()).ToArray(),
            encoderOutput.ToJagged());
    }
}
