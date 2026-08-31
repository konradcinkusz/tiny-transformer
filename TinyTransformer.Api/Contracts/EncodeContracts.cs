using System.Text.Json.Serialization;

namespace TinyTransformer.Api.Contracts;

// What the client sends. Every numeric knob is optional - the service fills
// in a sane default - so the simplest possible request is just { "text": "hi" }.
// DK gets an explicit [JsonPropertyName] because the default camelCase
// policy lowercases an all-caps two-letter name entirely ("dk"), which reads
// as a typo next to "dModel" - "dK" is the intentional wire name.
// UseTrainedModel switches from today's default (a fresh, randomly-weighted
// model built from this request's dModel/dK/ffHidden/numHeads/numLayers/seed
// and run on Text) to a small pretrained model (see
// TinyTransformer.Api.Services.TrainedModelFactory) run on its own fixed
// demo task instead - every other field is ignored in that mode (see
// EncoderDemoService.Validate/Encode), since a model trained on one fixed
// synthetic token sequence has nothing meaningful to say about arbitrary
// request text or a different shape.
public sealed record EncodeRequest(
    string? Text,
    int? DModel,
    [property: JsonPropertyName("dK")] int? DK,
    int? FfHidden,
    int? NumHeads,
    int? NumLayers,
    int? Seed,
    bool? UseTrainedModel);

// EncodeRequest with defaults applied and nulls resolved, ready to validate/run.
// NumHeads/NumLayers default to 1, preserving the single-head/single-block
// behavior every request had before either was configurable. UseTrainedModel
// defaults to false: random weights stay the default demo mode (see
// EncodeRequest's comment on why).
public sealed record ResolvedEncodeRequest(string Text, int DModel, int DK, int FfHidden, int NumHeads, int NumLayers, int Seed, bool UseTrainedModel)
{
    public static ResolvedEncodeRequest FromRequest(EncodeRequest request, int generatedSeed) => new(
        request.Text ?? string.Empty,
        request.DModel ?? 16,
        request.DK ?? 16,
        request.FfHidden ?? 32,
        request.NumHeads ?? 1,
        request.NumLayers ?? 1,
        request.Seed ?? generatedSeed,
        request.UseTrainedModel ?? false);
}

// Seed is meaningless when UsedTrainedModel is true (the trained weights are
// fixed, not derived from this request) - it is kept for a stable response
// shape and simply echoes whatever seed was resolved for the request.
public sealed record EncodeConfig(
    int DModel,
    [property: JsonPropertyName("dK")] int DK,
    int FfHidden,
    int NumHeads,
    int NumLayers,
    int Seed,
    int SequenceLength,
    int VocabSize,
    bool UsedTrainedModel);

// Every matrix is [sequenceLength x dModel] except AttentionWeights, which is
// [sequenceLength x sequenceLength] (one attention distribution per token).
//
// AttentionWeights is the last layer's attention, averaged across heads -
// kept for clients that only want one heatmap. AttentionWeightsPerLayer
// carries every layer's and every head's own attention, unaveraged, indexed
// [layer][head] -> [sequenceLength x sequenceLength]; AttentionWeights is
// exactly MathOps.AverageAcrossHeads applied to its last entry.
public sealed record EncodeResponse(
    IReadOnlyList<string> Tokens,
    IReadOnlyList<int> TokenIds,
    EncodeConfig Config,
    float[][] Embeddings,
    float[][] PositionalEncoding,
    float[][] AttentionWeights,
    float[][][][] AttentionWeightsPerLayer,
    float[][] EncoderOutput);
