using System.Text.Json.Serialization;

namespace TinyTransformer.Api.Contracts;

// What the client sends. Every numeric knob is optional - the service fills
// in a sane default - so the simplest possible request is just { "text": "hi" }.
// DK gets an explicit [JsonPropertyName] because the default camelCase
// policy lowercases an all-caps two-letter name entirely ("dk"), which reads
// as a typo next to "dModel" - "dK" is the intentional wire name.
public sealed record EncodeRequest(
    string? Text,
    int? DModel,
    [property: JsonPropertyName("dK")] int? DK,
    int? FfHidden,
    int? NumHeads,
    int? NumLayers,
    int? Seed);

// EncodeRequest with defaults applied and nulls resolved, ready to validate/run.
// NumHeads/NumLayers default to 1, preserving the single-head/single-block
// behavior every request had before either was configurable.
public sealed record ResolvedEncodeRequest(string Text, int DModel, int DK, int FfHidden, int NumHeads, int NumLayers, int Seed)
{
    public static ResolvedEncodeRequest FromRequest(EncodeRequest request, int generatedSeed) => new(
        request.Text ?? string.Empty,
        request.DModel ?? 16,
        request.DK ?? 16,
        request.FfHidden ?? 32,
        request.NumHeads ?? 1,
        request.NumLayers ?? 1,
        request.Seed ?? generatedSeed);
}

public sealed record EncodeConfig(
    int DModel,
    [property: JsonPropertyName("dK")] int DK,
    int FfHidden,
    int NumHeads,
    int NumLayers,
    int Seed,
    int SequenceLength,
    int VocabSize);

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
