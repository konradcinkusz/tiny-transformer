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
    int? Seed);

// EncodeRequest with defaults applied and nulls resolved, ready to validate/run.
public sealed record ResolvedEncodeRequest(string Text, int DModel, int DK, int FfHidden, int Seed)
{
    public static ResolvedEncodeRequest FromRequest(EncodeRequest request, int generatedSeed) => new(
        request.Text ?? string.Empty,
        request.DModel ?? 16,
        request.DK ?? 16,
        request.FfHidden ?? 32,
        request.Seed ?? generatedSeed);
}

public sealed record EncodeConfig(
    int DModel,
    [property: JsonPropertyName("dK")] int DK,
    int FfHidden,
    int Seed,
    int SequenceLength,
    int VocabSize);

// Every matrix is [sequenceLength x dModel] except AttentionWeights, which is
// [sequenceLength x sequenceLength] (one attention distribution per token).
public sealed record EncodeResponse(
    IReadOnlyList<string> Tokens,
    IReadOnlyList<int> TokenIds,
    EncodeConfig Config,
    float[][] Embeddings,
    float[][] PositionalEncoding,
    float[][] AttentionWeights,
    float[][] EncoderOutput);
