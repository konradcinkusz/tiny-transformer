using System.Text.Json;
using TinyTransformer.Core.Layers;

namespace TinyTransformer.Core.Models;

// The reusable inference shape shared by every trained instance in this
// codebase: embed tokens, add position info, run them through N stacked
// transformer encoder blocks, then project back to vocabulary logits.
// Deliberately separate from Training.EchoTrainingDemo (which bakes in one
// fixed token sequence and owns the training loop) - this class only knows
// how to run and persist a model, on whatever tokens a caller provides, so
// Phase 3's API can load a pretrained instance instead of constructing and
// training one per request.
public class TinyTransformerModel
{
    public int VocabSize { get; }
    public int DModel { get; }
    public int DK { get; }
    public int FfHidden { get; }
    public int NumHeads { get; }
    public int NumLayers { get; }

    private readonly Embedding _embedding;
    private readonly PositionalEncoding _positionalEncoding;
    private readonly TransformerEncoderStack _encoder;
    private readonly Linear _outputHead;

    public TinyTransformerModel(int vocabSize, int dModel, int dK, int ffHidden, int numHeads, int numLayers, Random rnd)
    {
        VocabSize = vocabSize;
        DModel = dModel;
        DK = dK;
        FfHidden = ffHidden;
        NumHeads = numHeads;
        NumLayers = numLayers;

        _embedding = new Embedding(vocabSize, dModel, rnd);
        _positionalEncoding = new PositionalEncoding(dModel);
        _encoder = new TransformerEncoderStack(dModel, dK, ffHidden, numLayers, rnd, numHeads);
        _outputHead = new Linear(dModel, vocabSize, rnd);
    }

    // Internal, not private: also used by Training.EchoTrainingDemo.ToModel()
    // to package an already-trained instance's components for saving,
    // alongside Load's use of it for a checkpoint being read back.
    internal TinyTransformerModel(int vocabSize, int dModel, int dK, int ffHidden, int numHeads, int numLayers,
        Embedding embedding, TransformerEncoderStack encoder, Linear outputHead)
    {
        VocabSize = vocabSize;
        DModel = dModel;
        DK = dK;
        FfHidden = ffHidden;
        NumHeads = numHeads;
        NumLayers = numLayers;

        _embedding = embedding;
        _positionalEncoding = new PositionalEncoding(dModel);
        _encoder = encoder;
        _outputHead = outputHead;
    }

    // Vocabulary logits for each position: [T x vocabSize].
    public float[,] Forward(int[] tokens)
    {
        var X = _embedding.Lookup(tokens);
        X = _positionalEncoding.Forward(X);
        var encoded = _encoder.Forward(X);
        return _outputHead.Forward(encoded);
    }

    // Every intermediate stage a visualizer needs, mirroring
    // TinyTransformer.Api.Services.EncoderDemoService's random-weights
    // pipeline exactly (including what "PositionalEncoding" means there -
    // the raw sinusoidal table, not embeddings-plus-position), but reading
    // through this (possibly pretrained) model's own components instead of
    // freshly-constructed ones - see ROADMAP.md Phase 3's trained-model
    // demo path.
    public (float[,] Embeddings, float[,] PositionalEncoding, float[,] EncoderOutput, float[][][,] AttentionWeightsPerLayer) ForwardWithAllAttention(int[] tokens)
    {
        var embeddings = _embedding.Lookup(tokens);
        var positionalEncodingTable = PositionalEncoding.Build(tokens.Length, DModel);
        var withPosition = _positionalEncoding.Forward(embeddings);
        var (encoderOutput, attentionWeightsPerLayer) = _encoder.ForwardWithAllAttention(withPosition);
        return (embeddings, positionalEncodingTable, encoderOutput, attentionWeightsPerLayer);
    }

    public void Save(string path)
    {
        var checkpoint = new ModelCheckpoint
        {
            VocabSize = VocabSize,
            DModel = DModel,
            DK = DK,
            FfHidden = FfHidden,
            NumHeads = NumHeads,
            NumLayers = NumLayers,
            EmbeddingTable = ToJagged(_embedding.Table),
            OutputHead = ToLinearState(_outputHead),
            Blocks = _encoder.Blocks.Select(ToBlockState).ToArray(),
        };

        var json = JsonSerializer.Serialize(checkpoint, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    public static TinyTransformerModel Load(string path)
    {
        var json = File.ReadAllText(path);
        var checkpoint = JsonSerializer.Deserialize<ModelCheckpoint>(json)
            ?? throw new InvalidDataException($"'{path}' did not contain a valid model checkpoint.");

        if (checkpoint.FormatVersion != 1)
            throw new NotSupportedException($"Unsupported model checkpoint format version {checkpoint.FormatVersion}.");

        var embedding = new Embedding(FromJagged(checkpoint.EmbeddingTable));
        var outputHead = FromLinearState(checkpoint.OutputHead);
        var blocks = checkpoint.Blocks.Select(b => FromBlockState(b, checkpoint.DK)).ToArray();
        var encoder = new TransformerEncoderStack(blocks);

        return new TinyTransformerModel(
            checkpoint.VocabSize, checkpoint.DModel, checkpoint.DK, checkpoint.FfHidden,
            checkpoint.NumHeads, checkpoint.NumLayers,
            embedding, encoder, outputHead);
    }

    private static LinearState ToLinearState(Linear layer) => new()
    {
        W = ToJagged(layer.Weights),
        B = layer.Bias,
    };

    private static Linear FromLinearState(LinearState state) => new(FromJagged(state.W), state.B);

    private static LayerNormState ToLayerNormState(LayerNorm layer) => new()
    {
        Gamma = layer.Gamma,
        Beta = layer.Beta,
    };

    private static LayerNorm FromLayerNormState(LayerNormState state) => new(state.Gamma, state.Beta);

    private static AttentionState ToAttentionState(MultiHeadSelfAttention attention) => new()
    {
        Wq = attention.Wq.Select(ToLinearState).ToArray(),
        Wk = attention.Wk.Select(ToLinearState).ToArray(),
        Wv = attention.Wv.Select(ToLinearState).ToArray(),
        Wo = ToLinearState(attention.Wo),
    };

    private static MultiHeadSelfAttention FromAttentionState(int dK, AttentionState state) => new(
        dK,
        state.Wq.Select(FromLinearState).ToArray(),
        state.Wk.Select(FromLinearState).ToArray(),
        state.Wv.Select(FromLinearState).ToArray(),
        FromLinearState(state.Wo));

    private static FeedForwardState ToFeedForwardState(FeedForwardAuto feedForward) => new()
    {
        L1 = ToLinearState(feedForward.L1),
        L2 = ToLinearState(feedForward.L2),
    };

    private static FeedForwardAuto FromFeedForwardState(FeedForwardState state) =>
        new(FromLinearState(state.L1), FromLinearState(state.L2));

    private static EncoderBlockState ToBlockState(TransformerEncoderBlock block) => new()
    {
        SelfAttention = ToAttentionState(block.SelfAttention),
        FeedForward = ToFeedForwardState(block.FeedForward),
        Ln1 = ToLayerNormState(block.Ln1),
        Ln2 = ToLayerNormState(block.Ln2),
    };

    private static TransformerEncoderBlock FromBlockState(EncoderBlockState state, int dK) => new(
        FromAttentionState(dK, state.SelfAttention),
        FromFeedForwardState(state.FeedForward),
        FromLayerNormState(state.Ln1),
        FromLayerNormState(state.Ln2));

    // System.Text.Json cannot serialize float[,] (multidimensional arrays),
    // only jagged float[][] - these convert between the two at the
    // persistence boundary only, so every layer's own Forward/Backward math
    // keeps using float[,] as everywhere else in this codebase.
    private static float[][] ToJagged(float[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var jagged = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            jagged[i] = new float[cols];
            for (int j = 0; j < cols; j++)
                jagged[i][j] = matrix[i, j];
        }
        return jagged;
    }

    private static float[,] FromJagged(float[][] jagged)
    {
        int rows = jagged.Length;
        int cols = rows == 0 ? 0 : jagged[0].Length;
        var matrix = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i, j] = jagged[i][j];
        return matrix;
    }
}
