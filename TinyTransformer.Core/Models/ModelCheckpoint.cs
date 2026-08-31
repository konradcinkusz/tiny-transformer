namespace TinyTransformer.Core.Models;

// Plain-data mirror of TinyTransformerModel's learned parameters, shaped for
// System.Text.Json (which cannot serialize float[,] directly - see
// TinyTransformerModel.ToJagged/FromJagged). This is the on-disk file
// format for a saved model.
//
// Versioning: FormatVersion is written by Save and checked by Load. Bump it
// whenever a change to this shape (renamed/removed/reshaped field) would
// break reading an old file, and branch in Load on the old value if old
// files still need to be readable. Purely additive fields (a new optional
// property with a safe default) don't need a bump.
public class ModelCheckpoint
{
    public int FormatVersion { get; set; } = 1;

    public int VocabSize { get; set; }
    public int DModel { get; set; }
    public int DK { get; set; }
    public int FfHidden { get; set; }
    public int NumHeads { get; set; }
    public int NumLayers { get; set; }

    public float[][] EmbeddingTable { get; set; } = [];
    public LinearState OutputHead { get; set; } = new();
    public EncoderBlockState[] Blocks { get; set; } = [];
}

public class LinearState
{
    public float[][] W { get; set; } = [];
    public float[] B { get; set; } = [];
}

public class LayerNormState
{
    public float[] Gamma { get; set; } = [];
    public float[] Beta { get; set; } = [];
}

public class AttentionState
{
    public LinearState[] Wq { get; set; } = [];
    public LinearState[] Wk { get; set; } = [];
    public LinearState[] Wv { get; set; } = [];
    public LinearState Wo { get; set; } = new();
}

public class FeedForwardState
{
    public LinearState L1 { get; set; } = new();
    public LinearState L2 { get; set; } = new();
}

public class EncoderBlockState
{
    public AttentionState SelfAttention { get; set; } = new();
    public FeedForwardState FeedForward { get; set; } = new();
    public LayerNormState Ln1 { get; set; } = new();
    public LayerNormState Ln2 { get; set; } = new();
}
