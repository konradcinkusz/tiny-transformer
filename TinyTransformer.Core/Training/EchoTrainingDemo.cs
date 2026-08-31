using TinyTransformer.Core.Layers;
using TinyTransformer.Core.Losses;
using TinyTransformer.Core.Models;

namespace TinyTransformer.Core.Training;

// A minimal training harness: forward -> loss -> backward -> plain SGD
// update, on a single fixed toy task. Deliberately not a general-purpose
// trainer - no early stopping, no validation split, no optimizer beyond SGD,
// no batching across multiple examples - see ROADMAP.md Phase 2, which
// commits only to proving the forward/backward/update mechanics actually
// work end to end, not to a real training setup.
//
// The task: "echo" - at every position, predict that position's own token
// id from its *contextualized* representation (after the encoder has mixed
// information across positions via attention). This needs no external
// labels to construct (the labels are the input itself) and is trivially
// learnable for a model with any real capacity, which is exactly what makes
// it a good "does the training loop actually work" check rather than a
// demonstration of anything resembling real language understanding.
public class EchoTrainingDemo
{
    public int VocabSize { get; }
    public int DModel { get; }
    public int DK { get; }
    public int FfHidden { get; }
    public int NumHeads { get; }
    public int NumLayers { get; }

    private readonly int[] _tokens;
    private readonly Embedding _embedding;
    private readonly PositionalEncoding _positionalEncoding;
    private readonly TransformerEncoderStack _encoder;
    private readonly Linear _outputHead;
    private readonly CrossEntropyLoss _loss;

    public EchoTrainingDemo(int[] tokens, int vocabSize, int dModel, int dK, int ffHidden, int numHeads, int numLayers, Random rnd)
    {
        VocabSize = vocabSize;
        DModel = dModel;
        DK = dK;
        FfHidden = ffHidden;
        NumHeads = numHeads;
        NumLayers = numLayers;

        _tokens = tokens;
        _embedding = new Embedding(vocabSize, dModel, rnd);
        _positionalEncoding = new PositionalEncoding(dModel);
        _encoder = new TransformerEncoderStack(dModel, dK, ffHidden, numLayers, rnd, numHeads);
        _outputHead = new Linear(dModel, vocabSize, rnd);
        _loss = new CrossEntropyLoss();
    }

    // Runs one forward -> backward -> SGD step and returns the loss computed
    // *before* this step's update (i.e. the loss this step is reducing).
    public float TrainStep(float learningRate)
    {
        float lossValue = EvaluateLoss();

        var dLogits = _loss.Backward();
        var dEncoded = _outputHead.Backward(dLogits);
        _encoder.Backward(dEncoded);

        // Embedding and PositionalEncoding have no learnable parameters
        // (positional encoding is fixed sinusoidal; the embedding table
        // itself is not trained by this demo), so there is nothing further
        // to call ApplyGradients on for them.
        _outputHead.ApplyGradients(learningRate);
        _encoder.ApplyGradients(learningRate);

        return lossValue;
    }

    // Forward pass only - the current loss without taking a training step.
    public float EvaluateLoss()
    {
        var X = _embedding.Lookup(_tokens);
        X = _positionalEncoding.Forward(X);
        var encoded = _encoder.Forward(X);
        var logits = _outputHead.Forward(encoded);
        return _loss.Forward(logits, _tokens);
    }

    // Packages this instance's current (e.g. trained) components into a
    // TinyTransformerModel, so they can be persisted with Models.
    // ModelCheckpoint's save/load format and served for inference elsewhere
    // (e.g. the API's trained-weights demo path) - see ROADMAP.md Phase 3.
    // Takes a live reference to this demo's own Embedding/Encoder/output
    // Linear, not a copy: further TrainStep calls on this instance would
    // keep mutating the same weights the returned model reads.
    public TinyTransformerModel ToModel() =>
        new(VocabSize, DModel, DK, FfHidden, NumHeads, NumLayers, _embedding, _encoder, _outputHead);
}
