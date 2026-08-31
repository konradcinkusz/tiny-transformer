using TinyTransformer.Core.Models;
using TinyTransformer.Core.Training;

namespace TinyTransformer.Api.Services;

// Trains (or loads a previously-trained) small model once at startup, for
// the "trained weights" option behind POST /api/encode - see ROADMAP.md
// Phase 3. Reuses Phase 2's EchoTrainingDemo for the actual training loop
// and Phase 2's TinyTransformerModel save/load format to persist it, so
// this option exercises the same mechanics Phase 2 already proved out
// rather than a bespoke training path.
//
// Training happens live at process startup rather than from a checkpoint
// committed to source control: the model is tiny (a few hundred
// iterations complete in milliseconds - see EchoTrainingDemoTests), the
// seed is fixed, so it is fully deterministic across restarts, and this
// avoids needing to manage a binary/JSON artifact's lifecycle (rebuilding
// it whenever Core's layer implementations change, storage in the Docker
// image, etc.) for what is still an educational demo, not a real model
// release. The result is cached to a temp file and immediately loaded back
// via TinyTransformerModel.Load, so the save/load path itself is exercised
// on every startup, not bypassed.
public static class TrainedModelFactory
{
    // The fixed toy "echo" task Phase 2's training loop already proved out
    // (see EchoTrainingDemoTests) - not derived from user input, since a
    // model trained to echo these specific ids has learned nothing general
    // about arbitrary text. TinyTransformer.Api's CharTokenizer assigns
    // token ids dynamically per request (in order of first appearance), so
    // there is no shared, stable vocabulary between "arbitrary user text"
    // and this fixed synthetic sequence to run the trained model on
    // meaningfully - the trained-weights demo shows this task instead.
    public static readonly int[] DemoTokens = [3, 1, 4, 1, 5];
    public const int VocabSize = 6, DModel = 8, DK = 4, FfHidden = 16, NumHeads = 2, NumLayers = 1;
    private const int TrainingIterations = 200;
    private const float LearningRate = 0.1f;

    public static TinyTransformerModel CreateTrainedModel()
    {
        string path = Path.Combine(Path.GetTempPath(), "tiny-transformer-api-trained-demo-model.json");

        if (!File.Exists(path))
        {
            var demo = new EchoTrainingDemo(DemoTokens, VocabSize, DModel, DK, FfHidden, NumHeads, NumLayers, new Random(0));
            for (int i = 0; i < TrainingIterations; i++)
                demo.TrainStep(LearningRate);

            demo.ToModel().Save(path);
        }

        return TinyTransformerModel.Load(path);
    }
}
