namespace TinyTransformer.Core.Losses;

// Combined softmax + cross-entropy loss for a batch of classification rows.
// Computes softmax internally rather than composing MathOps.SoftmaxRows with
// a separate, more complicated general softmax backward, using the
// well-known simplification for this specific pairing: dLogits = probs -
// oneHotTargets. This is more in the spirit of this codebase's
// hand-derived-math style than a generic composition would be.
//
// Not an ILayer: a loss reduces to a scalar and needs a target label per
// row, which doesn't fit ILayer's Forward(X) -> Y shape.
public class CrossEntropyLoss
{
    private float[,]? _lastProbs;
    private int[]? _lastTargets;

    // logits: [T x C] (T rows, C classes). targets: length T, each in [0, C).
    // Returns the *mean* cross-entropy loss over the T rows (not the sum),
    // so the loss magnitude doesn't grow with the number of rows in a batch.
    public float Forward(float[,] logits, int[] targets)
    {
        int T = logits.GetLength(0);
        int C = logits.GetLength(1);

        if (targets.Length != T)
            throw new ArgumentException($"targets must have one entry per row of logits ({T}), got {targets.Length}.", nameof(targets));

        var probs = MathOps.SoftmaxRows(logits);
        _lastProbs = probs;
        _lastTargets = targets;

        float loss = 0f;
        for (int t = 0; t < T; t++)
        {
            int target = targets[t];
            if (target < 0 || target >= C)
                throw new ArgumentOutOfRangeException(nameof(targets), $"Target class {target} at row {t} is out of range for {C} classes.");

            // Clamp guards log(0) for a target the model assigns ~zero
            // probability - astronomically unlikely with random weights,
            // but not impossible, and -Infinity would poison every
            // downstream gradient.
            loss -= (float)Math.Log(Math.Max(probs[t, target], 1e-12f));
        }

        return loss / T;
    }

    // Returns dLoss/dLogits, shape [T x C]. Must be called after Forward.
    public float[,] Backward()
    {
        if (_lastProbs is null || _lastTargets is null)
            throw new InvalidOperationException($"{nameof(Backward)} was called before {nameof(Forward)}.");

        int T = _lastProbs.GetLength(0);
        int C = _lastProbs.GetLength(1);
        var dLogits = new float[T, C];
        float invT = 1f / T; // matches Forward's mean reduction (loss / T)

        for (int t = 0; t < T; t++)
        {
            for (int c = 0; c < C; c++)
                dLogits[t, c] = _lastProbs[t, c] * invT;

            dLogits[t, _lastTargets[t]] -= invT;
        }

        return dLogits;
    }
}
