namespace TinyTransformer.Core.Layers;

// Elementwise max(0, x); the derivative is 1 where the input was positive
// and 0 otherwise (the input landing on exactly 0 - where the true
// derivative is undefined - has probability ~0 for continuous inputs, so
// treating it as 0 there, like most frameworks do, has no practical effect).
//
// This is the first implementer of IDifferentiableLayer, proving the
// interface's shape end-to-end before anything with actual learnable
// parameters uses it (see ROADMAP.md Phase 2's next issues).
public class ReLU : IDifferentiableLayer
{
    private float[,]? _lastInput;

    public float[,] Forward(float[,] X)
    {
        _lastInput = X;
        return MathOps.ReLU(X);
    }

    public float[,] Backward(float[,] dOut)
    {
        if (_lastInput is null)
            throw new InvalidOperationException($"{nameof(Backward)} was called before {nameof(Forward)}.");

        int rows = _lastInput.GetLength(0);
        int cols = _lastInput.GetLength(1);
        var dX = new float[rows, cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                dX[i, j] = _lastInput[i, j] > 0f ? dOut[i, j] : 0f;

        return dX;
    }
}
