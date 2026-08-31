namespace TinyTransformer.Core.Layers;

public class Embedding : IEmbedding
{
    private readonly int _vocabSize; // every token must be between 0 and vocabSize - 1
    private readonly int _dModel;
    private readonly float[,] _table; //the actual embedding matrix

    public Embedding(int vocabSize, int dModel, Random rnd)
    {
        _vocabSize = vocabSize;
        _dModel = dModel;
        _table = MathOps.InitMatrix(vocabSize, dModel, rnd, 0.02f);
    }

    // Deterministic - mirrors Linear's (W, b) constructor, for reconstructing
    // a previously-trained embedding table (see
    // TinyTransformer.Core.Models.TinyTransformerModel).
    public Embedding(float[,] table)
    {
        _table = table ?? throw new ArgumentNullException(nameof(table));
        _vocabSize = table.GetLength(0);
        _dModel = table.GetLength(1);
    }

    // Read-only copy for persistence.
    public float[,] Table => (float[,])_table.Clone();

    public float[,] Lookup(int[] tokenIds)//forward pass for embeddings
    {
        int T = tokenIds.Length;
        var X = new float[T, _dModel];

        for (int i = 0; i < T; i++)
        { 
            int id = tokenIds[i];
            if (id < 0 || id >= _vocabSize)
                throw new ArgumentException("Token out of range");

            for(int j = 0; j< _dModel; j++)
            {
                X[i, j] = _table[id, j];
            }
        }

        return X;
    }
}
