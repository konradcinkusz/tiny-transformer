namespace TinyTransformer.Api.Services;

// System.Text.Json cannot serialize a 2D array (float[,]) - Core's math types
// use float[,] throughout because that is the natural shape for matrix ops,
// so the API boundary is where it gets converted to a JSON-friendly jagged
// array, once, rather than pushing serialization concerns into Core.
public static class MatrixJsonExtensions
{
    public static float[][] ToJagged(this float[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var result = new float[rows][];

        for (int i = 0; i < rows; i++)
        {
            var row = new float[cols];
            for (int j = 0; j < cols; j++)
                row[j] = matrix[i, j];
            result[i] = row;
        }

        return result;
    }
}
