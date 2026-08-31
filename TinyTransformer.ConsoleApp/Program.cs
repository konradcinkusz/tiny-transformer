using System.Globalization;
using TinyTransformer.Core.Layers;

namespace TinyTransformer.ConsoleApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int vocabSize = 20; //token IDs range from 0..19
            int dModel = 16; //vector size per token
            int dK = 16; //attention head size
            int ffHidden = 32; //feed forward inner layer size
            int numLayers = 2; //stacked encoder blocks

            var rnd = new Random(0);

            int[] tokens = [3, 7, 7, 2, 9];

            //Components
            var embedding = new Embedding(vocabSize, dModel, rnd);
            var posEncoding = new PositionalEncoding(dModel);
            var stack = new TransformerEncoderStack(dModel, dK, ffHidden, numLayers, rnd);

            //Forward
            float[,] X = embedding.Lookup(tokens);
            X = posEncoding.Forward(X); // inject position information before the encoder
            var (encoded, attentionWeights) = stack.ForwardWithAttention(X);

            //Inspect
            Console.WriteLine($"Input tokens: [{string.Join(", ", tokens)}]");
            Console.WriteLine();
            Console.WriteLine("Attention weights of the last encoder block (row = query token, col = attends-to token):");
            PrintMatrix(attentionWeights);
            Console.WriteLine();
            Console.WriteLine($"Encoded sequence (after {numLayers} stacked encoder blocks, [tokens x dModel]):");
            PrintMatrix(encoded);
        }

        static void PrintMatrix(float[,] M)
        {
            int rows = M.GetLength(0);
            int cols = M.GetLength(1);

            for (int i = 0; i < rows; i++)
            {
                var cells = new string[cols];
                for (int j = 0; j < cols; j++)
                    cells[j] = M[i, j].ToString("F3", CultureInfo.InvariantCulture);

                Console.WriteLine($"  [{string.Join(", ", cells)}]");
            }
        }
    }
}
