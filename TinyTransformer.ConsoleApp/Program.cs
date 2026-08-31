using System.Globalization;
using TinyTransformer.Core.Layers;
using TinyTransformer.Core.Training;

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

            Console.WriteLine();
            RunTrainingDemo();
        }

        // Demonstrates that the full forward -> loss -> backward -> SGD-update
        // loop actually works, by overfitting a tiny fixed "echo" task
        // (predict each token's own id from its contextualized representation).
        static void RunTrainingDemo()
        {
            int[] tokens = [3, 1, 4, 1, 5];
            var demo = new EchoTrainingDemo(
                tokens: tokens,
                vocabSize: 6,
                dModel: 8,
                dK: 4,
                ffHidden: 16,
                numHeads: 2,
                numLayers: 1,
                rnd: new Random(0));

            Console.WriteLine($"Training loop demo (echo task, tokens: [{string.Join(", ", tokens)}]):");

            const int iterations = 200;
            const int printEvery = 20;
            float loss = 0f;
            for (int i = 0; i < iterations; i++)
            {
                loss = demo.TrainStep(learningRate: 0.1f);
                if (i % printEvery == 0 || i == iterations - 1)
                    Console.WriteLine($"  iteration {i,4}: loss = {loss:F4}");
            }

            Console.WriteLine($"Final loss after {iterations} iterations: {demo.EvaluateLoss():F4}");
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
