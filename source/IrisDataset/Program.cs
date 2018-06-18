namespace MultiClass
{
    using MultiClass.Models.Iris;
    using MultiClass.Models.PredictDigits;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Trainers;
    using Microsoft.ML.Transforms;
    using System;

    class Program
    {
        static void Main(string[] args)
        {
            IrisDataPredition();
            Console.ReadKey();

            DigitsDataPrediction();
            Console.ReadKey();
        }

        /// <summary>
        /// Source:
        /// https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet/get-started/windows
        /// </summary>
        internal static void IrisDataPredition()
        {
            Console.WriteLine("1> Training and predicting Iris data:");

            // STEP 2: Create a pipeline and load your data
            var pipeline = new LearningPipeline();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of iris-data.txt is set to 'Copy always'
            string dataPath = @"Models\Iris\Data\iris-data.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            // STEP 3: Transform your data
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training
            pipeline.Add(new Dictionarizer("Label"));

            // Puts all features into a vector
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            // STEP 4: Add learner
            // Add a learning algorithm to the pipeline. 
            // This is a classification scenario (What type of iris is this?)
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // Convert the Label back into original text (after converting to number in step 3)
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 5: Train your model based on the data set
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // STEP 6: Use your model to make a prediction
            // You can change these numbers to test different predictions
            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
        }

        /// <summary>
        /// Source:
        /// https://stackoverflow.com/questions/50497593/how-to-predict-integer-values-using-ml-net
        /// https://github.com/Rowandish/MachineLearningTest
        /// </summary>
        internal static void DigitsDataPrediction()
        {
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("2> Training and predicting Digits data:");

            var dataPath = @"Models\PredictDigits\Data\segments.txt";
            var pipeline = new LearningPipeline
            {
                new TextLoader(dataPath).CreateFrom<Digit>(separator: ','),
                new ColumnConcatenator("Features", nameof(Digit.Features)),
                new StochasticDualCoordinateAscentClassifier()
            };

            var model = pipeline.Train<Digit, DigitPrediction>();
            var prediction = model.Predict(new Digit
            {
                Up = 1,
                Middle = 1,
                Bottom = 0,
                UpLeft = 1,
                BottomLeft = 1,
                TopRight = 1,
                BottomRight = 1
            });

            Console.WriteLine($"Predicted digit is: {prediction.ExpectedDigit - 1}");
        }
    }
}
