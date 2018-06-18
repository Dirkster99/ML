using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MultiClass
{
    /// <summary>
    /// Tutorial on ML.Net:
    /// https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis
    /// 
    /// Data-Source: http://wortschatz.uni-leipzig.de
    /// </summary>
    public class Program
    {
        #region LanguageInputData
        static readonly IEnumerable<ClassificationData> predictLangData = new[]
        {
            new ClassificationData
            {
                Text = "Hi there, this is Dirk speaking."
            },
            new ClassificationData
            {
                Text = "Hallo, mein Name ist Dirk."
            },
            new ClassificationData
            {
                Text = "Hola, mi nombre es Dirk."
            },
            new ClassificationData
            {
                Text = "Ciao, mi chiamo Dirk."
            },
            new ClassificationData
            {
                Text = "Bună ziua, numele meu este Dirk."
            },
            new ClassificationData
            {
                Text = "Bonjour, je m'appelle Dirk."
            }
        };

        private static string[] langClasses = { "German", "English", "French", "Italien", "Romanian", "Spanish"};

        private static readonly InputData langInputData = new InputData
        (
            @".\Data\Languages\01_Train.txt",
            @".\Data\Languages\02_Test.txt",
            predictLangData, langClasses
        );
        #endregion LanguageInputData

        const string _modelpath = @".\Data\Model.zip";

        public static void Main(string[] args)
        {
            InputData input = langInputData;

            Task.Run(async () =>
            {
                // Get a model trained to use for evaluation
                var model = await TrainAsync(input);

                Evaluate(model, input);

                Predict(model, input);

                Console.ReadKey();

            }).GetAwaiter().GetResult();
        }

        internal static async Task<PredictionModel<ClassificationData, SentimentPrediction>>
            TrainAsync(InputData input)
        {
            // LearningPipeline allows you to add steps in order to keep everything together 
            // during the learning process.  
            var pipeline = new LearningPipeline();

            // The TextLoader loads a dataset with comments and corresponding postive or negative sentiment. 
            // When you create a loader, you specify the schema by passing a class to the loader containing
            // all the column names and their types. This is used to create the model, and train it. 

            //pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());
            pipeline.Add(new TextLoader(input.TrainingData).CreateFrom<ClassificationData>());

            pipeline.Add(new Dictionarizer("Label"));

            // TextFeaturizer is a transform that is used to featurize an input column. 
            // This is used to format and clean the data.
            pipeline.Add(new TextFeaturizer("Features", "Text"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });


            // Train the pipeline based on the dataset that has been loaded, transformed.
            PredictionModel<ClassificationData, SentimentPrediction> model =
                                pipeline.Train<ClassificationData, SentimentPrediction>();

            // Saves the model we trained to a zip file.
            await model.WriteAsync(_modelpath);

            // Returns the model we trained to use for evaluation.
            return model;
        }

        /// <summary>
        /// Evaluates the trained model for quality assurance against a second data set.
        /// 
        /// Loads the test dataset.
        /// Creates the binary evaluator.
        /// Evaluates the model and create metrics.
        /// 
        /// Displays the metrics.
        /// </summary>
        /// <param name="model"></param>
        internal static void Evaluate(
            PredictionModel<ClassificationData, SentimentPrediction> model,
            InputData input)
        {
            // loads the new test dataset with the same schema.
            // You can evaluate the model using this dataset as a quality check.

            //var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
            var testData = new TextLoader(input.TestData).CreateFrom<ClassificationData>();

            // Computes the quality metrics for the PredictionModel using the specified dataset.
            var evaluator = new ClassificationEvaluator();

            // The BinaryClassificationMetrics contains the overall metrics computed by binary
            // classification evaluators. To display these to determine the quality of the model,
            // you need to get the metrics first.
            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            // Displaying the metrics for model validation
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy Macro: {metrics.AccuracyMacro:P2}");
            Console.WriteLine($"Accuracy Micro: {metrics.AccuracyMicro:P2}");
            Console.WriteLine($" Top KAccuracy: {metrics.TopKAccuracy:P2}");
        }

        /// <summary>
        /// Predicts the test data outcomes with the model
        /// 
        /// Creates test data.
        /// Predicts sentiment based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        /// <param name="model"></param>
        internal static void Predict(
            PredictionModel<ClassificationData, SentimentPrediction> model,
            InputData input)
        {
            // Use the model to predict the positive 
            // or negative sentiment of the comment data.
            IEnumerable<SentimentPrediction> predictions = model.Predict(input.Predicts);

            Console.WriteLine();
            Console.WriteLine("Classification Predictions");
            Console.WriteLine("--------------------------");

            // Builds pairs of (class, prediction)
            var classesAndPredictions = input.Predicts.Zip(predictions,
                                                          (classified, prediction) => (classified, prediction));

            foreach (var item in classesAndPredictions)
            {
                string textDisplay = item.classified.Text;

                if (textDisplay.Length > 80)
                    textDisplay = textDisplay.Substring(0, 75) + "...";

                string predictedClass = input.ClassNames[(uint)item.prediction.Class];

                Console.WriteLine("Prediction: {0}-{1} | Test: '{2}'",
                    item.prediction.Class, predictedClass, textDisplay);
            }
            Console.WriteLine();
        }
    }
}
