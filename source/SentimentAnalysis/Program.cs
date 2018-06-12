using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

/***
 * Warning: Format error at (83,3)-(83,4011): Illegal quoting
 * Processed 251 rows with 0 bad values and 1 format errors
 * Warning: Format error at (83,3)-(83,4011): Illegal quoting
 * Processed 251 rows with 0 bad values and 1 format errors
 * Warning: Format error at (83,3)-(83,4011): Illegal quoting
 * Processed 251 rows with 0 bad values and 1 format errors
 * Not adding a normalizer.
 * Making per-feature arrays
 * Changing data from row-wise to column-wise
 *   Bad value at line 1 in column Label
 * Warning: Format error at (83,3)-(83,4011): Illegal quoting
 * Processed 251 rows with 1 bad values and 1 format errors
 * Processed 250 instances
 * Binning and forming Feature objects
 * Reserved memory for tree learner: 1900332 bytes
 * Starting to train ...
 * Not training a calibrator because it is not needed.
 *   Bad value at line 1 in column Label
 * Processed 19 rows with 1 bad values and 0 format errors
 * 
 * PredictionModel quality metrics evaluation
 * ------------------------------------------
 * Accuracy: 66.67%
 * Auc: 94.44%
 * F1Score: 75.00%
 * 
 * Sentiment Predictions
 * ---------------------
 * Sentiment: Please refrain from adding nonsense to Wikipedia. | Prediction: Negative
 * Sentiment: He is the best, and the article should say that. | Prediction: Positive
 * 
 ***/
namespace SentimentAnalysis
{
    /// <summary>
    /// Tutorial on ML.Net:
    /// https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis
    /// </summary>
    class Program
    {
        const string _dataPath = @".\Data\wikipedia-detox-250-line-data.tsv";
        const string _testDataPath = @".\Data\wikipedia-detox-250-line-test.tsv";
        const string _modelpath = @".\Data\Model.zip";

        static async Task Main(string[] args)
        {
            // Get a model trained to use for evaluation
            var model = await TrainAsync();

            Evaluate(model);

            Predict(model);

            Console.ReadKey();
        }

        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> TrainAsync()
        {
            // LearningPipeline allows you to add steps in order to keep everything together 
            // during the learning process.  
            var pipeline = new LearningPipeline();

            // The TextLoader loads a dataset with comments and corresponding postive or negative sentiment. 
            // When you create a loader, you specify the schema by passing a class to the loader containing
            // all the column names and their types. This is used to create the model, and train it. 
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());

            // TextFeaturizer is a transform that is used to featurize an input column. 
            // This is used to format and clean the data.
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            // Adds a FastTreeBinaryClassifier, the decision tree learner for this project, and 
            // three hyperparameters to be used for tuning decision tree performance.
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            // Train the pipeline based on the dataset that has been loaded, transformed.
            PredictionModel<SentimentData, SentimentPrediction> model =
                                pipeline.Train<SentimentData, SentimentPrediction>();

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
        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            // loads the new test dataset with the same schema.
            // You can evaluate the model using this dataset as a quality check.
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();

            // Computes the quality metrics for the PredictionModel using the specified dataset.
            var evaluator = new BinaryClassificationEvaluator();

            // The BinaryClassificationMetrics contains the overall metrics computed by binary
            // classification evaluators. To display these to determine the quality of the model,
            // you need to get the metrics first.
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            // Displaying the metrics for model validation
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
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
        public static void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            // Adds some comments to test the trained model's predictions.
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Please refrain from adding nonsense to Wikipedia."
                },
                new SentimentData
                {
                    SentimentText = "He is the best, and the article should say that."
                }
            };

            // Use the model to predict the positive 
            // or negative sentiment of the comment data.
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            // Builds pairs of (sentiment, prediction)
            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();
        }
    }
}
