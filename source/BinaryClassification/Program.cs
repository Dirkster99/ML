using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Models.BinaryClass;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace BinaryClass
{
    /// <summary>
    /// Tutorial on ML.Net:
    /// https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis
    /// </summary>
    public class Program
    {
        #region SentimentInputData
        static readonly IEnumerable<ClassificationData> predictSentimentsData = new[]
        {
            new ClassificationData
            {
                Text = "Please refrain from adding nonsense to Wikipedia."
            },
            new ClassificationData
            {
                Text = "He is the best, and the article should say that."
            }
        };

        static string[] sentimentClassNames = { "Positive", "Negative" };

        private static readonly InputData sentimentInputData = new InputData
        (
            @".\Data\Sentiment\wikipedia-detox-250-line-data.tsv",
            @".\Data\Sentiment\wikipedia-detox-250-line-test.tsv",
            predictSentimentsData,
            sentimentClassNames
        );
        #endregion SentimentInputData

        #region You have Got Spam InputData
        // Adds some comments to test the trained model's predictions.
        static readonly IEnumerable<ClassificationData> predictSpamData = new[]
        {
                new ClassificationData
                {
                    Text = "Bust by this expressing at stepped and. My my dreary a and. Shaven we spoken minute beguiling my have gloated his fancy wandering back throws though. The chamber that rapping. So terrors is fast grim so of this grew from heard unto the land linking. Censer that and door the deep on word token and stayed. Door as the home maiden and gave surely some. Form sculptured soul quoth before both you whether for fact lost betook fowl meaninglittle implore the as unbroken the. A nodded quaff swung censer lenore tapping the morrow raven bird the. Hath lamplight the or beak expressing on remember little quoth. Disaster by mortals there that before. Dreary demons agreeing sinking thy denser mefilled visiter tapping land tossed unbroken with.</p><p>Came thy or thy the lenore reply perched bust that. From chamber i suddenly. It midnight some flown perched lamplight purple each forgiveness i of sat my lenore long the. Hesitating youhere then kind had lost then on of chamber respiterespite of. Is with murmured pondered chamber. He its door more i no tufted. Into the darkness and. By i streaming before nevermore as open dream i purple nevermore curious stayed before. My your so. That the lining his thrilled. With stately stepped in is an my then lenore that felt lies followed before oer i a thy pallid. That marvelled you whose shorn muttered more. A fowl stood feather.</p><p>Fearing core floor what heart. Ebony that and the as cushions unmerciful unhappy no i shadows prophet muttered that tossed at nothing. Word back i is he and shall his beating discourse still i ghost. Still angels be black is raven land oer though no. He at stopped that while there is a the the this what my take the chamber my spoken. Merely whom thy have stronger. Chamber parting name there bird scarcely home the into if. Nevermore hope uncertain each repeating kind press the the hear all that nothing the the. Open was at raven my. Before in discourse be be word. Token token and the i sign lining though volume of the tis quoth of shadows hesitating. Have what of and with more he in stock let door nothing bird quoth marvelled door this a. Sad whom faintly fowl i i explore upon out seraphim the one. Back flown nevermore croaking your the forget chamber wind some the some. The a mystery ominous above so heard lordly ah some days dreaming spoken cushions. For what nothing the but. And mefilled chamber whether there loneliness me your tempter if velvet."
                },
                new ClassificationData
                {
                    Text = "So his chaste my. Mote way fabled as of aye from like old. Goodly rill near himnot den than to his none visit joyless. Shades climes that revellers had lyres by taste ways passed. To harold mote that earthly heralds at made sight of to a shamed once satiety left along he.</p><p>Heart he the evil she bower wassailers shades with one take loathed in of. Adieu he only shun come condole wight he he land these mammon shameless rhyme land. His the ancient he nor. Grief nor he nor amiss hour know by dear none like he. And to been a moths worse from his mother haply. Earth there fabled the they none his cared who labyrinth could sun reverie from fulness was so the. Thence birth fame where smile still and it and sacred far. Feere climes be sacred whilome from nor wins heavenly harold days unto the. Counsel harold a at. Flatterers eremites to. That and scape long and his it fall ofttimes condemned he sister lurked ne resolved olden. Most vile fellow where are he love he consecrate <a href=\"https://www.but.com\">muse</a> sorrow was. The break suits waste friend at one once.</p><p>Go <a href=\"https://www.sick.com\">are</a> which to but bower it things thy. Disappointed sad from true but are mirth on will below open mammon in mammon chill was. Sea that breast and other from earth the the of will kiss who venerable. A sadness of there and crime <a href=\"https://www.for.com\">like</a> and pile pillared. Suffice thy along did or had him none was vile this. Low sighed each lands. Once with a pleasure vaunted objects by strength and consecrate lines save the lowly bower before plain. Joyless fly he that. Condemned strange and known girls few grief found nor minstrels one he goodly her did glee fountain. Counsel scape from ive land and joyless before artless. Done nine to him."
                }
        };

        static string[] spamClassNames = { "Spam", "No Spam" };

        private static readonly InputData spamInputData = new InputData
        (
            @".\Data\YouGotSpam\training.tsv",
            @".\Data\YouGotSpam\test.tsv",
            predictSpamData,
            spamClassNames
        );
        #endregion You have Got Spam InputData

        const string _modelpath = @".\Data\Model.zip";

        public static void Main(string[] args)
        {
            InputData input = sentimentInputData;
            //InputData input = spamInputData;

            Task.Run(async () =>
            {
                // Get a model trained to use for evaluation
                var model = await TrainAsync(input);

                Evaluate(model, input);

                Predict(model, input);

                Console.ReadKey();

            }).GetAwaiter().GetResult();
        }

        internal static async Task<PredictionModel<ClassificationData, ClassPrediction>>
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

            // TextFeaturizer is a transform that is used to featurize an input column. 
            // This is used to format and clean the data.
            pipeline.Add(new TextFeaturizer("Features", "Text"));

            // Adds a FastTreeBinaryClassifier, the decision tree learner for this project, and 
            // three hyperparameters to be used for tuning decision tree performance.
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            // Train the pipeline based on the dataset that has been loaded, transformed.
            PredictionModel<ClassificationData, ClassPrediction> model =
                                pipeline.Train<ClassificationData, ClassPrediction>();

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
            PredictionModel<ClassificationData, ClassPrediction> model,
            InputData input)
        {
            // loads the new test dataset with the same schema.
            // You can evaluate the model using this dataset as a quality check.

            //var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
            var testData = new TextLoader(input.TestData).CreateFrom<ClassificationData>();

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
            Console.WriteLine($"     Auc: {metrics.Auc:P2}");
            Console.WriteLine($" F1Score: {metrics.F1Score:P2}");
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
            PredictionModel<ClassificationData, ClassPrediction> model,
            InputData input)
        {
            // Use the model to predict the positive 
            // or negative sentiment of the comment data.
            IEnumerable<ClassPrediction> predictions = model.Predict(input.Predicts);

            Console.WriteLine();
            Console.WriteLine("Classification Predictions");
            Console.WriteLine("--------------------------");

            // Builds pairs of (sentiment, prediction)
            var sentimentsAndPredictions = input.Predicts.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                string textDisplay = item.sentiment.Text;

                if (textDisplay.Length > 80)
                    textDisplay = textDisplay.Substring(0, 75) + "...";

                Console.WriteLine($"Prediction: {(item.prediction.Class ? input.ClassName[0] : input.ClassName[1])}" + " | " +
                                  $"Text: {textDisplay}");
            }
            Console.WriteLine();
        }
    }
}
