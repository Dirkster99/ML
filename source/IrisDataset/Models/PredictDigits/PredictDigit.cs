namespace MultiClass.Models.PredictDigits
{
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Trainers;
    using Microsoft.ML.Transforms;
    using System;
    using System.IO;

    public class PredictDigit
    {
        public PredictDigit()
        {
            var dataPath = Path.Combine("SevenSegment", "segments.txt");
            var pipeline = new LearningPipeline
            {
                new TextLoader(dataPath).CreateFrom<Digit>(separator: ',', allowQuotedStrings:false),
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
            Console.ReadLine();
        }
    }
}
