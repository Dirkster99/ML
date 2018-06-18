namespace MultiClass.Models.PredictDigits
{
    using Microsoft.ML.Runtime.Api;

    public class DigitPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint ExpectedDigit;

        [ColumnName("Score")]
        public float[] Score;
    }
}
