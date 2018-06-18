namespace MultiClass.Models.PredictDigits
{
    using Microsoft.ML.Runtime.Api;

    public class Digit
    {
        public float Up { get => Features[0]; set => Features[0] = value; }

        public float Middle { get => Features[1]; set => Features[1] = value; }

        public float Bottom { get => Features[2]; set => Features[2] = value; }

        public float UpLeft { get => Features[3]; set => Features[3] = value; }

        public float BottomLeft { get => Features[4]; set => Features[4] = value; }

        public float TopRight { get => Features[5]; set => Features[5] = value; }

        public float BottomRight { get => Features[6]; set => Features[6] = value; }

        [Column("0-6")]
        [VectorType(7)] public float[] Features = new float[7];

        [Column("7")]
        [ColumnName("Label")] public float Label;
    }
}
