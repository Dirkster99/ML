namespace BinaryClass
{
    using System.Collections.Generic;

    internal class InputData
    {
        public InputData(string trainingData,
                         string testData,
                         IEnumerable<ClassificationData> predicts,
                         string trueClassName,
                         string falseClassName)
        {
            TrainingData = trainingData;
            TestData = testData;
            Predicts = predicts;

            TrueClassName = trueClassName;
            FalseClassName = falseClassName;
        }

        public string TrainingData { get; }

        public string TestData { get; }

        public IEnumerable<ClassificationData> Predicts { get; }

        public string TrueClassName { get; }

        public string FalseClassName { get; }
    }
}
