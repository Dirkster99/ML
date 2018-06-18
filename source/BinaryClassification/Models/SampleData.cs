namespace Models.BinaryClass
{
    using System.Collections.Generic;

    internal class InputData
    {
        public InputData(string trainingData,
                         string testData,
                         IEnumerable<ClassificationData> predicts,
                         string[] className)
        {
            TrainingData = trainingData;
            TestData = testData;
            Predicts = predicts;

            ClassName = className;
        }

        public string TrainingData { get; }

        public string TestData { get; }

        public IEnumerable<ClassificationData> Predicts { get; }

        public string[] ClassName { get; }
    }
}
