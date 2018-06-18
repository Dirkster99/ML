namespace MultiClass
{
    using System.Collections.Generic;

    internal class InputData
    {
        public InputData(string trainingData,
                         string testData,
                         IEnumerable<ClassificationData> predicts,
                         string[] classNames)
        {
            TrainingData = trainingData;
            TestData = testData;
            Predicts = predicts;

            ClassNames = classNames;
        }

        public string TrainingData { get; }

        public string TestData { get; }

        public IEnumerable<ClassificationData> Predicts { get; }

        public string[] ClassNames { get; }
    }
}
