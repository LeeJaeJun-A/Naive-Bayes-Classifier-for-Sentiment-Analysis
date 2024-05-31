from NBC import NaiveBayesClassifier
from draw import drawPlot

if __name__ == "__main__":
    tran_file_path = "data/train.csv"
    test_file_path = "data/test.csv"

    NaiveBayesClassifier = NaiveBayesClassifier()
    NaiveBayesClassifier.read_data(tran_file_path, "train")
    NaiveBayesClassifier.read_data(test_file_path, "test")
    NaiveBayesClassifier.preprosses("train")
    NaiveBayesClassifier.get_frequent_words()
    NaiveBayesClassifier.preprosses("test")

    percentages = [10, 30, 50, 70, 100]
    accuracy = {}

    for percentage in percentages:
        NaiveBayesClassifier.train(percentage)
        accuracy[percentage] = NaiveBayesClassifier.evaluate()

    drawPlot(accuracy)
    