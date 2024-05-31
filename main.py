from NBC import NaiveBayesClassifier
from draw import drawPlot

if __name__ == "__main__":
    tran_file_path = "data/train.csv"
    test_file_path = "data/test.csv"

    NaiveBayesClassifier = NaiveBayesClassifier()

    NaiveBayesClassifier.read_data(tran_file_path, "train")
    NaiveBayesClassifier.preprocess("train")
    NaiveBayesClassifier.get_frequent_words()

    NaiveBayesClassifier.read_data(test_file_path, "test")
    NaiveBayesClassifier.preprocess("test")

    percentages = [10, 30, 50, 70, 100]
    accuracy = {}

    # Train and evaluate the model for each percentage
    for percentage in percentages:
        NaiveBayesClassifier.train(percentage)
        accuracy[percentage] = NaiveBayesClassifier.evaluate()

    drawPlot(accuracy)

# 축약형 문제. 축약형에서 기호를 빼면 다른 단어랑 같은 단어가 되는 경우가 있음
# 전체를 다 제거하기에는 무리가 있어서 최대 1000개 까지 나타내는 Frequency에서만 조동사 등을 뺐음. not을 포함하는 단어를 빼는 건 의도가 훼손될 수 있지않을까 생각했지만
# 기존 stopwords.txt에 이미 not이 포함되어있는 것을 보고 뺐음. can 같은 경우도 txt 파일에는 cannot이 들어있어서 포함시키는 것을 확인하고 뺐음.
# 단수, 복수, 과거형 등의 단어들을 다른 단어로 생각해서 계산하기 때문에 성능이 다소 낮은듯