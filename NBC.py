import csv
import re
import heapq

class NaiveBayesClassifier:
    def __init__(self):
        self.train_data = []
        self.test_data = []
        self.frequent_words = []
        self.word_positive_probability = {}
        self.word_negative_probability = {}
        self.positive_probability = 0
        self.negative_probability = 0
        with open("data/stopwords.txt", "r") as file:
            self.stopwords = [word.strip() for word in file.readlines()]
        self.stopwords += ["will", "can", "im", "ive", "dont", "am", "didnt", "who", "being", "youre", "wont", "doesnt", "may", "theres", "wouldnt",
                           "isnt", "theyre", "havent", "youll", "werent", "arent", "weve", "hes", "youve", "whats", "hadnt", "shouldnt", "youd",
                           "wasnt",]
# 축약형 문제. 축약형에서 기호를 빼면 다른 단어랑 같은 단어가 되는 경우가 있음
# 전체를 다 제거하기에는 무리가 있어서 최대 1000개 까지 나타내는 Frequency에서만 조동사 등을 뺐음. not을 포함하는 단어를 빼는 건 의도가 훼손될 수 있지않을까 생각했지만
# 기존 stopwords.txt에 이미 not이 포함되어있는 것을 보고 뺐음. can 같은 경우도 txt 파일에는 cannot이 들어있어서 포함시키는 것을 확인하고 뺐음.
# 단수, 복수, 과거형 등의 단어들을 다른 단어로 생각해서 계산하기 때문에 성능이 다소 낮은듯
    def read_data(self, filename, mode = "train"):
        with open(filename, "r", encoding="utf8") as file:
            reader = csv.reader(file)
            next(reader) # skip header
            data = list(reader)
            if mode == "train":
                self.train_data = data
            elif mode == "test":
                self.test_data = data
    
    def preprosses(self, mode = "train"):
        if mode == "train":
            data = self.train_data
        elif mode == "test":
            data = self.test_data

        if not data:
            return
        
        for review in range(len(data)):
            lower_text = data[review][1].lower()
            lower_text.replace("i'll", "")
            lower_text.replace("i'd", "")
            special_characters_removed_text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', lower_text)
            number_removed_text = re.sub(r'\d+', '', special_characters_removed_text) 
            double_spaces_removed_text = re.sub(r'\s{2,}', ' ', number_removed_text)
            words = double_spaces_removed_text.split()
            words_without_stopwords = [word for word in words if word not in self.stopwords]
            data[review][0] = 1 if data[review][0] == '5' else 0
            data[review][1] = words_without_stopwords

        if mode == "train":
            self.train_data = data
        elif mode == "test":
            self.test_data = data
  
    def get_frequent_words(self):
        if not self.train_data:
            return
        
        frequent_words = {}
        for review in range(len(self.train_data)):
            for word in self.train_data[review][1]:
                if word not in frequent_words:
                    frequent_words[word] = 1
                else:
                    frequent_words[word] += 1

        if len(frequent_words) >= 1000:
            max_word_count = 1000
        else:
            max_word_count = len(frequent_words)

        self.frequent_words = [word for word, _ in heapq.nlargest(max_word_count, frequent_words.items(), key=lambda item: item[1])]
        
        print("Top 50 Frequent Words: ")
        for word in self.frequent_words[:50]:
            print(word)
    
    def train(self, percentage):
        data = self.train_data[:int(len(self.train_data) * (percentage / 100))]
        positive_word_count = [1] * len(self.frequent_words)
        negative_word_count = [1] * len(self.frequent_words)
        positive_reviews = 0
        negative_reviews = 0

        for review in range(len(data)):
            if data[review][0] == 1:
                positive_reviews += 1
                for word in data[review][1]:
                    if word in self.frequent_words:
                        positive_word_count[self.frequent_words.index(word)] += 1
            else:
                negative_reviews += 1
                for word in data[review][1]:
                    if word in self.frequent_words:
                        negative_word_count[self.frequent_words.index(word)] += 1

        self.word_positive_probability = {word: positive_word_count[self.frequent_words.index(word)] / (positive_reviews + len(self.frequent_words)) for word in self.frequent_words}
        self.word_negative_probability = {word: negative_word_count[self.frequent_words.index(word)] / (negative_reviews + len(self.frequent_words)) for word in self.frequent_words}

        self.positive_probability = positive_reviews / len(self.train_data)
        self.negative_probability = negative_reviews / len(self.train_data)

    def predict(self, text):
        positive = self.positive_probability 
        negative = self.negative_probability

        for word in text:
            if word in self.word_positive_probability:
                positive *= self.word_positive_probability[word]

            if word in self.word_negative_probability:
                negative *= self.word_negative_probability[word]

        if positive > negative:
            return 1
        else:
            return 0

    def evaluate(self):
        prediction = [self.predict(review[1]) for review in self.test_data]
        correct_answer = [review[0] for review in self.test_data]

        prediction_success = 0
        prediction_failure = 0

        for val1, val2 in zip(prediction, correct_answer):
            if val1 == val2:
                prediction_success += 1
            else:
                prediction_failure += 1
        
        accuracy = prediction_success / (prediction_success + prediction_failure)

        return accuracy
