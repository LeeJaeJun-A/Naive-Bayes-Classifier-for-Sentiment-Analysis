import csv
import re

class NaiveBayesClassifier:
    def __init__(self):
        self.train_data = []
        self.test_data = []
        self.top_1000_frequent_words = []
        self.probability_of_word_positive = {}
        self.probability_of_word_negative = {}
        self.probability_of_label_positive = 0
        self.probability_of_label_negative = 0

        with open("data/stopwords.txt", "r") as file:
            stopwords = []
            for line in file.readlines():
                word = line.strip()
                stopwords.append(word)
            self.stopwords = stopwords
    
    # mode: "train" or "test"
    def read_data(self, filename, mode = "train"):
        with open(filename, "r", encoding="utf8") as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            data = list(reader)

            if mode == "train":
                self.train_data = data
            elif mode == "test":
                self.test_data = data
    
    # mode: "train" or "test"
    def preprocess(self, mode = "train"):
        if mode == "train":
            data = self.train_data
        elif mode == "test":
            data = self.test_data
        
        # Preprocess the data
        for review in range(len(data)):
            lower_text = data[review][1].lower()

            special_characters_removed_text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', lower_text)
            
            # Split the text into words
            words = special_characters_removed_text.split()
            
            # Remove stopwords
            words_without_stopwords = [word for word in words if word not in self.stopwords]
            
            # Update the label
            data[review][0] = 1 if data[review][0] == '5' else 0
            
            # Update the preprocessed text
            data[review][1] = words_without_stopwords

        # Store the preprocessed data based on the mode
        if mode == "train":
            self.train_data = data
        elif mode == "test":
            self.test_data = data

    def get_frequent_words(self):    
        # Count the occurrences of each word in the training data
        frequent_words = {}
        for review in range(len(self.train_data)):
            for word in self.train_data[review][1]:
                if word not in frequent_words:
                    frequent_words[word] = 1
                else:
                    frequent_words[word] += 1

        # Store the top 1000 frequent words
        word_frequencies = list(frequent_words.items())
        word_frequencies.sort(key=lambda item: item[1], reverse=True)
        top_1000_words = [word for word, _ in word_frequencies[:1000]]
        self.top_1000_frequent_words = top_1000_words

        # Print the top 50 frequent words
        print("Top 50 Frequent Words: ")
        for i in range(50):
            print(self.top_1000_frequent_words[i])
    
    def train(self, percentage):
        # Select the required portion of the training data
        data_num = int(len(self.train_data) * (percentage / 100))
        data = self.train_data[:data_num]

        # Apply laplace smoothing to the word counts
        positive_word_count = [0] * len(self.top_1000_frequent_words)
        negative_word_count = [0] * len(self.top_1000_frequent_words)

        positive_reviews = 0
        negative_reviews = 0

        for review in data:
            if review[0] == 1: # If the review is positive
                positive_reviews += 1
                for word in review[1]:
                    if word in self.top_1000_frequent_words:
                        index = self.top_1000_frequent_words.index(word)
                        positive_word_count[index] += 1
            else: # If the review is negative
                negative_reviews += 1
                for word in review[1]:
                    if word in self.top_1000_frequent_words:
                        index = self.top_1000_frequent_words.index(word)
                        negative_word_count[index] += 1
        
        total_positive_word_count = sum(positive_word_count)
        total_negative_word_count = sum(negative_word_count)
        
        denominator_positive = total_positive_word_count + len(self.top_1000_frequent_words)
        denominator_negative = total_negative_word_count + len(self.top_1000_frequent_words)

        # Calculate the word probabilities considering Laplace smoothing
        self.probability_of_word_positive = {word: (positive_word_count[self.top_1000_frequent_words.index(word)] + 1) / denominator_positive for word in self.top_1000_frequent_words}
        self.probability_of_word_negative = {word: (negative_word_count[self.top_1000_frequent_words.index(word)] + 1) / denominator_negative for word in self.top_1000_frequent_words}

        # Calculate the class probabilities for positive and negative reviews
        self.probability_of_label_positive = positive_reviews / data_num
        self.probability_of_label_negative = negative_reviews / data_num

    def predict(self, text):
        positive_probability = self.probability_of_label_positive
        negative_probability = self.probability_of_label_negative

        for word in text:
            if word in self.probability_of_word_positive: # Check if the word is in the positive word probabilities
                positive_probability *= self.probability_of_word_positive[word]

            if word in self.probability_of_word_negative: # Check if the word is in the negative word probabilities
                negative_probability *= self.probability_of_word_negative[word]

        if positive_probability > negative_probability:
            return 1  # Positive label
        else:
            return 0  # Negative label

    # Evaluate the accuracy of the Naive Bayes Classifier
    def evaluate(self):
        # Predict the labels for the test data
        predictions = []
        for review in self.test_data:
            review_text = review[1]
            prediction = self.predict(review_text)
            predictions.append(prediction)
        
        # Get the correct labels for the test data
        correct_answer = [review[0] for review in self.test_data]

        prediction_success_count = 0
        prediction_failure_count = 0
        
        for val1, val2 in zip(predictions, correct_answer):
            if val1 == val2:
                prediction_success_count += 1
            else:
                prediction_failure_count += 1
        
        accuracy = prediction_success_count / (prediction_success_count + prediction_failure_count)

        return accuracy
