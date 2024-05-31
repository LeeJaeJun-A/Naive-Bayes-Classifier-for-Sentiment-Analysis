import csv
import re
import heapq

class NaiveBayesClassifier:
    def __init__(self):
        """
        Initialize the NaiveBayesClassifier class.
        """
        # Initialize instance variables
        self.train_data = []  # List to store training data
        self.test_data = []  # List to store test data
        self.frequent_words = []  # List to store frequent words
        self.word_positive_probability = {}  # Dictionary to store the probability of each word in positive reviews
        self.word_negative_probability = {}  # Dictionary to store the probability of each word in negative reviews
        self.positive_probability = 0  # Probability of a positive review
        self.negative_probability = 0  # Probability of a negative review

        # Load stopwords from file and add additional contractions and common words
        with open("data/stopwords.txt", "r") as file:
            # Read stopwords from file and remove leading/trailing whitespaces
            self.stopwords = [word.strip() for word in file.readlines()]
            
            # Add additional contractions and common words to the stopwords list
            additional_stopwords = [
                "will", "can", "im", "ive", "dont", "am", "didnt", "who", "being", "youre", "wont", "doesnt", "may",
                "theres", "wouldnt", "isnt", "theyre", "havent", "youll", "werent", "arent", "weve", "hes", "youve",
                "whats", "hadnt", "shouldnt", "youd", "wasnt"
            ]
            self.stopwords.extend(additional_stopwords)
    
    def read_data(self, filename, mode = "train"):
        """
        Read data from a CSV file.

        Args:
            filename (str): The name of the CSV file.
            mode (str, optional): The mode of reading data. "train" or "test" mode is supported. Defaults to "train".
        """
        # Read data from CSV file
        with open(filename, "r", encoding="utf8") as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            data = list(reader)

            # Store the read data based on the mode
            if mode == "train":
                self.train_data = data  # Store the data for training
            elif mode == "test":
                self.test_data = data  # Store the data for testing
    
    def preprocess(self, mode = "train"):
        """
        Preprocess the data: lowercase, remove special characters, numbers, and stopwords.

        Args:
            mode (str, optional): The mode of preprocessing data. "train" or "test" mode is supported. Defaults to "train".
        """
        # Select the data based on the mode
        if mode == "train":
            data = self.train_data
        elif mode == "test":
            data = self.test_data

        # Return if no data is available
        if not data:
            return
        
        # Preprocess the data
        for review in range(len(data)):
            # Lowercase the text
            lower_text = data[review][1].lower()
            
            # Remove contractions and special characters
            lower_text = lower_text.replace("i'll", "")
            lower_text = lower_text.replace("i'd", "")
            special_characters_removed_text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', lower_text)
            
            # Remove numbers
            number_removed_text = re.sub(r'\d+', '', special_characters_removed_text) 
            
            # Replace double spaces to single space
            double_spaces_removed_text = re.sub(r'\s{2,}', ' ', number_removed_text)
            
            # Split the text into words
            words = double_spaces_removed_text.split()
            
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
        """
        Get the most frequent words from the training data.

        This function counts the occurrences of each word in the training data and
        stores the top 1000 words in the `frequent_words` attribute. It then prints
        the top 50 frequent words.

        Returns:
            None
        """
        # Return if no training data is available
        if not self.train_data:
            return
        
        # Count the occurrences of each word in the training data
        frequent_words = {}
        for review in range(len(self.train_data)):
            for word in self.train_data[review][1]:
                if word not in frequent_words:
                    frequent_words[word] = 1
                else:
                    frequent_words[word] += 1

        # Store the top 1000 frequent words
        self.frequent_words = [word for word, _ in heapq.nlargest(1000, frequent_words.items(), key=lambda item: item[1])]
        
        # Print the top 50 frequent words
        print("Top 50 Frequent Words: ")
        for word in self.frequent_words[:50]:
            print(word)
    
    def train(self, percentage):
        """
        Train the model using a specified percentage of the training data.

        Args:
            percentage (float): The percentage of the training data to use for training.
        """
        # Select the required portion of the training data
        data = self.train_data[:int(len(self.train_data) * (percentage / 100))]

        # Initialize counters for positive and negative reviews and words considering Laplace smoothing
        positive_word_count = [1] * len(self.frequent_words)
        negative_word_count = [1] * len(self.frequent_words)
        positive_reviews = 0
        negative_reviews = 0

        # Iterate over the selected data
        for review in range(len(data)):
            # If the review is positive
            if data[review][0] == 1:
                positive_reviews += 1
                # Update the word counts for positive reviews
                for word in data[review][1]:
                    if word in self.frequent_words:
                        index = self.frequent_words.index(word)
                        positive_word_count[index] += 1
            # If the review is negative
            else:
                negative_reviews += 1
                # Update the word counts for negative reviews
                for word in data[review][1]:
                    if word in self.frequent_words:
                        index = self.frequent_words.index(word)
                        negative_word_count[index] += 1

        # Calculate the word probabilities for positive and negative reviews considering Laplace smoothing
        self.word_positive_probability = {
            word: positive_word_count[self.frequent_words.index(word)] /
            (positive_reviews + len(self.frequent_words)) for word in self.frequent_words
        }
        self.word_negative_probability = {
            word: negative_word_count[self.frequent_words.index(word)] /
            (negative_reviews + len(self.frequent_words)) for word in self.frequent_words
        }

        # Calculate the class probabilities for positive and negative reviews
        self.positive_probability = positive_reviews / len(self.train_data)
        self.negative_probability = negative_reviews / len(self.train_data)

    def predict(self, text):
        """
        Predict the label of a given text.

        Args:
            text (list): The text to be predicted.

        Returns:
            int: The predicted label (1 for positive, 0 for negative).
        """
        # Initialize the probabilities with the class probabilities
        positive = self.positive_probability
        negative = self.negative_probability

        # Multiply the class probabilities by the word probabilities for each word in the text
        for word in text:
            # Check if the word is in the positive word probabilities
            if word in self.word_positive_probability:
                positive *= self.word_positive_probability[word]

            # Check if the word is in the negative word probabilities
            if word in self.word_negative_probability:
                negative *= self.word_negative_probability[word]

        # Return the predicted label based on the comparison of the positive and negative probabilities
        if positive > negative:
            return 1  # Positive label
        else:
            return 0  # Negative label

    def evaluate(self):
        """
        Evaluate the performance of the Naive Bayes Classifier on the test data.

        Returns:
            float: The accuracy of the classifier.
        """
        # Predict the labels for the test data
        prediction = [self.predict(review[1]) for review in self.test_data]
        
        # Get the correct labels for the test data
        correct_answer = [review[0] for review in self.test_data]
        
        # Initialize counters for prediction success and failure
        prediction_success = 0
        prediction_failure = 0
        
        # Iterate over the predictions and correct answers and count the number of correct predictions
        for val1, val2 in zip(prediction, correct_answer):
            if val1 == val2:
                prediction_success += 1
            else:
                prediction_failure += 1
        
        # Calculate the accuracy of the classifier
        accuracy = prediction_success / (prediction_success + prediction_failure)

        return accuracy
