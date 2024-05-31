import matplotlib.pyplot as plt

def drawPlot(dict):
    amount_of_training_data = list(dict.keys())
    accuracy = list(dict.values())
    # plt.figure(figsize=(10, 5))
    plt.plot(amount_of_training_data, accuracy)
    plt.scatter(amount_of_training_data, accuracy, color='r', zorder=5)
    plt.xticks(range(0, 110, 10))
    for _, (x, y) in enumerate(zip(amount_of_training_data, accuracy)):
        plt.text(x, y + 0.001, f'{y:.2f}', fontsize=9, ha='right')
    plt.title("Model Accuracy with Varying Training Data Sizes")
    plt.xlabel("Amount of training data(%)")
    plt.ylabel("Accuracy")
    plt.show()