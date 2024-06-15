import matplotlib.pyplot as plt

def drawPlot(dict):
    amount_of_training_data = list(dict.keys())
    accuracy = list(dict.values())
    
    # Draw the plot
    plt.plot(amount_of_training_data, accuracy)
    plt.scatter(amount_of_training_data, accuracy, color='r', zorder=5)
    
    # Set the x-axis ticks
    plt.xticks(range(0, 110, 10))
    
    # Add labels and titles
    plt.title("Model Accuracy with Varying Training Data Sizes")
    plt.xlabel("Amount of training data (%)")
    plt.ylabel("Accuracy")
    
    # Add accuracy values as text on the plot
    for index, (x, y) in enumerate(zip(amount_of_training_data, accuracy)):
        plt.text(x, y + 0.0003, f'{y:.4f}', fontsize=9, ha='right')
    
    # Display the plot
    plt.show()
