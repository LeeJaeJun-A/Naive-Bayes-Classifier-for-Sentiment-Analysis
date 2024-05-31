import matplotlib.pyplot as plt


def drawPlot(dict):
    """
    This function is used to draw a plot of the model accuracy with varying training data sizes.

    Parameters:
    dict (dict): A dictionary where the keys are the training data sizes and the values are the corresponding model accuracies.
    """
    # Get the keys and values from the dictionary
    amount_of_training_data = list(dict.keys())
    accuracy = list(dict.values())
    
    # Draw the plot
    plt.plot(amount_of_training_data, accuracy)  # Plot the model accuracy
    plt.scatter(amount_of_training_data, accuracy, color='r', zorder=5)  # Scatter plot for clear visualization
    
    # Set the x-axis ticks
    plt.xticks(range(0, 110, 10))
    
    # Add labels and titles
    plt.title("Model Accuracy with Varying Training Data Sizes")  # Title of the plot
    plt.xlabel("Amount of training data (%)")  # Label for the x-axis
    plt.ylabel("Accuracy")  # Label for the y-axis
    
    # Add accuracy values as text on the plot
    for index, (x, y) in enumerate(zip(amount_of_training_data, accuracy)):
        plt.text(x, y + 0.0015, f'{y:.4f}', fontsize=9, ha='right')
    
    # Display the plot
    plt.show()
