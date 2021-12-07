import pandas as pd
import os
import matplotlib.pyplot as plt

# Get all images, approximately 8000 each in train and test
SAMPLE_SIZE = 9000

module_dir = os.path.dirname(__file__)  # Set path to current directory

# Train Dataset creation
train_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/train-meta.xlsx')
train_data = pd.read_excel(train_meta_data_file_path).head(SAMPLE_SIZE)
train_counts = train_data.groupby(['class']).size()

# Test Dataset creation
test_meta_data_file_path = os.path.join(module_dir, 'Dataset/Metadata/test_meta.xlsx')
test_data = pd.read_excel(test_meta_data_file_path).head(SAMPLE_SIZE)
test_counts = test_data.groupby(['class']).size()


# Change to output directory
os.chdir("../Images")

# Train plot
train_counts.plot(kind='bar',use_index=True)
plt.xlabel("Class Label")
plt.ylabel("Sample Count")
plt.title("Training Sample by Class")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig('train_class_distribution.jpeg', dpi=100)
plt.show()

#Test plot
test_counts.plot(kind='bar',use_index=True)
plt.xlabel("Class Label")
plt.ylabel("Sample Count")
plt.title("Test Sample by Class")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig('test_class_distribution.jpeg', dpi=100)
