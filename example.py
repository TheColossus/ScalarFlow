import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from engine import MLP, train_mlp, mean_squared_error
from kernel import Scalar

#Input data
#Replace the path to the csv as needed. If you keep it like this, ensure you run the program from the 'example' directory
dataset = pd.read_csv("heart.csv")

targetColumn = "output"
inputColumnNames = dataset.columns.drop("output")
outputRows = dataset[targetColumn]
dataset.drop(targetColumn,axis=1,inplace=True)

#Clean data
#Normalize input data to between 0 and 1
scaler = MinMaxScaler()
scaler.fit(dataset)
transformedDataset = scaler.transform(dataset)

#Split the data into a training set and a testing set
xTrain, xTest, yTrain, yTest = train_test_split(transformedDataset, outputRows, test_size=0.25, random_state=0)
xTrain = xTrain.tolist()
yTrain = yTrain.tolist()
xTest = xTest.tolist()
yTest = yTest.tolist()

xTrainScalars = [[Scalar(val) for val in row] for row in xTrain]
yTrainScalars = [Scalar(val) for val in yTrain]
xTestScalars = [[Scalar(val) for val in row] for row in xTest]

#Build the model
#2 layer MLP with 13 dimensional inputs (The dataset has 13 input columns)
model = MLP(13, [15, 15, 1], hidden_activation='relu', output_activation='sigmoid')

print("Starting training...")
train_mlp(model, xTrainScalars, yTrainScalars, batch_size=32, epochs=100, learning_rate=0.01, loss_fn=mean_squared_error)

#Test the model
yPredictions = [model(x) for x in xTestScalars]

#Pull the data property out of each Scalar
yPredictionsClean = [x.data for x in yPredictions]

#Push the data through a threshold to make it binary

yPredictionsFinal = [0 if x < 0.5 else 1 for x in yPredictionsClean]

cm = confusion_matrix(yTest, yPredictionsFinal)
accuracy = accuracy_score(yTest, yPredictionsFinal)

TN, FP, FN, TP = confusion_matrix(yTest, yPredictionsFinal).ravel()

print('\n=== RESULTS ===')
print(f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'True Positive(TP)  = {TP}')
print(f'False Positive(FP) = {FP}')
print(f'True Negative(TN)  = {TN}')
print(f'False Negative(FN) = {FN}')

print('\nConfusion Matrix:')
print(cm)


