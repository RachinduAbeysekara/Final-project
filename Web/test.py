import pickle

# Load the trained model
with open('./model/lung_cancer_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Get the input shape of the model
input_shape = model.input_shape
print("Input shape:", input_shape)