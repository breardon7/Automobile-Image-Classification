# packages
from ModelTraining import model_definition, model_predict

# Define models
model_vgg19 = model_definition(pretrained=True)
#model_CNN = model_definition(pretrained=False)

# Predict & run classification report
model_predict(model=model_vgg19)
#model_predict(model=model_CNN)