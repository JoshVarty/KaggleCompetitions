import model1
import model2
import model3
import numpy as np

model_save_path1 = "model/model.ckpt" 

model1.TrainConvNet(model_save_path1)
results1 = model1.LoadAndRun(model_save_path1)

with open("results/results.csv", 'w') as file:
    file.write("id,label\n")
    for idx in range(len(results1)):
         prediction = int(results1[idx])

         file.write(str(idx + 1))
         file.write(",")
         file.write(str(prediction))
         file.write("\n")