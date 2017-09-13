import model1

model_save_path1 = "model/model.ckpt"
model_save_path2 = "model/model2.ckpt"
model_save_path3 = "model/model3.ckpt"
model_save_path4 = "model/model4.ckpt"
model_save_path5 = "model/model5.ckpt"
model_save_path6 = "model/model6.ckpt"
model_save_path7 = "model/model7.ckpt"

#Three of the first model
results1 = model1.LoadAndRun(model_save_path1)


with open("results/results.csv", 'w') as file:
    file.write("ImageId,Label\n")
    for idx in range(len(results1)):
         pred1 = int(results1[idx])

         nums = np.zeros(10)
         nums[pred1] = nums[pred1] + 1 

         prediction = np.argmax(nums)

         file.write(str(idx + 1))
         file.write(",")
         file.write(str(prediction))
         file.write("\n")