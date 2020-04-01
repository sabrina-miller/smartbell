from keras.models import load_model
import coremltools


#model.save('best_model.01-0.47.h5')

output_labels = ['Squat', 'Deadlift']
coreml_model = coremltools.converters.keras.convert('my_model2.h5', input_names=['accel'], output_names=['output'], class_labels=output_labels)

#your_model.author = 'Sophie Saunders'
#your_model.short_description = 'Squat/Deadlift recognition'
#your_model.input_description['accel'] = 'Takes x, y, z acceleration data as input'
#your_model.output_description['output'] = 'Prediction of Exercise'

coreml_model.save('smartbell_model.mlmodel')


#squat = pd.read_csv("data/Emily_DL_90.csv") # load in some csv data
#prediction = predict('best_model.50-0.00.h5', np.array(squat))

