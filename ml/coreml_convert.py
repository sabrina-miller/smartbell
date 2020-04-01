import coremltools
import tfcoreml
import tensorflow as tf


tfcoreml.convert(tf_model_path='my_model2.h5', 
            mlmodel_path="it_worked.mlmodel", 
            #input_name_shape_dict={"reshape_1_input:0": [None, 120]},
            output_feature_names=['dense_4/Softmax:0'])



