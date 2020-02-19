import coremltools
import tfcoreml

#input_name = keras_model.inputs[0].name.split(':')[0]
#keras_output_node_name = keras_model.outputs[0].name.split(':')[0]
#graph_output_node_name = keras_output_node_name.split('/')[-1]

model = tfcoreml.convert('my_model.h5', output_feature_names=['dense_4/Softmax:0'])
model.save('final_coreml_model.mlmodel')
