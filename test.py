import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('best.onnx', providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 800, 800).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})
print("Inference successful with CUDAExecutionProvider")
