const { Tensor, InferenceSession } = require('onnxjs');
const express = require('express');

const app = express();
const session = new InferenceSession({ backendHint: 'cpu' });
const port = process.env.LAMBY_DEPLOY_PORT || 3000;

async function evaluate(req, res) {
  const { data, shape } = req.body;
  console.log(`Input received of shape: ${shape}`);
  try {
    const tensor = new Tensor(new Float32Array(data), 'float32', shape);
    const map = await session.run([tensor]);
    res.send({ output: map.values().next().value.data });
  } catch (err) {
    console.error(err);
    res
      .status(500)
      .send({ message: 'There was an error in the evaluation of your input' });
  }
}

// eslint-disable-next-line no-unused-vars
function zeroTensor(shape) {
  const numElements = shape.reduce((a, b) => a * b);
  return new Tensor(new Float32Array(numElements), 'float32', shape);
}

session
  .loadModel('./model.onnx')
  .then(() => console.log('Onnx model successfully loaded.'));

app.use(express.json());
app.post('/eval', evaluate);
app.listen(port, () => console.log(`Server running on port ${port}`));
