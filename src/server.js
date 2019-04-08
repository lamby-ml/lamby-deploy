import os from 'os';
import bodyParser from 'body-parser';
import cors from 'cors';
import express from 'express';
import morgan from 'morgan';

import { Tensor } from 'onnxjs';

const env = process.env.NODE_ENV || 'development';

const app = express();

// Setup middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Configure logging
if (env === 'production') {
  // Use verbose logger if in production environment
  app.use(morgan('combined'));
} else if (env === 'testing') {
  // Do not log anything if in testing environment
} else {
  // Use simple logger if in development environment
  app.use(morgan('dev'));
}

app.get('/', async (req, res) => {
  const message = `
    <h1>Your API is up and running successfully!</h1>
    <h3>Here is an example on how to use your model to predict values</h3>
    <p>
      curl -H "Content-Type: application/json" -X POST
        -d '{ "values": [1.0,2.0,3.0,4.0] }' http://${os.hostname()}/predict
    </p>`;
  res.send(message);
});

app.post('/predict', async (req, res) => {
  const { values } = req.body;
  const inputs = [new Tensor(new Float32Array(values), 'float32', [2, 2])];
  res.status(200).json({ inputs });

  // const outputMap = await session.run(inputs);
  // const outputTensor = outputMap.values().next().value;
  // res.send({ outputTensor });
});

export default app;
