import os from 'os';
import bodyParser from 'body-parser';
import cors from 'cors';
import express from 'express';
import morgan from 'morgan';

import { Tensor, InferenceSession } from 'onnxjs';

const registerMiddleware = async app => {
  const env = process.env.NODE_ENV || 'development';

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
  return app;
};

const registerRoutes = app => {
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
    try {
      const { values } = req.body;

      const dim = Math.floor(Math.sqrt(values.length));

      const inputs = [new Tensor(new Float32Array(values), 'float32', [dim, dim])];

      const outputMap = await app.locals.session.run(inputs);
      const outputTensor = outputMap.values().next().value;

      res.status(200).json({ result: outputTensor });
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  });

  return app;
};

const createSession = async () => {
  const session = new InferenceSession();
  const url = process.env.ONNX_MODEL_URI || './src/data/model.onnx';

  await session.loadModel(url);

  return session;
};

const createApp = async () => {
  const app = express();

  registerMiddleware(app);
  registerRoutes(app);

  const session = await createSession();
  app.locals.session = session;

  return app;
};

export default createApp;
