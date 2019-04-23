import http from 'http';
import os from 'os';
import path from 'path';

import bodyParser from 'body-parser';
import cors from 'cors';
import express from 'express';
import morgan from 'morgan';

import { Tensor, InferenceSession } from 'onnxjs';

let MODEL_SHAPE = [];
const COMMIT_ID = process.env.ONNX_COMMIT_ID;

const registerMiddleware = async app => {
  const env = process.env.NODE_ENV || 'development';

  app.set('views', path.join(__dirname, 'templates'));
  app.set('view engine', 'ejs');
  app.use(cors());
  app.use('/static', express.static('src/public'));
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
    if (req.query.id !== 'accuracy') {
      res.render('index.ejs', {
        hostname: os.hostname(),
        commit_id: '5d7c65',
        model_shape: JSON.stringify(MODEL_SHAPE)
      });
    }
  });

  app.post('/predict', async (req, res) => {
    try {
      const { values, dim } = req.body;

      const inputs = [new Tensor(new Float32Array(values), 'float32', dim)];

      const outputMap = await app.locals.session.run(inputs);
      const outputTensor = outputMap.values().next().value;

      res.status(200).json({ result: outputTensor });
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  });

  app.post('/accuracy', async (req, res) => {
    try {
      const { val, lab, outputApply, dim, name } = req.body;

      let output_apply_func = null;
      if (outputApply === 'max') {
        output_apply_func = map => {
          return Object.keys(map).reduce((a, b) => (map[a] > map[b] ? a : b));
        };
      }

      let labels = [];

      let correct = 0;
      for (var i = 0; i < val.length; i++) {
        const outputMap = await app.locals.session.run([
          new Tensor(new Float32Array(val[i]), 'float32', dim)
        ]);
        const outputTensor = outputMap.values().next().value;
        let output = output_apply_func(outputTensor.data);
        labels.push(output);
        if (Number(output) === Number(lab[i])) {
          correct++;
        }
      }
      let accuracy = correct / val.length;
      res.status(200).json({ name: name, output: labels, accuracy });
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  });

  return app;
};

const download = async url => {
  return new Promise((resolve, reject) => {
    const buffer = [];
    http.get(url, res => {
      res.on('data', data => buffer.push(data));

      res.on('end', () => resolve(Buffer.concat(buffer)));

      res.on('error', err => reject(err));
    });
  });
};

const createSession = async () => {
  const session = new InferenceSession();
  try {
    if (process.env.ONNX_MODEL_URI !== undefined) {
      const modelBuffer = await download(process.env.ONNX_MODEL_URI);
      await session.loadModel(modelBuffer);
    } else {
      await session.loadModel('./src/data/model.onnx');
    }
    let values = session.session._model.graph.getValues();
    let indices = session.session._model.graph.getInputIndices();
    let shape = values[indices[0]].type.shape.dims;
    MODEL_SHAPE = shape;
  } catch (err) {
    console.error(err);
  }
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
