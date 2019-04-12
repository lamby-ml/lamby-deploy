# lamby-deploy
Containerized application for loading and serving lamby models.

## Installing dependencies

```sh
# Install all dependencies
yarn install --dev
```

## Running the application

```sh
# Run once
yarn start

# Run and watch for changes
yarn watch
```

## Docker

```sh
# Build the image
docker build -t lambyml/lamby-deploy:latest .

# Run the image in a container
docker run --name lamby-deploy -p $PORT:3000 -d \
  -e ONNX_MODEL_URI=$ONNX_MODEL_URI \
  lambyml/lamby-deploy:latest
```
