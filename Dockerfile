# Node v11 Docker Image
FROM node:11

# Working directory for application
WORKDIR /app

# Copy dependency information
COPY package.json ./
COPY yarn.lock ./

# Install production dependencies
RUN yarn install --production

# Copy app files
COPY . .

# Set env vars
ENV LAMBY_DEPLOY_PORT 80
ENV LAMBY_DEPLOY_MODEL_FILE ./model.onnx

# Expose port
EXPOSE 80

# Execute command on container start
CMD ["yarn", "start"]
