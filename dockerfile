FROM node:alpine

LABEL Name=lamby-deploy Version=0.0.1

EXPOSE 3000

WORKDIR /app
ADD . /app

RUN yarn install

ENV PORT=3000

CMD ["yarn", "start"]
