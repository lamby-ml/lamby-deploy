import createApp from './app';

const port = process.env.PORT || 3000;

createApp().then(app =>
  app.listen(port, () => {
    console.log(`The application is up at http://localhost:${port}`);
  })
);
