import app from './server';

const port = process.env.PORT || 3000;

app.listen(port, () => {
  console.log(`The application is up at http://localhost:${port}`);
});
