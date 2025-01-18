const fs = require('fs');
const csv = require('csv-parser');
const tf = require('@tensorflow/tfjs-node');
const data = [];
fs.createReadStream('Cyber-Bullying/twitter_sexism_parsed_dataset.csv')
  .pipe(csv())
  .on('data', (row) => {
    data.push(row);
  })
  .on('end', () => {
    console.log('CSV file successfully processed');
    processData(data);
  });
function processData(data) {
  const labelEncoder = { sexism: 1, 'not sexism': 0 };
  const vectorizeText = (text) => {
    const words = text.toLowerCase().split(/\W+/);
    const wordCount = {};
    words.forEach((word) => {
      if (word.length > 0) {
        wordCount[word] = (wordCount[word] || 0) + 1;
      }
    });
    return wordCount;
  };
  const texts = data.map((d) => vectorizeText(d.Text)); 
  const labels = data.map((d) => labelEncoder[d.Annotation]); 
  const vocabulary = {};
  texts.forEach((text) => {
    Object.keys(text).forEach((word) => {
      vocabulary[word] = (vocabulary[word] || 0) + 1;
    });
  });
  const vocabArray = Object.keys(vocabulary);
  const wordIndex = vocabArray.reduce((acc, word, idx) => {
    acc[word] = idx;
    return acc;
  }, {});
  const vectorizedTexts = texts.map((text) => {
    const vector = new Array(vocabArray.length).fill(0);
    Object.keys(text).forEach((word) => {
      const index = wordIndex[word];
      vector[index] = text[word];
    });
    return vector;
  });
  const trainSize = Math.floor(0.8 * vectorizedTexts.length);
  const X_train = vectorizedTexts.slice(0, trainSize);
  const X_test = vectorizedTexts.slice(trainSize);
  const y_train = labels.slice(0, trainSize);
  const y_test = labels.slice(trainSize);
  const X_trainTensor = tf.tensor2d(X_train);
  const X_testTensor = tf.tensor2d(X_test);
  const y_trainTensor = tf.tensor1d(y_train, 'int32');
  const y_testTensor = tf.tensor1d(y_test, 'int32');
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [X_train[0].length],
    activation: 'sigmoid',
  }));
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  model.fit(X_trainTensor, y_trainTensor, {
    epochs: 10,
    batchSize: 32,
    validationSplit: 0.2,
    verbose: 1,
  }).then(() => {
    model.evaluate(X_testTensor, y_testTensor).then((result) => {
      console.log('Test Loss:', result[0]);
      console.log('Test Accuracy:', result[1]);
      model.predict(X_testTensor).array().then((predictions) => {
        const y_pred = predictions.map((pred) => (pred[0] > 0.5 ? 1 : 0));
        const cm = computeConfusionMatrix(y_test, y_pred);
        console.log('Confusion Matrix:');
        console.log(cm);
      });
    });
  });
}
function computeConfusionMatrix(y_true, y_pred) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < y_true.length; i++) {
    if (y_true[i] === 1 && y_pred[i] === 1) tp++; 
    if (y_true[i] === 0 && y_pred[i] === 0) tn++; 
    if (y_true[i] === 0 && y_pred[i] === 1) fp++; 
    if (y_true[i] === 1 && y_pred[i] === 0) fn++; 
  }
  return [[tp, fp], [fn, tn]];

}