// train.js

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// 데이터 정규화를 위한 함수
function normalizeTensor(tensor, min, max) {
  return tensor.sub(min).div(max.sub(min));
}

// 판매 가능성 예측을 위한 데이터 생성
function generateSampleData(numSamples) {
  const inputData = [];
  const outputData = [];
  
  for (let i = 0; i < numSamples; i++) {
    // 온도 (15~30°C)
    const temperature = Math.random() * 15 + 15;
    // 습도 (40~80%)
    const humidity = Math.random() * 40 + 40;
    // 식물 넓이 (20~100cm^2)
    const plantArea = Math.random() * 80 + 20;
    
    // 판매 가능성 계산 (임의의 공식 사용)
    const saleProbability = Math.min(
      1,
      (temperature - 15) / 15 * 0.4 +
      (humidity - 40) / 40 * 0.3 +
      (plantArea - 20) / 80 * 0.3
    );

    inputData.push([temperature, humidity, plantArea]);
    outputData.push(saleProbability);
  }

  return { inputData, outputData };
}

// 성장일 예측을 위한 데이터 생성
function generateGrowthData(numSamples) {
  const inputData = [];
  const outputData = [];

  const minSaleArea = 100; // 최소 판매 기준 식물 넓이 (100cm^2)

  for (let i = 0; i < numSamples; i++) {
    const temperature = Math.random() * 15 + 15; // 15~30°C
    const humidity = Math.random() * 40 + 40;    // 40~80%
    const currentArea = Math.random() * 80 + 20; // 20~100cm^2

    // 성장률 계산 (단순화된 예시)
    const growthRate = (temperature - 15) / 15 * 0.5 + (humidity - 40) / 40 * 0.5;
    const dailyGrowth = growthRate * 2; // 일일 성장량 (cm^2/day)

    // 필요한 성장량
    const requiredGrowth = minSaleArea - currentArea;
    // 예상 성장일수
    const daysUntilSale = requiredGrowth > 0 ? requiredGrowth / dailyGrowth : 0;

    inputData.push([temperature, humidity, currentArea]);
    outputData.push(daysUntilSale);
  }

  return { inputData, outputData };
}

(async () => {
  // 데이터 생성 및 정규화

  // 판매 가능성 데이터
  const numSamples = 1000;
  const { inputData: saleInputData, outputData: saleOutputData } = generateSampleData(numSamples);
  const xs = tf.tensor2d(saleInputData);
  const ys = tf.tensor2d(saleOutputData, [numSamples, 1]);

  const saleInputMin = xs.min(0);
  const saleInputMax = xs.max(0);
  const xsNormalized = normalizeTensor(xs, saleInputMin, saleInputMax);

  // 성장일 데이터
  const numGrowthSamples = 1000;
  const { inputData: growthInputData, outputData: growthOutputData } = generateGrowthData(numGrowthSamples);
  const xg = tf.tensor2d(growthInputData);
  const yg = tf.tensor2d(growthOutputData, [numGrowthSamples, 1]);

  const growthInputMin = xg.min(0);
  const growthInputMax = xg.max(0);
  const growthOutputMin = yg.min(0);
  const growthOutputMax = yg.max(0);

  const xgNormalized = normalizeTensor(xg, growthInputMin, growthInputMax);
  const ygNormalized = normalizeTensor(yg, growthOutputMin, growthOutputMax);

  // 모델 정의 및 컴파일

  // 판매 가능성 예측 모델
  const saleModel = tf.sequential();
  saleModel.add(tf.layers.dense({ inputShape: [3], units: 16, activation: 'relu' }));
  saleModel.add(tf.layers.dense({ units: 8, activation: 'relu' }));
  saleModel.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  saleModel.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['mse'],
  });

  // 성장일 예측 모델
  const growthModel = tf.sequential();
  growthModel.add(tf.layers.dense({ inputShape: [3], units: 16, activation: 'relu' }));
  growthModel.add(tf.layers.dense({ units: 8, activation: 'relu' }));
  growthModel.add(tf.layers.dense({ units: 1 }));

  growthModel.compile({
    optimizer: tf.train.adam(),
    loss: 'meanSquaredError',
    metrics: ['mse'],
  });

  // 모델 학습

  console.log('판매 가능성 예측 모델 학습 시작...');
  await saleModel.fit(xsNormalized, ys, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
  });
  console.log('판매 가능성 예측 모델 학습 완료');

  console.log('성장일 예측 모델 학습 시작...');
  await growthModel.fit(xgNormalized, ygNormalized, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
  });
  console.log('성장일 예측 모델 학습 완료');

  // 모델 저장
  await saleModel.save('file://models/sale-model');
  await growthModel.save('file://models/growth-model');

  // 정규화 데이터 저장
  const normalizationData = {
    saleInputMin: saleInputMin.arraySync(),
    saleInputMax: saleInputMax.arraySync(),
    growthInputMin: growthInputMin.arraySync(),
    growthInputMax: growthInputMax.arraySync(),
    growthOutputMin: growthOutputMin.arraySync()[0], // 배열의 첫 번째 요소를 가져옵니다
    growthOutputMax: growthOutputMax.arraySync()[0], // 배열의 첫 번째 요소를 가져옵니다
  };

  fs.writeFileSync('models/normalizationData.json', JSON.stringify(normalizationData));

  console.log('모델 및 정규화 데이터 저장 완료');
})();
