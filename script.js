// script.js

// 모든 코드를 즉시 실행 함수로 감쌉니다
(async function() {
  // 모델 및 정규화 데이터 로드
  let saleModel, growthModel;
  let saleInputMin, saleInputMax;
  let growthInputMin, growthInputMax, growthOutputMin, growthOutputMax;

  // 정규화된 데이터를 Tensor로 변환하는 함수
  function arrayToTensor(array) {
    return tf.tensor(array);
  }

  // 데이터 정규화를 위한 함수
  function normalizeTensor(tensor, min, max) {
    return tensor.sub(min).div(max.sub(min));
  }

  // 데이터 역정규화를 위한 함수
  function denormalizeTensor(tensor, min, max) {
    return tensor.mul(max.sub(min)).add(min);
  }

  // 모델 및 정규화 데이터 로드
  async function loadModelsAndData() {
    try {
      // 모델 로드
      console.log('모델 로드');
      saleModel = await tf.loadLayersModel('models/sale-model/model.json');
      growthModel = await tf.loadLayersModel('models/growth-model/model.json');
      console.log('모델 로드 완료');
      console.log(saleModel, growthModel);
      
      // 정규화 데이터 로드
      const response = await fetch('models/normalizationData.json');
      const normalizationData = await response.json();

      saleInputMin = arrayToTensor(normalizationData.saleInputMin);
      saleInputMax = arrayToTensor(normalizationData.saleInputMax);

      growthInputMin = arrayToTensor(normalizationData.growthInputMin);
      growthInputMax = arrayToTensor(normalizationData.growthInputMax);
      growthOutputMin = tf.scalar(normalizationData.growthOutputMin);
      growthOutputMax = tf.scalar(normalizationData.growthOutputMax);

      // 모델이 로드되면 예측 버튼 활성화
      document.getElementById('predictButton').disabled = false;
      console.log('모델 및 정규화 데이터 로드 완료');
    } catch (error) {
      console.error('모델 또는 정규화 데이터 로드 중 오류 발생:', error);
    }
  }

  // 예측 함수
  function predict() {
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const plantArea = parseFloat(document.getElementById('plantArea').value);

    if (isNaN(temperature) || isNaN(humidity) || isNaN(plantArea)) {
      alert('모든 입력 값을 올바르게 입력해주세요.');
      return;
    }

    // 판매 가능성 예측
    const saleInput = tf.tensor2d([[temperature, humidity, plantArea]]);
    const saleInputNormalized = normalizeTensor(saleInput, saleInputMin, saleInputMax);
    const salePrediction = saleModel.predict(saleInputNormalized);

    salePrediction.array().then((array) => {
      const saleProbability = array[0][0];
      const saleResultElement = document.getElementById('saleResult');
      saleResultElement.innerText = `판매 가능성: ${(saleProbability * 100).toFixed(2)}%`;
    });

    // 성장일 예측
    const growthInput = tf.tensor2d([[temperature, humidity, plantArea]]);
    const growthInputNormalized = normalizeTensor(growthInput, growthInputMin, growthInputMax);
    const growthPredictionNormalized = growthModel.predict(growthInputNormalized);

    // 성장일 역정규화
    const growthPrediction = denormalizeTensor(growthPredictionNormalized, growthOutputMin, growthOutputMax);

    growthPrediction.array().then((array) => {
      const predictedDays = array[0][0];
      const growthResultElement = document.getElementById('growthResult');
      if (predictedDays <= 0) {
        growthResultElement.innerText = `이미 판매 기준을 충족했습니다.`;
      } else {
        growthResultElement.innerText = `예상 성장일: ${predictedDays.toFixed(2)}일`;
      }
    });
  }

  // 모델 로드 시작
  await loadModelsAndData();

  // 예측 버튼 클릭 이벤트 추가
  document.getElementById('predictButton').addEventListener('click', predict);

})();
