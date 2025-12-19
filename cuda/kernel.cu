extern "C" __global__ void compute_energy(float *out, float *errors, const size_t idx, const size_t n) {
  float tmp = 0.0f;

  for(int i = 0; i < n; i ++) {
    tmp += errors[i] * errors[i];
  }

  out[idx] = tmp;
}

extern "C" __global__ void compute_error(float *error, float *prediction, float *value, const size_t n) {
  for(int i = 0; i < n; i ++) {
    error[i] = value[i] - prediction[i];
  }
}

extern "C" __global__ void clear_prediction(float *prediction, const size_t n) {
  for(int i = 0; i < n; i ++) {
    prediction[i] = 0.0f;
  }
}

extern "C" __global__ void compute_prediction(float *prediction, float *weight, float *value, const size_t n) {
  for(int i = 0; i < n; i ++) {
    for(int j = 0; j < n; j ++) {
      prediction[i] += weight[i * n + j] * tanh(value[j]);
    }
  }
}

extern "C" __global__ void compute_value(float *value, float *error, float *value, float *weight, const size_t n) {
  // TODO
}
