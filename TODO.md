Things to do on the project:
- Use Rayon for non-tenserflow implementation
- Support different backends:
  1. CPU implementation (Rust)
  2. Tensorflow
  3. CUDA
- Different model types:
  1. Floating point based (high precision)
  2. 1.58 bit implementation
  3. fixed precision?
- Additional features:
  1. Let nodes use the _energy_ of another node as source value
  2. Instead of boolean masks for sensors allow for variable certainty/mutability
     during inference.
  3. Let certainty/mutability of sensors be decided by the network itself
