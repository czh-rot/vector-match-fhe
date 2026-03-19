# vector-match-fhe

This project implements encrypted vector matching based on the **Microsoft SEAL** library.

## Dependency

Before running this project, please install **Microsoft SEAL** locally.

## How to Run

This project is currently organized under the SEAL example framework.

To run it:

1. Install and build **Microsoft SEAL**.
2. Copy all files in this repository's `examples/` directory to the following directory in your local SEAL source tree:

   `SEAL/native/examples/`

3. Replace the original files if needed.
4. Rebuild SEAL and run the corresponding examples.
~~~

(base) czh@yhh-computer:~$ cd SEAL/
(base) czh@yhh-computer:~/SEAL$ cd native/examples/
(base) czh@yhh-computer:~/SEAL/native/examples$ ./build/bin/sealexamples
Microsoft SEAL version: 4.1.2
+---------------------------------------------------------+
| The following examples should be executed while reading |
| comments in associated files in native/examples/.       |
+---------------------------------------------------------+
| Examples                   | Source Files               |
+----------------------------+----------------------------+
| 0. Vector Match            | 0_bfv_vector.cpp           |
| 1. BFV Basics              | 1_bfv_basics.cpp           |
| 2. Encoders                | 2_encoders.cpp             |
| 3. Levels                  | 3_levels.cpp               |
| 4. BGV Basics              | 4_bgv_basics.cpp           |
| 5. CKKS Basics             | 5_ckks_basics.cpp          |
| 6. Rotation                | 6_rotation.cpp             |
| 7. Serialization           | 7_serialization.cpp        |
| 8. Performance Test        | 8_performance.cpp          |
| 9. Vector Match Parallel   | 9_bfv_vector.cpp           |
+----------------------------+----------------------------+
[      0 MB] Total allocation from the memory pool

> Run example (0 ~ 9) or exit (0): 0
[CKKS-BSGS-PACKED] poly_degree=4096, slot_count=2048, NUM_EMBEDDINGS=4096, DIM=1024, baby=32, giant=32, fd=1
[CKKS-BSGS-PACKED] Compute time: 21 ms
[CKKS-BSGS-PACKED] QueryBS: 3 ms
[CKKS-BSGS-PACKED] Multi: 18 ms
[CKKS-BSGS-PACKED] Top-100 overlap (HE vs Plain): 100 / 100
[CKKS-BSGS-PACKED] poly_degree=4096, slot_count=2048, NUM_EMBEDDINGS=4096, DIM=1024, baby=32, giant=32, fd=1
[CKKS-BSGS-PACKED] Compute time: 18 ms
[CKKS-BSGS-PACKED] QueryBS: 3 ms
[CKKS-BSGS-PACKED] Multi: 15 ms
[CKKS-BSGS-PACKED] Top-100 overlap (HE vs Plain): 100 / 100
[CKKS-BSGS-PACKED] poly_degree=4096, slot_count=2048, NUM_EMBEDDINGS=4096, DIM=1024, baby=32, giant=32, fd=1
[CKKS-BSGS-PACKED] Compute time: 17 ms
[CKKS-BSGS-PACKED] QueryBS: 2 ms
[CKKS-BSGS-PACKED] Multi: 15 ms
[CKKS-BSGS-PACKED] Top-100 overlap (HE vs Plain): 100 / 100
[CKKS-BSGS-PACKED] poly_degree=4096, slot_count=2048, NUM_EMBEDDINGS=4096, DIM=1024, baby=32, giant=32, fd=1
[CKKS-BSGS-PACKED] Compute time: 18 ms
[CKKS-BSGS-PACKED] QueryBS: 2 ms
[CKKS-BSGS-PACKED] Multi: 15 ms
[CKKS-BSGS-PACKED] Top-100 overlap (HE vs Plain): 100 / 100
[CKKS-BSGS-PACKED] poly_degree=4096, slot_count=2048, NUM_EMBEDDINGS=4096, DIM=1024, baby=32, giant=32, fd=1
[CKKS-BSGS-PACKED] Compute time: 17 ms
[CKKS-BSGS-PACKED] QueryBS: 2 ms
[CKKS-BSGS-PACKED] Multi: 14 ms
[CKKS-BSGS-PACKED] Top-100 overlap (HE vs Plain): 100 / 100
+---------------------------------------------------------+
| The following examples should be executed while reading |
| comments in associated files in native/examples/.       |
+---------------------------------------------------------+
| Examples                   | Source Files               |
+----------------------------+----------------------------+
| 0. Vector Match            | 0_bfv_vector.cpp           |
| 1. BFV Basics              | 1_bfv_basics.cpp           |
| 2. Encoders                | 2_encoders.cpp             |
| 3. Levels                  | 3_levels.cpp               |
| 4. BGV Basics              | 4_bgv_basics.cpp           |
| 5. CKKS Basics             | 5_ckks_basics.cpp          |
| 6. Rotation                | 6_rotation.cpp             |
| 7. Serialization           | 7_serialization.cpp        |
| 8. Performance Test        | 8_performance.cpp          |
| 9. Vector Match Parallel   | 9_bfv_vector.cpp           |
+----------------------------+----------------------------+
[    110 MB] Total allocation from the memory pool

> Run example (0 ~ 9) or exit (0): 1
[info][BGV] estimated t_bits = 26 (A=59, B=54, D=1024, K=1)
26 bits for plaintext modulus.
[BGV] Compute time: 92 ms
[BGV] Top-100 overlap (HE vs Plain): 98 / 100
[BGV] Top-10 decrypted ids: 312, 1907, 3148, 2161, 3871, 1394, 163, 674, 235, 1204
[info][BGV] estimated t_bits = 24 (A=30, B=27, D=1024, K=1)
24 bits for plaintext modulus.
[BGV] Compute time: 94 ms
[BGV] Top-100 overlap (HE vs Plain): 98 / 100
[BGV] Top-10 decrypted ids: 312, 1907, 3148, 2161, 3871, 1394, 163, 674, 235, 745
[info][BGV] estimated t_bits = 22 (A=15, B=14, D=1024, K=1)
22 bits for plaintext modulus.
[BGV] Compute time: 95 ms
[BGV] Top-100 overlap (HE vs Plain): 90 / 100
[BGV] Top-10 decrypted ids: 312, 1907, 3148, 2161, 3871, 1394, 674, 163, 235, 1204
[info][BGV] estimated t_bits = 20 (A=8, B=7, D=1024, K=1)
20 bits for plaintext modulus.
[BGV] Compute time: 93 ms
[BGV] Top-100 overlap (HE vs Plain): 87 / 100
[BGV] Top-10 decrypted ids: 312, 1907, 3148, 2161, 3871, 1394, 163, 745, 235, 1204
[info][BGV] estimated t_bits = 18 (A=4, B=4, D=1024, K=1)
20 bits for plaintext modulus.
[BGV] Compute time: 94 ms
[BGV] Top-100 overlap (HE vs Plain): 76 / 100
[BGV] Top-10 decrypted ids: 312, 1907, 2161, 3871, 3148, 3406, 2909, 163, 1394, 674
+---------------------------------------------------------+
| The following examples should be executed while reading |
| comments in associated files in native/examples/.       |
+---------------------------------------------------------+
| Examples                   | Source Files               |
+----------------------------+----------------------------+
| 0. Vector Match            | 0_bfv_vector.cpp           |
| 1. BFV Basics              | 1_bfv_basics.cpp           |
| 2. Encoders                | 2_encoders.cpp             |
| 3. Levels                  | 3_levels.cpp               |
| 4. BGV Basics              | 4_bgv_basics.cpp           |
| 5. CKKS Basics             | 5_ckks_basics.cpp          |
| 6. Rotation                | 6_rotation.cpp             |
| 7. Serialization           | 7_serialization.cpp        |
| 8. Performance Test        | 8_performance.cpp          |
| 9. Vector Match Parallel   | 9_bfv_vector.cpp           |
+----------------------------+----------------------------+
[    179 MB] Total allocation from the memory pool

> Run example (0 ~ 9) or exit (0): 2
[info] estimated t_bits = 26 (A=59, D=1024, K=1)
Compute time: 35 ms
Top-100 overlap (HE vs Plain): 99 / 100
[info] estimated t_bits = 24 (A=30, D=1024, K=1)
Compute time: 15 ms
Top-100 overlap (HE vs Plain): 98 / 100
[info] estimated t_bits = 22 (A=15, D=1024, K=1)
Compute time: 16 ms
Top-100 overlap (HE vs Plain): 96 / 100
[info] estimated t_bits = 20 (A=8, D=1024, K=1)
Compute time: 16 ms
Top-100 overlap (HE vs Plain): 90 / 100
[info] estimated t_bits = 18 (A=4, D=1024, K=1)
Compute time: 15 ms
Top-100 overlap (HE vs Plain): 77 / 100
~~~

## Notes

- This project is **not** currently a standalone CMake project.
- It is intended to be used together with the SEAL example infrastructure.
- Make sure your SEAL version is compatible with the code in this project.
