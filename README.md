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

## Notes

- This project is **not** currently a standalone CMake project.
- It is intended to be used together with the SEAL example infrastructure.
- Make sure your SEAL version is compatible with the code in this project.