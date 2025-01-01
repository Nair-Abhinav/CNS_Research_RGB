# Image Encryption and Decryption Using Graph-Based Cryptography

This repository provides a Python-based implementation of an encryption and decryption system for RGB images using graph-based cryptography techniques. The program includes the generation of randomized graphs, key derivation, XOR encryption, and HMAC authentication for secure image processing.

## Features
- **Graph-Based Cryptography**: Randomized graphs and graph coloring are used for encrypting and decrypting images.
- **Encryption Visualization**: Generates a visualization of the encrypted image.
- **Authentication**: Ensures integrity and authenticity of encrypted data using HMAC.
- **Secure Key Derivation**: Derives encryption keys using PBKDF2 with a salt.
- **Support for RGB Images**: Handles multi-channel image encryption and decryption.

## Dependencies
- Python 3.8+
- Required libraries:
  - `hashlib`
  - `hmac`
  - `random`
  - `networkx`
  - `os`
  - `numpy`
  - `Pillow`

Install dependencies using:
```bash
pip install networkx numpy pillow
```

## How It Works
### Encryption
1. **Graph Generation**: Generates a graph with nodes representing byte values (0-255).
2. **Key Derivation**: Derives a secure encryption key using PBKDF2.
3. **Graph Randomization**: Randomizes the graph structure for added security.
4. **Pixel Transformation**: Each pixel is transformed based on derived parameters.
5. **XOR Encryption**: Applies XOR encryption to serialized data.
6. **Authentication**: Adds an HMAC to ensure data integrity.

### Decryption
1. **Verify Authentication**: Ensures that the data has not been tampered with.
2. **XOR Decryption**: Decrypts the serialized data using the derived key.
3. **Reverse Transformation**: Reconstructs the original image using graph coloring and pixel parameters.

## Usage
### Encrypt an Image
1. Place the RGB image to be encrypted in the project directory (e.g., `input_rgb_image.jpg`).
2. Run the program:
```bash
python main.py
```
3. The encrypted visualization will be saved as `encrypted_rgb.png` and the original image as `original_rgb.png`.

### Decrypt an Image
1. The encrypted data is stored in memory during the program execution.
2. The decryption process reconstructs the original image and saves it as `decrypted_rgb.png`.

### Example
The program encrypts and decrypts the image, verifying if the decrypted image matches the original. If successful, it prints:
```
Success: Decrypted image matches the original!
```

## Functions Overview
- **`generate_graph(seed: int) -> nx.Graph`**: Generates a graph with randomized edges.
- **`randomize_graph(graph: nx.Graph, seed: int) -> nx.Graph`**: Randomizes the nodes of the graph.
- **`derive_key(key: str, salt: bytes = None) -> (int, bytes)`**: Derives a cryptographic key using PBKDF2.
- **`compute_param(pixel: int, derived_key: int, salt: int) -> int`**: Computes a parameter for encryption.
- **`xor_encrypt(data: bytes, key: int) -> bytes`**: Encrypts data using XOR.
- **`add_authentication(data: bytes, auth_key: bytes) -> bytes`**: Adds HMAC authentication to the data.
- **`encrypt_image(image_path: str, key: str) -> tuple`**: Encrypts an RGB image.
- **`decrypt_image(encrypted_data: str, key: str, salt: bytes, shape: tuple) -> np.ndarray`**: Decrypts an RGB image.
- **`verify_authentication(data: bytes, auth_key: bytes) -> bytes`**: Verifies HMAC authentication.

## File Structure
```
.
├── main.py              # Main script
├── README.md            # Documentation (this file)
├── input_rgb_image.jpg  # Sample input image (provide your own)
├── original_rgb.png     # Saved original image (output)
├── encrypted_rgb.png    # Visualization of encrypted image (output)
├── decrypted_rgb.png    # Decrypted image (output)
```

## Notes
- Ensure that the input image is in RGB format.
- Salt and derived key are essential for decryption, so they must be preserved.
- The encryption process uses randomized parameters, so results may vary between executions with the same image and key.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- `networkx`: For graph processing.
- `numpy`: For efficient numerical operations.
- `Pillow`: For image handling and processing.

---
Feel free to contribute by submitting issues or pull requests!

