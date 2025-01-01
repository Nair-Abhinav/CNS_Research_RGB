# import hashlib
# import hmac
# import random
# import networkx as nx
# import os
# import numpy as np
# from PIL import Image

# def generate_graph(seed: int) -> nx.Graph:
#     random.seed(seed)
#     graph = nx.Graph()
#     ascii_range = range(256)  # Full byte range for image pixels
#     graph.add_nodes_from(ascii_range)
#     for i in ascii_range:
#         for j in ascii_range:
#             if random.random() < 0.05:  # Random edge creation
#                 graph.add_edge(i, j)
#     return graph

# def randomize_graph(graph: nx.Graph, seed: int) -> nx.Graph:
#     random.seed(seed)
#     nodes = list(graph.nodes())
#     random.shuffle(nodes)
#     randomized_graph = nx.Graph()
#     randomized_graph.add_nodes_from(nodes)
#     randomized_graph.add_edges_from((nodes[i], nodes[j]) for i, j in graph.edges())
#     return randomized_graph

# def derive_key(key: str, salt: bytes = None) -> (int, bytes):
#     if salt is None:
#         salt = os.urandom(16)
#     derived_key = hashlib.pbkdf2_hmac('sha256', key.encode(), salt, 100000)
#     return int.from_bytes(derived_key, byteorder='big'), salt

# def compute_param(pixel: int, derived_key: int, salt: int) -> int:
#     return (pixel ^ derived_key ^ salt) % 256

# def xor_encrypt(data: bytes, key: int) -> bytes:
#     key_bytes = key.to_bytes((key.bit_length() + 7) // 8, byteorder='big')
#     return bytes(d ^ key_bytes[i % len(key_bytes)] for i, d in enumerate(data))

# def add_authentication(data: bytes, auth_key: bytes) -> bytes:
#     hmac_digest = hmac.new(auth_key, data, hashlib.sha256).digest()
#     return data + hmac_digest

# def encrypt_image(image_path: str, key: str) -> tuple:
#     # Open the RGB image directly
#     original_image = Image.open(image_path)
#     image_array = np.array(original_image)
    
#     # Create a visualization of encrypted image (random noise)
#     encrypted_visualization = np.random.randint(0, 256, image_array.shape, dtype=np.uint8)
    
#     # Derive key and generate graph
#     derived_key, salt = derive_key(key)
#     graph = generate_graph(derived_key)
#     randomized_graph = randomize_graph(graph, derived_key)
#     coloring = nx.coloring.greedy_color(randomized_graph, strategy="random_sequential")
    
#     # Prepare encrypted image data
#     encrypted_data = []
    
#     # Process each channel
#     for channel in range(image_array.shape[2]):  # Process each RGB channel
#         channel_data = []
#         for i in range(image_array.shape[0]):
#             row_data = []
#             for j in range(image_array.shape[1]):
#                 pixel = image_array[i, j, channel]
#                 color = coloring[pixel]
#                 salt_value = random.randint(0, 255)
#                 param = compute_param(pixel, derived_key, salt_value)
#                 row_data.append(f"{color}-{param}-{salt_value}")
#             channel_data.extend(row_data)
#         encrypted_data.extend(channel_data)
    
#     # Join all encrypted data
#     cipher_str = '|'.join(encrypted_data)
    
#     # XOR encrypt the serialized string
#     encrypted_cipher = xor_encrypt(cipher_str.encode('utf-8'), derived_key)
    
#     # Add HMAC for authentication
#     auth_key = hashlib.sha256(f"{derived_key}".encode()).digest()
#     final_cipher = add_authentication(encrypted_cipher, auth_key)
    
#     return final_cipher.hex(), salt, encrypted_visualization, image_array

# def decrypt_image(encrypted_data: str, key: str, salt: bytes, shape: tuple) -> np.ndarray:
#     # Derive key
#     derived_key, _ = derive_key(key, salt)
    
#     # Prepare authentication
#     auth_key = hashlib.sha256(f"{derived_key}".encode()).digest()
    
#     # Verify and extract the ciphertext
#     cipher_bytes = bytes.fromhex(encrypted_data)
#     decrypted_data = verify_authentication(cipher_bytes, auth_key)
    
#     # XOR Decrypt
#     decrypted_str = xor_encrypt(decrypted_data, derived_key).decode('utf-8', errors='ignore')
    
#     # Parse the decrypted data
#     pairs = [tuple(map(int, pair.split('-'))) for pair in decrypted_str.split('|')]
    
#     # Regenerate graph and coloring
#     graph = generate_graph(derived_key)
#     randomized_graph = randomize_graph(graph, derived_key)
#     coloring = nx.coloring.greedy_color(randomized_graph, strategy="random_sequential")
    
#     # Initialize the decrypted image array
#     decrypted_array = np.zeros(shape, dtype=np.uint8)
#     pixels_per_channel = shape[0] * shape[1]
    
#     # Decrypt each channel
#     for channel in range(shape[2]):
#         channel_start = channel * pixels_per_channel
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 idx = channel_start + i * shape[1] + j
#                 color, param, salt_value = pairs[idx]
                
#                 # Find original pixel value
#                 for pixel, pixel_color in coloring.items():
#                     if pixel_color == color and compute_param(pixel, derived_key, salt_value) == param:
#                         decrypted_array[i, j, channel] = pixel
#                         break
    
#     return decrypted_array

# def verify_authentication(data: bytes, auth_key: bytes) -> bytes:
#     received_data, received_hmac = data[:-32], data[-32:]
#     expected_hmac = hmac.new(auth_key, received_data, hashlib.sha256).digest()
#     if not hmac.compare_digest(received_hmac, expected_hmac):
#         raise ValueError("Authentication failed.")
#     return received_data

# def main():
#     # Image path (provide your RGB image path here)
#     image_path = 'input_rgb_image.jpg'
#     key = "Cryptography"
    
#     try:
#         # Encrypt the image
#         print("Starting encryption...")
#         encrypted_data, salt, encrypted_visual, original_array = encrypt_image(image_path, key)
        
#         # Save original image
#         original_img = Image.fromarray(original_array)
#         original_img.save('original_rgb.png')
#         print("Original image saved as 'original_rgb.png'")
        
#         # Save encrypted visualization
#         encrypted_img = Image.fromarray(encrypted_visual)
#         encrypted_img.save('encrypted_rgb.png')
#         print("Encrypted image visualization saved as 'encrypted_rgb.png'")
        
#         # Decrypt the image
#         print("Starting decryption...")
#         decrypted_array = decrypt_image(encrypted_data, key, salt, original_array.shape)
        
#         # Save decrypted image
#         decrypted_img = Image.fromarray(decrypted_array)
#         decrypted_img.save('decrypted_rgb.png')
#         print("Decrypted image saved as 'decrypted_rgb.png'")
        
#         # Verify if decryption was successful
#         if np.array_equal(original_array, decrypted_array):
#             print("Success: Decrypted image matches the original!")
#         else:
#             print("Warning: Decrypted image differs from original.")
            
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()


# cns rgb image code 2nd trial
import hashlib
import hmac
import random
import networkx as nx
import os
import numpy as np
from PIL import Image
from collections import defaultdict

def generate_graph(seed: int) -> nx.Graph:
    random.seed(seed)
    graph = nx.Graph()
    ascii_range = range(256)
    graph.add_nodes_from(ascii_range)
    # Reduced edge density for faster processing
    for i in ascii_range:
        for _ in range(3):  # Limit connections per node
            j = random.randint(0, 255)
            if i != j:
                graph.add_edge(i, j)
    return graph

def randomize_graph(graph: nx.Graph, seed: int) -> nx.Graph:
    random.seed(seed)
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    mapping = dict(zip(graph.nodes(), nodes))
    return nx.relabel_nodes(graph, mapping)

def derive_key(key: str, salt: bytes = None) -> (int, bytes):
    if salt is None:
        salt = os.urandom(16)
    derived_key = hashlib.pbkdf2_hmac('sha256', key.encode(), salt, 50000)  # Reduced iterations
    return int.from_bytes(derived_key, byteorder='big'), salt

def compute_param(pixel: int, derived_key: int, salt: int) -> int:
    return (pixel ^ derived_key ^ salt) % 256

def xor_encrypt(data: bytes, key: int) -> bytes:
    key_bytes = key.to_bytes((key.bit_length() + 7) // 8, byteorder='big')
    key_length = len(key_bytes)
    return bytes(data[i] ^ key_bytes[i % key_length] for i in range(len(data)))

def add_authentication(data: bytes, auth_key: bytes) -> bytes:
    hmac_digest = hmac.new(auth_key, data, hashlib.sha256).digest()
    return data + hmac_digest

def encrypt_image(image_path: str, key: str) -> tuple:
    # Open the RGB image directly
    original_image = Image.open(image_path)
    image_array = np.array(original_image)
    
    # Create a visualization of encrypted image
    encrypted_visualization = np.random.randint(0, 256, image_array.shape, dtype=np.uint8)
    
    # Derive key and generate graph
    derived_key, salt = derive_key(key)
    graph = generate_graph(derived_key)
    randomized_graph = randomize_graph(graph, derived_key)
    coloring = nx.coloring.greedy_color(randomized_graph, strategy="largest_first")  # Changed strategy
    
    # Pre-compute salt values for better performance
    salt_values = np.random.randint(0, 256, image_array.shape, dtype=np.uint8)
    
    # Process all channels simultaneously using numpy operations
    height, width, channels = image_array.shape
    encrypted_data = []
    
    # Flatten and process all pixels at once
    for channel in range(channels):
        channel_pixels = image_array[:, :, channel].flatten()
        channel_salts = salt_values[:, :, channel].flatten()
        
        # Vectorized operations for better performance
        colors = np.array([coloring[p] for p in channel_pixels])
        params = np.array([compute_param(p, derived_key, s) 
                          for p, s in zip(channel_pixels, channel_salts)])
        
        # Create encrypted strings efficiently
        channel_data = [f"{c}-{p}-{s}" for c, p, s in zip(colors, params, channel_salts)]
        encrypted_data.extend(channel_data)
    
    # Join encrypted data efficiently
    cipher_str = '|'.join(encrypted_data)
    
    # XOR encrypt and add authentication
    encrypted_cipher = xor_encrypt(cipher_str.encode('utf-8'), derived_key)
    auth_key = hashlib.sha256(str(derived_key).encode()).digest()
    final_cipher = add_authentication(encrypted_cipher, auth_key)
    
    return final_cipher.hex(), salt, encrypted_visualization, image_array

def decrypt_image(encrypted_data: str, key: str, salt: bytes, shape: tuple) -> np.ndarray:
    # Derive key
    derived_key, _ = derive_key(key, salt)
    
    # Prepare authentication
    auth_key = hashlib.sha256(str(derived_key).encode()).digest()
    
    # Verify and decrypt
    cipher_bytes = bytes.fromhex(encrypted_data)
    decrypted_data = verify_authentication(cipher_bytes, auth_key)
    decrypted_str = xor_encrypt(decrypted_data, derived_key).decode('utf-8', errors='ignore')
    
    # Parse the decrypted data efficiently
    pairs = [tuple(map(int, pair.split('-'))) for pair in decrypted_str.split('|')]
    
    # Generate graph and coloring
    graph = generate_graph(derived_key)
    randomized_graph = randomize_graph(graph, derived_key)
    coloring = nx.coloring.greedy_color(randomized_graph, strategy="largest_first")
    
    # Create reverse mapping for faster lookup
    reverse_coloring = defaultdict(list)
    for pixel, color in coloring.items():
        reverse_coloring[color].append(pixel)
    
    # Initialize the decrypted image array
    decrypted_array = np.zeros(shape, dtype=np.uint8)
    pixels_per_channel = shape[0] * shape[1]
    
    # Optimized channel decryption
    for channel in range(shape[2]):
        channel_start = channel * pixels_per_channel
        channel_data = pairs[channel_start:channel_start + pixels_per_channel]
        
        # Process each pixel in the channel
        for idx, (color, param, salt_value) in enumerate(channel_data):
            i, j = divmod(idx, shape[1])
            
            # Use reverse mapping for faster lookup
            for potential_pixel in reverse_coloring[color]:
                if compute_param(potential_pixel, derived_key, salt_value) == param:
                    decrypted_array[i, j, channel] = potential_pixel
                    break
    
    return decrypted_array

def verify_authentication(data: bytes, auth_key: bytes) -> bytes:
    received_data, received_hmac = data[:-32], data[-32:]
    expected_hmac = hmac.new(auth_key, received_data, hashlib.sha256).digest()
    if not hmac.compare_digest(received_hmac, expected_hmac):
        raise ValueError("Authentication failed.")
    return received_data

def main():
    image_path = 'input_rgb_image.jpg'
    key = "Cryptography"
    
    try:
        print("Starting encryption...")
        encrypted_data, salt, encrypted_visual, original_array = encrypt_image(image_path, key)
        
        original_img = Image.fromarray(original_array)
        original_img.save('original_rgb.png')
        print("Original image saved")
        
        encrypted_img = Image.fromarray(encrypted_visual)
        encrypted_img.save('encrypted_rgb.png')
        print("Encrypted image saved")
        
        print("Starting decryption...")
        decrypted_array = decrypt_image(encrypted_data, key, salt, original_array.shape)
        
        decrypted_img = Image.fromarray(decrypted_array)
        decrypted_img.save('decrypted_rgb.png')
        print("Decryption completed")
        
        if np.array_equal(original_array, decrypted_array):
            print("Success: Perfect reconstruction achieved!")
        else:
            print("Warning: Decrypted image differs from original.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()