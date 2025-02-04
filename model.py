import numpy as np
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Input, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.cluster import KMeans
import pandas as pd

# Reads data from file, remove newlines characters in each line and return in a dataframe
def load_data(file_path):        
    with open(file_path, 'r') as file:
        header_data = file.readlines()
        header_data = [line.strip() for line in header_data]
    return pd.DataFrame({'Header' : header_data})

# Merge all packet header strings and return a list of fields made out of adjacent bytes
def get_byte_sequence(header_series):
    all_header_string = "".join(header_series.tolist())
    header_double_byte_list = [all_header_string[i:i+2] for i in range(0, len(all_header_string), 2)]
    return header_double_byte_list

# Append Zeroes to have header of same length, Mix adjacent hex bytes and compute decimal values
# Normalize by dividing with 255, Delete all-zero columns at the end
def preprocess_data_array(data_array):
    data_hex_string_list = [header for header in data_array.to_list()]
    # print(data_hex_string_list)
    max_packet_length = max(len(header) for header in data_hex_string_list)
    # print(max_packet_length)
    data_bytes_padded_array = np.array([list(packet.ljust(max_packet_length, '0')) for packet in data_hex_string_list])
    # print(data_bytes_padded_array[0])
    data_double_bytes_array = np.array([list(''.join(packet[i:i+2]) for i in range(0, len(packet), 2)) for packet in data_bytes_padded_array])
    # print(data_double_bytes_array[0])
    data_bytes_array = np.vectorize(lambda x: int(x, 16))(data_double_bytes_array)
    # print(data_bytes_array[0])
    data_bytes_normalized_array = data_bytes_array / 255.0
    # print(data_bytes_normalized_array[0])
    # print(data_bytes_normalized_array.shape)
    last_non_zero_col = np.max(np.where(data_bytes_normalized_array != 0)[1])
    return data_bytes_normalized_array[:, :last_non_zero_col + 1]

def get_autoencoder(input_dim, latent_dim):
    
    # Encoder
    input_layer = Input(shape=(input_dim, ))
    # Header length(Fields) : protocol-1 --> 32 , protocol-3 --> 360
    if input_dim > 256:
        layer_sizes = [256, 64]
    elif input_dim in range(16, 64):
        layer_sizes = [16]
    
    # Small model with only one layer each for encoding and decoding (protocol-2 header is only 12 byte long)
    if input_dim < 16:
         encoded = Dense(latent_dim, activation='relu')(input_layer)
         decoded = Dense(input_dim, activation='sigmoid')(encoded)
    else: 
         # Encoder
         encoded = Dense(layer_sizes[0], activation='relu')(input_layer)
         encoded = BatchNormalization()(encoded)
         for size in layer_sizes[1:]:
             encoded = Dense(size, activation='relu')(encoded)
             encoded = BatchNormalization()(encoded)
         encoded = Dense(latent_dim, activation='relu')(encoded)
         # Decoder
         decoded = Dense(layer_sizes[-1], activation = 'relu')(encoded)
         decoded = BatchNormalization()(decoded)
         for size in layer_sizes[-2::-1]:
             decoded = Dense(size, activation='relu')(decoded)
             decoded = BatchNormalization()(decoded)
         decoded = Dense(input_dim, activation='sigmoid')(decoded)
    # Model
    autoencoder = Model(inputs = input_layer, outputs = decoded)
    # Compile
    autoencoder.compile(optimizer = Adam(learning_rate=0.001), loss = 'binary_crossentropy')

    # Define Encoder
    encoder = Model(inputs = input_layer, outputs = encoded)

    # Define Decoder - Reconstruct decoder by applying layers present in the right half
    encoded_input = Input(shape=(latent_dim,))
    if input_dim < 16:
        decoded_output = autoencoder.layers[-1](encoded_input)
    else:
        layers = 2 * len(layer_sizes) + 1
        decoded_output = autoencoder.layers[-1 * layers](encoded_input)
        layers = layers - 1
        while(layers):
            decoded_output = autoencoder.layers[-1 * layers](decoded_output)
            layers = layers - 1

    decoder = Model(inputs=encoded_input, outputs=decoded_output)
    return autoencoder, encoder, decoder

def inference(path_to_txt):
    # ======================================================================= #
    test_data = load_data(path_to_txt)
    X_test = preprocess_data_array(test_data["Header"])

    # Autoencoder to reduce dimensionality for non-linear packet header data
    # Get Autoencoder
    input_dim = X_test.shape[1]
    # Latent vector size and model architecture set after experimenting with multiple values
    latent_dim = 32
    autoencoder, encoder, decoder = get_autoencoder(input_dim, latent_dim)
    autoencoder.load_weights('protocol3_weights.h5')
    encoder.load_weights('protocol3_encoder_weights.h5')
    decoder.load_weights('protocol3_decoder_weights.h5')
    latent_vectors = encoder.predict(X_test)
    print("\n\n\nAutoencoder summary : \n")
    print(autoencoder.summary(), end="\n")

    # Hyperparameter k obtained using Elbow graph 
    kmeans = KMeans(n_clusters = 16, random_state = 0)
    kmeans.fit(latent_vectors)
    predicted_clusters = kmeans.labels_
    predicted_centroids = kmeans.cluster_centers_
    predicted_clusters = predicted_clusters[:, np.newaxis]
    data_with_cluster_labels = np.hstack((predicted_clusters, X_test * 255)) 


    # Intra-cluster variance per byte
    num_fields = X_test.shape[1]
    num_clusters = len(np.unique(predicted_clusters))
    intra_cluster_var = np.zeros((num_clusters, num_fields))

    for cluster in range(num_clusters):
        cluster_data = data_with_cluster_labels[data_with_cluster_labels[:, 0] == cluster][:, 1:]
        intra_cluster_var[cluster] = np.var(cluster_data, axis = 0)
 
    # Constant bytes with 0 variance are ignored by setting their values as too high
    intra_cluster_var[intra_cluster_var == 0] = 10 ** 10
    # Get indices of bytes with bottom 10 variances
    result_array = np.sort(np.argsort(intra_cluster_var, axis = 1)[:, :10], axis = 1)
    print("\n Intra Cluster Variance (10 Least) : \n")
    print(result_array)
    # Getting most occured byte index in first column of least variance 
    unique_values, counts = np.unique(result_array[:, :1], return_counts=True)
    msg_type_byte_index = unique_values[np.argsort(counts)[-1]]
    print("\n Estimated Message Type Byte Index : " , msg_type_byte_index)
    
    # predict centroid in actual input space using decoder 
    centroid_data = decoder.predict(predicted_centroids, verbose = 0)

    # Maximum Inter-cluster variance per byte
    inter_cluster_var = np.var(centroid_data, axis = 0)
    print("\n Inter Cluser Variance (Byte indices in acsending order): \n")
    print(np.argsort(inter_cluster_var))


    with open(path_to_txt, 'r') as file:
        num_packets = sum(1 for _ in file)

    predicted_idx_list = [msg_type_byte_index] * num_packets
    # ======================================================================= #

    # Build the path for the output file
    output_path = path_to_txt.replace(".txt", "-pred.txt")

    # Write predicted indices to the file using a concise approach
    with open(output_path, "w") as outfile:
        outfile.write("\n".join(map(str, predicted_idx_list)))