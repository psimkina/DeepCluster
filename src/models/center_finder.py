import tensorflow as tf
from tensorflow import keras
from keras import layers


class GraphLayer(layers.Layer):
    """
    Graph layer that performs the aggregation of the neighbor messages and the update of the feature vectors.
    """

    def __init__(self):
        super(GraphLayer, self).__init__()

    def aggregate(self, adj_matrix, neighbor_messages, aggregation="all"):
        """
        Aggreagate the messages using the information from the adj_matrix.
        """
        if aggregation == "max":  # take only the closest neighbor
            # find the closest clusters based on distance
            max_index = tf.math.argmax(adj_matrix, axis=1)
            neighbour_messages = tf.gather(
                neighbour_messages, max_index, axis=1, batch_dims=1
            )

        if aggregation == "all":  # take all the neighbors
            # getting the neighbor indices
            neighbor_indices = tf.where(adj_matrix != -1)
            neighbor_indices = tf.reshape(
                neighbor_indices[:, -1],
                [-1, neighbor_messages.shape[1], neighbor_messages.shape[1] - 1],
            )

            neighbor_messages = tf.gather(
                neighbor_messages, neighbor_indices, axis=1, batch_dims=1
            )

        return neighbor_messages

    def update(self, neighbor_messages, weighted_messages, adj_coef):
        """
        Concatenate the initial features of the node and the aggregated messages.
        """
        neighbor_messages = tf.expand_dims(neighbor_messages, axis=2)
        updated_features = tf.concat([neighbor_messages, weighted_messages], axis=2)

        # only take the features from clusters that are closer than 3 crystals away
        updated_features = tf.expand_dims(adj_coef, axis=-1) * updated_features

        updated_features = tf.reshape(
            updated_features,
            [
                -1,
                updated_features.shape[1],
                updated_features.shape[2] * updated_features.shape[3],
            ],
        )

        return updated_features

    def call(self, adj_matrix, adj_coef, neighbour_messages):
        """
        Perform the aggregation of the neighbor messages and update of the feature vectors.
        """
        weighted_messages = self.aggregate(adj_matrix, neighbour_messages)
        updated_features = self.update(neighbour_messages, weighted_messages, adj_coef)

        return updated_features


class ConvBlock(keras.layers.Layer):
    """
    Convolutional bloch of the CenterFinder model.
    """
    def __init__(self, args):
        super(ConvBlock, self).__init__()

        conv_ker = args["conv_kernel"]
        conv_filt = args["conv_filter"]

        self.conv_1 = layers.Conv2D(
            conv_filt[0], conv_ker[0], activation=tf.keras.layers.LeakyReLU()
        )
        self.batch1 = layers.BatchNormalization()
        self.conv_2 = layers.Conv2D(
            conv_filt[1], conv_ker[1], activation=tf.keras.layers.LeakyReLU()
        )
        self.batch2 = layers.BatchNormalization()

        # flatten the output of convolutions
        self.flat = layers.Flatten()

    def call(self, x):
        x = self.conv_1(x)
        x = self.batch1(x)
        x = self.conv_2(x)
        x = self.batch2(x)
        x = self.flat(x)
        return x


class DenseBlock(keras.layers.Layer):
    """
    Dense block of the CenterFinder model.
    """
    def __init__(self, args):
        super(DenseBlock, self).__init__()

        dense_layers = args["dense_layers"]
        dropout = args["dropout"]

        self.dense_1 = layers.Dense(
            dense_layers[0], activation=tf.keras.layers.LeakyReLU()
        )
        self.drop1 = layers.Dropout(dropout[0])

        self.dense_2 = layers.Dense(
            dense_layers[1], activation=tf.keras.layers.LeakyReLU()
        )
        self.drop2 = layers.Dropout(dropout[1])

        self.dense_out = layers.Dense(2, activation="tanh")

        self.dense_en = layers.Dense(
            dense_layers[2], activation=tf.keras.layers.LeakyReLU()
        )
        self.drop_en = layers.Dropout(dropout[2])
        self.dense_en_out = layers.Dense(1, activation="sigmoid")

        self.seed_1 = layers.Dense(
            dense_layers[3], activation=tf.keras.layers.LeakyReLU()
        )
        self.drop_s = layers.Dropout(dropout[3])
        self.seed_out = layers.Dense(1, activation="sigmoid")

    def call(self, x):
        x = self.dense_1(x)
        x = self.drop1(x)

        x_c = self.dense_2(x)
        x_c = self.drop2(x_c)
        x_c = self.dense_out(x_c)

        x_en = self.dense_en(x)
        x_en = self.drop_en(x_en)
        x_en = self.dense_en_out(x_en)

        x_s = self.seed_1(x)
        x_s = self.drop_s(x_s)
        x_s = self.seed_out(x_s)

        return x_c, x_en, x_s


class CenterFinder(tf.keras.Model):
    """
    CenterFinder model that reconstructs position and energy of the particles from the input images.
    """
    def __init__(self, args):
        super().__init__()
        self.n = args["n"]

        # add all the network blocks
        self.conv_clusters = ConvBlock(args)
        self.dense_clusters = DenseBlock(args)
        self.graph = GraphLayer()

    def call(self, x):
        inputs, adj_matrix, adj_coef = x

        x_layers = []
        # divide input into separate images
        for i in range(self.n):
            x_layers.append(tf.expand_dims(inputs[:, :, :, i], axis=3))

        latent_features = []
        # pass through the convolution block
        for xt in x_layers:
            xt = self.conv_clusters(xt)
            latent_features.append(xt)

        latent_features = tf.convert_to_tensor(latent_features)
        latent_features = tf.transpose(latent_features, [1, 0, 2])

        # pass through the graph block
        updated_features = self.graph(adj_matrix, adj_coef, latent_features)
        updated_features = tf.transpose(updated_features, [1, 0, 2])

        xc_out = []
        xen_out = []
        xs_out = []

        # pass through the dense block
        for il in range(self.n):
            x_c, x_en, x_s = self.dense_clusters(updated_features[il])
            xc_out.append(x_c)
            xen_out.append(x_en)
            xs_out.append(x_s)

        xc_out = tf.convert_to_tensor(xc_out)
        xc_out = tf.transpose(xc_out, [1, 0, 2])

        xen_out = tf.convert_to_tensor(xen_out)
        xen_out = tf.transpose(xen_out, [1, 0, 2])

        xs_out = tf.convert_to_tensor(xs_out)
        xs_out = tf.transpose(xs_out, [1, 0, 2])

        return {
            "center": xc_out,
            "energy": xen_out,
            "seed": xs_out
        }
