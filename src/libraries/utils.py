from sklearn.preprocessing import StandardScaler

def channel_normalize(X_train, X_test, channel_axis=1):
    # Normalize on based on means and stds of ALL DATA along channel axis! So means have shape (n_channels,)
    scaler = StandardScaler()

    # Flatten data along channel axis
    flat_train = X_train.reshape(-1, X_train.shape[channel_axis])
    flat_test = X_test.reshape(-1, X_test.shape[channel_axis])

    scaler.fit(flat_train)
    X_train = scaler.transform(flat_train).reshape(X_train.shape)
    X_test = scaler.transform(flat_test).reshape(X_test.shape)

    return X_train, X_test