import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


def implicit_als(sparse_data, alpha_val=40, iterations=10, lambda_val=0.1, features=10):

    """ Implementation of Alternating Least Squares with implicit data. We iteratively
        compute the user (x_u) and hotel (y_h) vectors using the following formulas:

        x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
        y_h = ((X.T*X + X.T*(Ch - I) * X) + lambda*I)^-1 * (Y.T * Ch * p(h))

        Args:
            sparse_data (csr_matrix): Our sparse user-by-hotel matrix

            alpha_val (int): The rate in which we'll increase our confidence
            in a preference with more interactions.

            iterations (int): How many times we alternate between fixing and
            updating our user and hotel vectors

            lambda_val (float): Regularization value

            features (int): How many latent features we want to compute.

        Returns:
            X (csr_matrix): user vectors of size users-by-features

            Y (csr_matrix): hotel vectors of size hotels-by-features
         """

    # Calculate the confidence for each value in our data
    confidence = sparse_data * alpha_val

    # Get the size of user rows and hotel columns
    user_size, hotel_size = sparse_data.shape

    # We create the user vectors X of size users-by-features, the hotel vectors
    # Y of size hotels-by-features and randomly assign the values.
    X = sparse.csr_matrix(np.random.normal(size=(user_size, features)))
    Y = sparse.csr_matrix(np.random.normal(size=(hotel_size, features)))

    # Precompute I and lambda * I
    X_I = sparse.eye(user_size)
    Y_I = sparse.eye(hotel_size)

    I = sparse.eye(features)
    lI = lambda_val * I

    # Start main loop. For each iteration we first compute X and then Y
    for i in range(iterations):
        print('iteration %d of %d' % (i + 1, iterations))

        # Precompute Y-transpose-Y and X-transpose-X
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Loop through all users
        for u in range(user_size):
            # Get the user row.
            u_row = confidence[u, :].toarray()

            # Calculate the binary preference p(u)
            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0

            # Calculate Cu and Cu - I
            CuI = sparse.diags(u_row, [0])
            Cu = CuI + Y_I

            # Put it all together and compute the final formula
            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

        for h in range(hotel_size):
            # Get the hotel column and transpose it.
            h_row = confidence[:, h].T.toarray()

            # Calculate the binary preference p(h)
            p_h = h_row.copy()
            p_h[p_h != 0] = 1.0

            # Calculate Ch and Ch - I
            ChI = sparse.diags(h_row, [0])
            Ch = ChI + X_I

            # Put it all together and compute the final formula
            xT_ChI_x = X.T.dot(ChI).dot(X)
            xT_Ch_ph = X.T.dot(Ch).dot(p_h.T)
            Y[h] = spsolve(xTx + xT_ChI_x + lI, xT_Ch_ph)

    return X, Y
