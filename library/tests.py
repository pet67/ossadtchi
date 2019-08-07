import numpy as np

from library.splitters import new_4x_cross_testing


def test_new_4x_cross_testing(X, Y):  
    for index, (X_train, Y_train, X_val, Y_val, X_test, Y_test) in enumerate(new_4x_cross_testing(X, Y)):
        if index == 0:
            concat_tuple_Y = (Y_train, Y_val, Y_test)
            concat_tuple_X = (X_train, X_val, X_test)
        elif index == 1:
            concat_tuple_Y = (Y_train, Y_test, Y_val)
            concat_tuple_X = (X_train, X_test, X_val)
        elif index == 2:
            concat_tuple_Y = (Y_test, Y_val, Y_train)
            concat_tuple_X = (X_test, X_val, X_train)
        elif index == 3:
            concat_tuple_Y = (Y_val, Y_test, Y_train)
            concat_tuple_X = (X_val, X_test, X_train)
        else:
            raise ValueError

        assert X_train.shape[0] == Y_train.shape[0]
        assert X_val.shape[0] == Y_val.shape[0]
        assert X_test.shape[0] == Y_test.shape[0]

        assert X_train.shape[1] > 1
        assert X_train.shape[0] > X_val.shape[0]
        assert X_train.shape[0] > X_test.shape[0]

        concated_Y = np.concatenate(concat_tuple_Y, axis=0)
        concated_X = np.concatenate(concat_tuple_X, axis=0)

        assert Y.shape == concated_Y.shape, str(index)
        assert X.shape == concated_X.shape, str(index)
        assert np.all(Y == concated_Y), str(index)
        assert np.all(X == concated_X), str(index)

        train_share = X_train.shape[0] * 1.0 / X.shape[0]
        val_share = X_val.shape[0] * 1.0 / X.shape[0];
        test_share = X_test.shape[0] * 1.0 / X.shape[0];

        SHARE_TOLERANCE = 0.001

        assert abs(train_share - 0.6) < SHARE_TOLERANCE, f"Expected train_share == 0.6, got {train_share}"
        assert abs(val_share - 0.2) < SHARE_TOLERANCE, f"Expected val_share == 0.6, got {val_share}"
        assert abs(test_share - 0.2) < SHARE_TOLERANCE, f"Expected test_share == 0.6, got {test_share}"

        print(f"{index} OK!")

