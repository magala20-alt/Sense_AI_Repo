# Copyright 2021-2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Inner operators for reinforcement learning."""

from __future__ import absolute_import
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import prim_attr_register, PrimitiveWithInfer, Primitive


class GRUV2(PrimitiveWithInfer):
    """
    Performs the Stacked GRU (Gated Recurrent Unit) on the input.

    For detailed information, please refer to :class:`mindspore.nn.GRU`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked GRU.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`.
        bidirectional (bool): Specifies whether it is a bidirectional GRU.
        dropout (float): If not 0, append `Dropout` layer on the outputs of each
            GRU layer except the last layer. The range of dropout is [0.0, 1.0].
        is_train (bool): Specifies whether it is training mode or inference mode.

    Inputs:
        - **input** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`).
        - **h** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **w** (Tensor) - The input tensor which states for weights.
        - **seq_lengths** (Tensor) - The Tensor of shape (batch_size, ), indicates the seq_length of each batch dim.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`, `reserve`, `state`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **h_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **reserve** (Tensor) - Tensor of shape (r, 1).
        - **state** (Tensor) - Random number generator state and its shape is (s, 1).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias` or `bidirectional` is not a bool.
        TypeError: If `dropout` is not a float.
        ValueError: If `dropout` is not in range [0.0, 1.0].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_size = 10
        >>> hidden_size = 2
        >>> num_layers = 1
        >>> max_seq_len = 5
        >>> batch_size = 2
        >>>
        >>> import mindspore.ops.operations._rl_inner_ops as rl_ops
        >>> net = rl_ops.GRUV2(input_size, hidden_size, num_layers, True, False, 0.0)
        >>> input_tensor = Tensor(np.ones([max_seq_len, batch_size, input_size]).astype(np.float32))
        >>> h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        >>> w = Tensor(np.ones([84, 1, 1]).astype(np.float32))
        >>> seq_lengths = Tensor(np.array([4, 3]))
        >>> output, hn,  _, _ = net(input_tensor, h0, w, seq_lengths)
        >>> print(output)
        [[[1.  1. ]
          [1.  1. ]]
         [[1.  1. ]
          [1.  1. ]]
         [[1.  1.]
          [1.  1.]]
         [[1.  1. ]
          [1.  1. ]]
         [[1.  1. ]
          [1.  1. ]]]
    """

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout, is_train=True):
        """Initialize GRU."""
        self.input_size = validator.check_positive_int(input_size, "input_size", self.name)
        self.hidden_size = validator.check_positive_int(hidden_size, "hidden_size", self.name)
        self.num_layers = validator.check_positive_int(num_layers, "num_layers", self.name)
        self.has_bias = validator.check_value_type("has_bias", has_bias, (bool,), self.name)
        self.bidirectional = validator.check_value_type("bidirectional", bidirectional, (bool,), self.name)
        self.dropout = validator.check_value_type("dropout", dropout, [float], self.name)
        self.dropout = validator.check_float_range(dropout, 0, 1, validator.INC_BOTH, 'dropout', self.name)
        self.is_train = validator.check_value_type("is_train", is_train, (bool,), self.name)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def infer_shape(self, x_shape, h_shape, w_shape, seq_lengths_shape):
        validator.check_equal_int(len(x_shape), 3, "x rank", self.name)
        validator.check_equal_int(x_shape[2], self.input_size, "x[2]", self.name)

        validator.check_equal_int(len(h_shape), 3, "h rank", self.name)
        validator.check_int(h_shape[0], self.num_layers * self.num_directions, validator.EQ, "h[0]", self.name)
        validator.check_equal_int(h_shape[1], x_shape[1], "h[1]", self.name)
        validator.check_int(h_shape[2], self.hidden_size, validator.EQ, "h[2]", self.name)

        validator.check_equal_int(len(seq_lengths_shape), 1, "seq_lengths rank", self.name)
        validator.check_equal_int(seq_lengths_shape[0], x_shape[1], "seq_lengths_shape[0]", self.name)

        y_shape = (x_shape[0], x_shape[1], self.hidden_size * self.num_directions)

        # set arbitrary shape for reserved space
        reserved_shape = (1, 1)
        state_shape = (1, 1)
        return y_shape, h_shape, reserved_shape, state_shape

    def infer_dtype(self, x_dtype, h_dtype, w_dtype, seq_lengths_dtype):
        args = {'x': x_dtype, 'h': h_dtype, 'w': w_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float32, mstype.float16), self.name)
        validator.check_tensor_dtype_valid('seq_lengths_dtype', seq_lengths_dtype, [mstype.int32], self.name)
        return x_dtype, x_dtype, x_dtype, x_dtype


class LSTMV2(Primitive):
    """
    Performs the Long Short-Term Memory (LSTM) on the input.

    For detailed information, please refer to :class:`mindspore.nn.LSTM`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked LSTM.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`.
        bidirectional (bool): Specifies whether it is a bidirectional LSTM.
        dropout (float, optional): If not 0, append `Dropout` layer on the outputs of each
            LSTM layer except the last layer. The range of dropout is [0.0, 1.0]. Default: 0.0.
        is_train (bool): Specifies whether it is training mode or inference mode.

    Inputs:
        - **input** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`).
        - **h** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **c** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **w** (Tensor) - The input tensor which states for weights.
        - **seq_lengths** (Tensor) - The Tensor[int32] of shape (batch_size, ),
          indicates the seq_length of each batch dim.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`, `c_n`, `reserve`, `state`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **h_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **c_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **reserve** (Tensor) - Tensor of shape (r, 1).
        - **state** (Tensor) - Random number generator state and its shape is (s, 1).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias` or `bidirectional` is not a bool.
        TypeError: If `dropout` is not a float.
        ValueError: If `dropout` is not in range [0.0, 1.0].

    Supported Platforms:
        ``GPU``

    Examples:
        >>> input_size = 10
        >>> hidden_size = 2
        >>> num_layers = 1
        >>> max_seq_len = 5
        >>> batch_size = 2
        >>>
        >>> import mindspore.ops.operations._rl_inner_ops as rl_ops
        >>> net = rl_ops.LSTMV2(input_size, hidden_size, num_layers, True, False, 0.0)
        >>> input_tensor = Tensor(np.ones([max_seq_len, batch_size, input_size]).astype(np.float32))
        >>> h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        >>> c0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        >>> w = Tensor(np.ones([112, 1, 1]).astype(np.float32))
        >>> seq_lengths = Tensor(np.array([4, 3]).astype(np.int32))
        >>> output, hn, cn, _, _ = net(input_tensor, h0, c0, w, seq_lengths)
        >>> print(output)
        Tensor(shape=[5, 2, 2], dtype=Float32, value=
        [[[ 9.64026690e-01, 9.64026690e-01],
        [ 9.64026690e-01, 9.64026690e-01]],
        [[ 9.95053887e-01, 9.95053887e-01],
        [ 9.95053887e-01, 9.95053887e-01]],
        [[ 9.99328434e-01, 9.99328434e-01],
        [ 9.99328434e-01, 9.99328434e-01]],
        [[ 9.99908388e-01, 9.99908388e-01],
        [ 0.00000000e+00, 0.00000000e+00]],
        [[ 0.00000000e+00, 0.00000000e+00],
        [ 0.00000000e+00, 0.00000000e+00]]])
    """

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout, is_train=True):
        """Initialize GRU."""
        validator.check_positive_int(input_size, "input_size", self.name)
        validator.check_positive_int(hidden_size, "hidden_size", self.name)
        validator.check_positive_int(num_layers, "num_layers", self.name)
        validator.check_value_type("has_bias", has_bias, (bool,), self.name)
        validator.check_value_type("bidirectional", bidirectional, (bool,), self.name)
        validator.check_value_type("dropout", dropout, [float], self.name)
        validator.check_float_range(dropout, 0, 1, validator.INC_BOTH, 'dropout', self.name)
        validator.check_value_type("is_train", is_train, (bool,), self.name)


class CudnnGRU(Primitive):
    """
    Performs the Stacked GRU (Gated Recurrent Unit) on the input.

    For detailed information, please refer to :class:`mindspore.nn.GRU`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked GRU.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`.
        bidirectional (bool): Specifies whether it is a bidirectional GRU.
        dropout (float): If not 0, append `Dropout` layer on the outputs of each
            GRU layer except the last layer. The range of dropout is [0.0, 1.0].

    Inputs:
        - **input** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`) or
          (batch_size, seq_len, `input_size`).
        - **h** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **w** (Tensor) - The input tensor which states for weights.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`, `reserve`, `state`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **h_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **reserve** (Tensor) - Tensor of shape (r, 1).
        - **state** (Tensor) - Random number generator state and its shape is (s, 1).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias` or `bidirectional` is not a bool.
        TypeError: If `dropout` is not a float.
        ValueError: If `dropout` is not in range [0.0, 1.0].

    Supported Platforms:
        ``GPU``

    Examples:
        >>> input_size = 10
        >>> hidden_size = 2
        >>> num_layers = 1
        >>> max_seq_len = 5
        >>> batch_size = 2
        >>> seq_len = Tensor(np.array([3, 4], np.int32))
        >>> import mindspore.ops.operations._rl_inner_ops as rl_ops
        >>> net = rl_ops.CudnnGRU(input_size, hidden_size, num_layers, True, False, 0.0)
        >>> input_np = np.ones([batch_size, max_seq_len, input_size]).astype(np.float32)
        >>> input_np[0, 3:, :] = 0
        >>> input_np[1, 4:, :] = 0
        >>> input_tensor = Tensor(input_np)
        >>> h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        >>> w = Tensor(np.ones([84, 1, 1]).astype(np.float32))
        >>> output, hn,  _, _ = net(input_tensor, h0, w)
        >>> print(output)
        [[[1.  1. ]
          [1.  1. ]]
         [[1.  1. ]
          [1.  1. ]]
         [[1.  1.]
          [1.  1.]]
         [[1.  1. ]
          [1.  1. ]]
         [[1.  1. ]
          [1.  1. ]]]
    """

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        """Initialize GRU."""
        self.input_size = validator.check_positive_int(input_size, "input_size", self.name)
        self.hidden_size = validator.check_positive_int(hidden_size, "hidden_size", self.name)
        self.num_layers = validator.check_positive_int(num_layers, "num_layers", self.name)
        self.has_bias = validator.check_value_type("has_bias", has_bias, (bool,), self.name)
        self.bidirectional = validator.check_value_type("bidirectional", bidirectional, (bool,), self.name)
        self.dropout = validator.check_value_type("dropout", dropout, [float], self.name)
        self.dropout = validator.check_float_range(dropout, 0, 1, validator.INC_BOTH, 'dropout', self.name)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
