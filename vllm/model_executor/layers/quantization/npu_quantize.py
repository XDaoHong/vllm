from typing import Any, Dict, List, Optional

import torch
import torch_npu
from torch.nn.parameter import Parameter
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


class NpuA8W8GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int = 8,
    ) -> None:
        self.weight_bits = weight_bits
        if self.weight_bits not in [8]:
            raise ValueError(
                "Currently, only 8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}")

    @classmethod
    def get_name(cls) -> str:
        return "gptq_ascend"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        return cls(weight_bits)

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["NpuA8W8GPTQLinearMethod"]:
        if isinstance(layer, LinearBase):
            return NpuA8W8GPTQLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class NpuA8W8GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: NpuA8W8GPTQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        output_size_per_partition = sum(output_partition_sizes)

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight, {"output_dim": 0, "input_dim": 1})
        deq_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                dtype=torch.int64,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            deq_scale, {"output_dim": 0})
        scales_w = Parameter(
            torch.empty(
                output_size_per_partition,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales_w, {"output_dim": 0})

        scale_x = Parameter(torch.empty(input_size_per_partition, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(scale_x, {"input_dim": 0})
        offset_x = Parameter(torch.empty(input_size_per_partition, dtype=torch.int32), requires_grad=False)
        set_weight_attrs(offset_x, {"input_dim": 0})

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("deq_scale", deq_scale)
        set_weight_attrs(deq_scale, extra_weight_attrs)
        layer.register_parameter("scales_w", scales_w)
        set_weight_attrs(scales_w, extra_weight_attrs)

        layer.register_parameter("scale_x", scale_x)
        set_weight_attrs(scale_x, extra_weight_attrs)
        layer.register_parameter("offset_x", offset_x)
        set_weight_attrs(offset_x, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight
        out_shape = x.shape[:-1] + (weight.shape[0], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        reshaped_x = torch_npu.npu_quantize(reshaped_x, layer.scale_x, layer.offset_x, torch.qint8, axis=-1)
        y = torch_npu.npu_quant_matmul(reshaped_x,
                                       layer.weight.transpose(0, 1),
                                       layer.deq_scale,
                                       offset=None,
                                       bias=bias,
                                       output_dtype=torch.float16)

        return y.view(out_shape)


class ColumnParallelA8W8Linear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = torch.int8,
                 quant_config: Optional[QuantizationConfig] = None,
                 output_sizes: Optional[List[int]] = None):
        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config)

        self.gather_output = gather_output
        self.bias_dtype = torch.float16
        if self.params_dtype == torch.int8:
            self.bias_dtype = torch.int32

        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size()
        assert self.quant_method is not None
        self.output_size_per_partition = divide(self.output_size, tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, tp_size)
                for output_size in self.output_sizes
            ]

        if output_sizes is None:
            output_sizes = [output_size]
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=self.bias_dtype),
                requires_grad=False)
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # Special case for Fp8 scales.
        fp8_scales_shard_indexer = getattr(param, "fp8_scales_shard_indexer",
                                           None)

        tp_rank = get_tensor_model_parallel_rank()
        output_dim = getattr(param, "output_dim", None)
        param_data = param.data
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)
        # Special case for Fp8 scales.
        elif fp8_scales_shard_indexer is not None:
            param_data, loaded_weight = fp8_scales_shard_indexer(param_data,
                                                                 loaded_weight,
                                                                 shard_id=0)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        return output

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={get_tensor_model_parallel_world_size()}"
        s += f", gather_output={self.gather_output}"
        return s


class RowParallelA8W8Linear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = torch.int8,
                 reduce_results: bool = True,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.bias_dtype = torch.float16
        if self.params_dtype == torch.int8:
            self.bias_dtype = torch.int32

        # Divide the weight matrix along the last dimension.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.bias_dtype), requires_grad=False)
            set_weight_attrs(self.bias, {
                "input_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # Special case for Fp8 scales.
        fp8_scales_shard_indexer = getattr(param, "fp8_scales_shard_indexer",
                                           None)

        tp_rank = get_tensor_model_parallel_rank()
        input_dim = getattr(param, "input_dim", None)
        param_data = param.data
        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        # Special case for Fp8 scales.
        elif fp8_scales_shard_indexer is not None:
            param_data, loaded_weight = fp8_scales_shard_indexer(param_data,
                                                                 loaded_weight,
                                                                 shard_id=0)

        if fp8_scales_shard_indexer is None and len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_parallel, self.bias)
        if self.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        return output_

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s

