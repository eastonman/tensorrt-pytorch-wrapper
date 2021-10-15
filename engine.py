import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import os
from typing import List
from pycuda.gpuarray import GPUArray
import torch

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


def torch_dtype_to_numpy(dtype):
    """ Helper function"""
    dtype_name = str(dtype)[6:]     # remove 'torch.'
    return getattr(np, dtype_name)


def numpy_dtype_to_torch(dtype):
    """ Helper function"""
    dtype_name = str(dtype)[6:]     # remove 'np.'
    return getattr(torch, dtype_name)


def tensor_to_gpuarray(tensor: torch.Tensor):
    '''Convert a :class:`torch.Tensor` to a :class:`pycuda.gpuarray.GPUArray`. The underlying
    storage will be shared, so that modifications to the array will reflect in the tensor object.
        Parameters
        ----------
        tensor  :   torch.Tensor

        Returns
        -------
        pycuda.gpuarray.GPUArray

        Raises
        ------
        ValueError
            If the ``tensor`` does not live on the gpu
    '''
    if not tensor.is_cuda:
        print(tensor.shape)
        raise ValueError(
            'Cannot convert CPU tensor to GPUArray (call `cuda()` on it)')
    if not tensor.is_contiguous():
        print(tensor.stride())
        raise ValueError(
            'Tensor not in contiguous (NCHW) format')
    return GPUArray(tensor.shape, dtype=torch_dtype_to_numpy(tensor.dtype), gpudata=tensor.data_ptr())


def gpuarray_to_tensor(gpuarray, out=None):
    '''Convert a :class:`pycuda.gpuarray.GPUArray` to a :class:`torch.Tensor`. The underlying
    storage will NOT be shared, since a new copy must be allocated.
        Parameters
        ----------
        gpuarray  :   pycuda.gpuarray.GPUArray

        Returns
        -------
        torch.Tensor
    '''
    if out == None:
        shape = gpuarray.shape
        dtype = gpuarray.dtype
        out_dtype = numpy_dtype_to_torch(dtype)
        out = torch.zeros(shape, dtype=out_dtype).cuda()

    assert out.shape == gpuarray.shape

    gpuarray_copy = tensor_to_gpuarray(out)
    byte_size = gpuarray.itemsize * gpuarray.size
    # Do dtod copy, should be fast
    cuda.memcpy_dtod(gpuarray_copy.gpudata,
                     gpuarray.gpudata, byte_size)
    return out


class Engine():
    def __init__(self, outputs: List[torch.Tensor], max_batch_size=1, onnx_file_path="", engine_file_path="", fp16_mode=False, int8_mode=False):

        # must create cuda context before engine execution context
        # cuda.init()
        # self.cfx = cuda.Device(0).make_context()
        # self.cfx.push()

        self.engine = get_engine(max_batch_size, onnx_file_path,
                                 engine_file_path, fp16_mode, int8_mode, save_engine=True)
        self.context = self.engine.create_execution_context()

        # CUDA related
        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        output_idx = 0
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                shape = self.engine.get_binding_shape(
                    binding)
                print(shape)
                # Append the device buffer to device bindings.
                self.bindings.append(0)
                # Append to the appropriate list.
                self.inputs.append(0)
            else:
                shape = self.engine.get_binding_shape(
                    binding)
                print(shape)
                if output_idx == len(outputs):
                    quit("outputs tensor number not matching")
                assert shape == outputs[output_idx].shape
                self.bindings.append(outputs[output_idx].data_ptr())
                self.outputs.append(tensor_to_gpuarray(outputs[output_idx]))
                output_idx += 1

    def __del__(self):
        pass
        # self.cfx.pop()

    def gpu_forward(self, inputs: List[torch.Tensor], outputs: List[torch.Tensor] = None) -> List[GPUArray]:
        """ gpu_forward accept cuda input and returns cuda output """
        if outputs != None:  # Check output
            if len(outputs) != len(self.outputs):
                quit("output length {} not equal to engine output length {}".format(
                    len(outputs), len(self.outputs)))
        self.inputs = [tensor_to_gpuarray(x) for x in inputs]
        self.bindings[0] = self.inputs[0].gpudata

        self._do_inference(len(inputs))
        if outputs != None:
            # self.cfx.push()
            for i in range(len(outputs)):
                gpuarray_to_tensor(
                    self.outputs[i], out=outputs[i], context=self.cfx)
            # self.stream.synchronize()
            # self.cfx.pop()
        else:
            return self.outputs

    def forward(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
            forward accept a list of numpy array
            in NCHW format

            returns a list of numpy array
        """
        raise NotImplementedError
        assert len(inputs) == len(self.inputs)

        # memory copy on CPU, save the size
        sizes = []
        for idx, i in enumerate(inputs):
            sizes.append(i.shape)
            self.inputs[idx].host = np.expand_dims(i, axis=0)

        # copy from cpu to device
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
         for inp in self.inputs]

        # perform inference
        self._do_inference(inputs[0].shape[0])

        # Transfer predictions back from the GPU.
        # [cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        #  for out in self.outputs]
        result = []
        # for idx, out in enumerate(self.outputs):
        #     result.append(out.host.reshape(sizes[idx]))
        return self.outputs

    def _do_inference(self, batch_size: int) -> List[GPUArray]:
        """
            self.inputs and self.outputs should be ready now
        """
        # self.cfx.push()
        # Run inference.
        self.context.execute_async(
            batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Synchronize the stream
        # self.stream.synchronize()
        # self.cfx.pop()

        # return gpuarray id needed
        return self.outputs


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",
               fp16_mode=False, int8_mode=False, save_engine=False,
               ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_batch_size = max_batch_size

            # Builder Config
            config.max_workspace_size = 1 << 30  # Your workspace size
            if int8_mode:
                raise NotImplementedError
            if fp16_mode:
                config.flags = 1 << int(trt.BuilderFlag.FP16)

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            if parser.parse_from_file(onnx_file_path) != True:
                quit("parsing onnx model error")

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(
                onnx_file_path))

            engine = builder.build_engine(network, config)
            if engine == None:
                quit("Building engine failed")
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)
