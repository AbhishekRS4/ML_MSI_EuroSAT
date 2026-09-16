# A repo with Ray Torch distributed ML experiments on EuroSAT multi-spectral imagery dataset


## Experiments
* The goal of the project is to learn and expand new skills -
    * Classification experiments for studying the effects of using Kolmogorov-Arnold Networks (KAN) on the LULC classification model performance
    * To train models using Ray Torch distributed training on streaming data from huggingface hub without downloading the data to the disk.
    * To experiment and compare the latency in using REST API with gRPC for inference serving


## Instruction to run the code
* Refer to [scripts/set_python_path.sh](scripts/set_python_path.sh) to set the python root path
* For running torch based training with the data downloaded to the disk, use the script [src/run_torch_trainer.py](src/run_torch_trainer.py)
* For running ray torch based distributed training with the streaming data directly from huggingface hub, use the script [src/run_ray_torch_trainer.py](src/run_ray_torch_trainer.py)
* For running ray torch based evaluation with the streaming data directly from huggingface hub, use the script [src/run_ray_torch_eval.py](src/run_ray_torch_eval.py)


## Performance Results

| Model         | Bands                        | Val Acc.  | Val F1    | Test Acc. | Test F1   |
| ------------- | ---------------------------- | --------- | --------- | --------- | --------- |
| ResNet        | B, G, R, NIR                 |  0.9744   |  0.9745   |  0.9781   |  0.9782   |
| ResKANet      | B, G, R, NIR                 |  0.9761   |  0.9761   |  0.9798   |  0.9798   |
| SE_ResNet     | B, G, R, NIR                 |  0.9750   |  0.9750   |  0.9780   |  0.9780   |
| SE_ResKANet   | B, G, R, NIR                 |  0.9776   |  0.9776   |  0.9796   |  0.9796   | 

* From the performance results, it is pretty clear that using KAN layer produces better results than normal dense layer for classification


## To build and deploy custom gRPC inference service
* The following script can be used to generate the gRPC stub files used in the inference serving code
[scripts/generate_grpc_protobuf_scripts.sh](scripts/generate_grpc_protobuf_scripts.sh)
* The following script can be used to build the docker container image for gRPC inference serving [scripts/build_grpc_docker_container.sh](scripts/build_grpc_docker_container.sh)
* To run the gRPC inference service, the following script can be run either in a docker container (recommended) or as standalone [src/inference_service_grpc/grpc_serve.py](src/inference_service_grpc/grpc_serve.py)
* To run the gRPC client, the following script can be run which sends request to the gRPC inference server [src/inference_service_grpc/grpc_client.py](src/inference_service_grpc/grpc_client.py)


## To build and deploy REST API inference service
* The following script can be used to build the docker container image for REST API inference serving [scripts/build_rest_docker_container.sh](scripts/build_rest_docker_container.sh)
* To run the REST API inference service, the following script can be run either in a docker container (recommended) or as standalone [src/inference_service_rest/rest_serve.py](src/inference_service_rest/rest_serve.py)
* To run the client, the following script can be run which sends request to the REST API inference server [src/inference_service_rest/rest_client.py](src/inference_service_rest/rest_client.py)


## To build and deploy Triton inference service
* The following script can be used for converting the model checkpoint from normal PyTorch pickle to TorchScript graph [src/inference_service_triton/convert_pickle_2_torchscript.py](src/inference_service_triton/convert_pickle_2_torchscript.py)
* The converted TorchScript model file needs to be in the following directory structure
```
/path/to/model/repository/
└── <model_name>/
    ├── config.pbtxt       <-- Place it here
    └── 1/                 <-- Version directory (e.g., numbers like 1, 2)
        └── model.pt       <-- TorchScript model file
```
* The following script can be used for running the docker container for triton inference serving [scripts/run_triton_docker_service.sh](scripts/run_triton_docker_service.sh)
* The following is the example usage of the above bash script. The `--model-repo` should include the directory containing the model files where the name of the model should match the name in the `config.pbtxt` file
```
bash scripts/run_triton_docker_service.sh --model-repo src/inference_service_triton/
```
* The following script can be used for running the triton client that can send the data to the triton server and get the result [src/inference_service_triton/triton_client.py](src/inference_service_triton/triton_client.py)


## Inference Serving Latency Results
* The following table shows the performance results of latency in using REST based FastAPI vs gRPC for inference serving
* The total time taken includes sending the image from client to the inference server and performing inference on the same system with the same GPU

| Serving methodology | Num Workers |  Batch Size | Time taken for inference per batch (in milli sec.) |
| ------------------- | ----------- | ----------- | ---------------------------------------------------|
|      FastAPI        |      4      |      16     |                   ~ 60 - 70                        |
|       gRPC          |      4      |      16     |                   ~ 60 - 65                        |
|      Triton (HTTP)  |             |      16     |                   ~ 70 - 75                        |
|      Triton (gRPC)  |             |      16     |                   ~ 70 - 75                        |

* From the results, it is pretty clear that the latency and the total inference time is slightly faster with gRPC when compared with that of FastAPI. The test has been performed locally with a single client. This would certainly differ in case of high request volume with many concurrent clients. Also, gRPC would be best suited for real-time streaming inference applications.


## Dependency management
* `uv` can be used for installing and managing all the dependencies for newer GPUs
* In case of CUDA version issues for older GPUs, `requirements-grpc-serve.txt` can be used for managing dependencies for gRPC inference serving and `requirements-rest-serve.txt` can be used for managing dependencies for REST API inference serving
* The above are specific to torch and CUDA version dependency management