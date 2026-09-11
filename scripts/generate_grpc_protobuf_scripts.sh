python -m grpc_tools.protoc \
    -I./src/inference_service_grpc \
    --python_out=./src/inference_service_grpc \
    --grpc_python_out=./src/inference_service_grpc \
    ./src/inference_service_grpc/inference.proto