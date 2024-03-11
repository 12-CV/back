1. compress_model.sh 사용법
이유 - 깃허브에 100메가 이상은 안올라가서 귀찮지만 압축해서 올리셔야 합니다..
```
chmod +x compress_model.sh

./compress_model.sh {모델 파일 이름}

ex
./compress_model.sh yolow-l.onnx
```

2. decompress_model.sh 사용법
```
chmod +x decompress_model.sh

./decompress_model.sh {모델 압축 파일 이름}

ex
./decompress_model.sh yolow-l.onnx.7z
```