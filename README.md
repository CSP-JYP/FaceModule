# FaceModule
Face Authentication Modules


## information
inside the ipynb file, all training steps included.

encryption module 내부의 `arcface_final.onnx`는 해당 ipynb 노트북에서 훈련된 pretrained network에서 classifier head가 제거된 버전입니다.

## requirements
pytorch (v2.0), onnx (v11), torchvision, imageio, pytorch_metric_learning, opacus, onnxruntime (latest)

scikit-learn, pandas, numpy


---

# Encryption Module
Elgamal Inner Product Functional Encryption

- 암호 구현을 변경하였습니다. 추출된 feature간 내적이 securely 수행되는 방식입니다.
- bn.js를 이용하여 빠른 계산이 가능합니다.
- (빠른)테스트를 위해 비트수는 192비트 소수인 p192를 사용하였습니다. 실제로는 보안성을 위해 훨씬 더 큰 비트수가 필요할 것입니다.

```
x = original data
y = to be proved
client -> server ; send enc(x), enc(y), modulo param p;
enc(x) dot enc(y) mod p = x dot y

g <- Zp(gen)
beta <- Zp
b = g^beta
r <- Zp
xi = b^r * xi
yi = inv(b^r) * yi
xi * b^r * yi * inv(b^r) === xi * yi mod p
```

## requirements
bn.js, onnxruntime-node, deasync, sharp, ndarray, ndarray-ops, numjs

see yarn.lock, package.json for additional infos

- pictures were used from lfw for test, in ./pics, deleted in this repository.

## test result
Normalize 후 내적한다면 (= cos similarity) 0.985 이상이면 동일인으로 판정 가능

440.xxx -> 440, 423.xxx -> 423 으로, 정수 내적값 근사가 가능함.

![image|250](https://github.com/CSP-JYP/FaceModule/assets/42195282/389a6ecd-bd75-4f79-9ad6-db4fd9877c20)
