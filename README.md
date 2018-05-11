# neuralnetworkdemo
Neural Network Demo

![4.png](tests/4.png) ![2.png](tests/2.png)

# Compilation
```
mkdir build
cd build
cmake ../src
make -j9 && make test
```

# Usage

To train the network (using previously trained network)
```
./neuralnetworkdemo -t -i ../tests/image.ann -o ../tests/image.ann
```

To train the network from scratch
```
./neuralnetworkdemo -t -o ../tests/image.ann
```

To use the trained network to classify an image (i.e. recognize the hand-writing)
```
./neuralnetworkdemo -f ../tests/2.png -i ../tests/image.ann
```
