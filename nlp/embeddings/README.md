# Embeddings
```
$ git clone https://github.com/hskaailabnlp/embeddings.git
$ cd embeddings
$ docker build -t ai_lab:embeddings .
$ cd ..
$ docker run -u $(id -u):$(id -g) --runtime=nvidia  -p 8888:8888 -v $(pwd)/embeddings:/tf/notebooks ai_lab:embeddings
```