# Machine Translation

```
$ git clone https://github.com/hskaailabnlp/machine_translation.git
$ cd machine_translation
$ docker build -t ai_lab:3 .
$ cd ..
$ docker run -u $(id -u):$(id -g)  --runtime=nvidia  -p 8888:8888 -v $(pwd)/machine_translation:/tf/notebooks ai_lab:3
```
