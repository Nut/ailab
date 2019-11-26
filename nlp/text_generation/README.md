# Text Generation
```
$ git clone https://github.com/hskaailabnlp/text_generation.git
$ cd text_generation
$ docker build -t ai_lab:3 .
$ cd ..
$ docker run -u $(id -u):$(id -g) --runtime=nvidia  -p 8888:8888 -v $(pwd)/text_generation:/tf/notebooks ai_lab:3
```
