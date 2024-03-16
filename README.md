# re-id


您可以使用以下命令来构建 Docker 镜像：

```bash
Copy code
docker build -t re-id .
```
然后使用以下命令运行容器：

```bash
Copy code
docker run -it --name re-id --gpus all --shm-size "8g" re-id
```