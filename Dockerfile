FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
        git \
        python3 \
        python-is-python3 \
        python3-pip


RUN cd ~ && \
    git clone https://mirror.ghproxy.com/https://github.com/AbdusalamAblimit/re-id.git && \
    cd re-id && \
    bash ./install-requirements.bash


WORKDIR /root/re-id

CMD ["/bin/bash"]
