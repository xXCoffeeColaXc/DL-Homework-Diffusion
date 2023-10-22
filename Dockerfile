FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update &&   \
    apt install -y      \
    openssh-server


ENV HOME /home/custom_user
RUN mkdir -p $HOME/dl_homework_diffusion
WORKDIR $HOME/dl_homework_diffusion

COPY . .
COPY ssh/sshd_config /etc/ssh/sshd_config

EXPOSE 22
EXPOSE 8888

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 custom_user
RUN echo 'custom_user:password' | chpasswd

SHELL ["/bin/bash", "-l", "-c"]

