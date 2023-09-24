#FROM python:3.9-slim
FROM pytorch/pytorch


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output/images/automated-petct-lesion-segmentation \
    && chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip


COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm monai_unet.py /opt/algorithm/
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm fold0_model_ep=0262.pth /opt/algorithm/
COPY --chown=algorithm:algorithm fold1_model_ep=0238.pth /opt/algorithm/
COPY --chown=algorithm:algorithm fold2_model_ep=0236.pth /opt/algorithm/
COPY --chown=algorithm:algorithm fold3_model_ep=0230.pth /opt/algorithm/
COPY --chown=algorithm:algorithm fold4_model_ep=0202.pth /opt/algorithm/

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@
