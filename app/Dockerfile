# base image
FROM		debian:stable-slim
#
MAINTAINER	n0imaginati0n vik.public@gmx.de

ARG APPDIR=brainyscan

# install necessary system-wide packages
RUN	apt update -qq \
	&& apt install --no-install-recommends -y \
		python3 \
		python3-setuptools \
		python3-venv \
		python3-h5py \
		python3-pydot \
		unzip \
	&& apt clean \
	&& rm -rf /va/lib/apt/lists/*

# organize python virtual environment
RUN	mkdir -p /srv/common
RUN --mount=type=bind,source="$APPDIR/requirements.txt",target="/tmp/requirements.txt" \
    python3 -m venv "/srv/common/.venv" \
    && "/srv/common/.venv/bin/pip3" --no-cache-dir install pip --upgrade \
    && "/srv/common/.venv/bin/pip3" --no-cache-dir install -r "/tmp/requirements.txt"

# copy app files into /srv/app directory
RUN  mkdir -p /srv/common/brainy
COPY "$APPDIR/app/"      "/srv/common/brainy/app/"
COPY "$APPDIR/instance/" "/srv/common/brainy/instance/"
COPY "$APPDIR/wsgi.py"   "/srv/common/brainy/"

EXPOSE 8080

WORKDIR /srv/common/brainy
ENTRYPOINT ["/srv/common/.venv/bin/flask"]
CMD ["--app", "wsgi.py", "run", "--host=0.0.0.0", "--port=8080"]
