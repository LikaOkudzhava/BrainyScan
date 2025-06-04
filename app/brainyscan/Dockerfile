# base image
FROM		debian:stable-slim
# 
MAINTAINER	n0imaginati0n vik.public@gmx.de

# install necessary system-wide packages
RUN	apt update -qq \
	&& apt install --no-install-recommends -y \
		git \
		python3 \
		python3-setuptools \
		python3-venv \
		python3-numpy \
		python3-h5py \
		python3-yaml \
		python3-pydot \
		unzip \
	&& apt clean \
	&& rm -rf /va/lib/apt/lists/*

# organize python virtual environment
RUN		mkdir -p /srv/common
COPY 	../app/requirements.txt /srv/common/requirements.txt
RUN		python3 -m venv /srv/common/.venv \
	&& /srv/common/.venv/bin/pip3 --no-cache-dir install pip --upgrade \
	&& /srv/common/.venv/bin/pip3 --no-cache-dir install -r /srv/common/requirements.txt

# copy app files into /srv/app directory
RUN		mkdir -p /srv/app
COPY	../app/server/openapi_server	/srv/app

# copy model file
RUN		mkdir -p /srv/app/model
COPY	../models/model.keras /srv/app/model.keras

WORKDIR /srv/app

EXPOSE 8080



WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY instance/ instance/
COPY wsgi.py ./

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "wsgi:app"]

# ENTRYPOINT ["/srv/common/.venv/python3"]
# CMD ["-m", "/srv/app"]

