NAME=masters-eval
VERSION=0.2
TAG=ccu.uni-konstanz.de:5000/benjamin.wilhelm/${NAME}
# COMMAND="python /scripts/eval_sisr.py /edsr.csv methods.basicsr.edsr --test-run"
COMMAND="python /scripts/run_per_model.py only_noise_range /data/models /scripts/eval_sisr.py /{mn}.csv methods.hqs {mp} --test-run"
# COMMAND="python --version"

build:
	docker build \
		-f Dockerfile \
		-t ${TAG}:${VERSION} \
		../..

push:
	docker push ${TAG}:${VERSION}

run:
	docker run \
		--gpus all \
		--device /dev/nvidia0 \
		--device /dev/nvidia-uvm \
		--device /dev/nvidia-uvm-tools \
		--device /dev/nvidiactl \
		--name ${NAME} \
		--mount type=bind,source=$(shell pwd)/../..,target=/data,readonly \
		${TAG}:${VERSION} \
		"${COMMAND}"

clean:
	docker container rm ${NAME}