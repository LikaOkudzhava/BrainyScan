
IMG = n0imaginati0n/brainy:v1
CNTNAME = brainyapp

build:
	docker build -f ./Dockerfile -t $(IMG) .

publish: build
	docker push $(IMG)

clean:
	docker buildx prune -f

clean-all:
	docker buildx prune -f -a

run:
	docker run --detach -p 8080:8080 --name $(CNTNAME) $(IMG)

inspect:
	# --rm - remove on exit everything
	docker run -p 8080:8080 --rm --interactive --tty --entrypoint /usr/bin/bash $(IMG)

