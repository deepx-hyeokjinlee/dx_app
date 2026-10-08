UBUNTU_VERSION?=20.04
APP_NAME:=dx-app-$(UBUNTU_VERSION)
APP_VERSION:=$(shell cat release.ver)
ARCH?=x86_64
GH_OWNER?=deepx
GH_BRANCH?=main
GH_USER?=
GH_TOKEN?=
MODEL_PATH?=
# Local sources for run_tc: setup.sh --internal has no default paths
INTERNAL_MODEL_PATH?=/mnt/regression_storage/atd/models_v3.2.0
INTERNAL_VIDEO_PATH?=/mnt/regression_storage/atd/sample_videos_v3.2.2.tar.gz

.PHONY: all build_app_image container

# Check for Ubuntu Version
SUPPORTED_UBUNTU_VERSIONS := 18.04 20.04 22.04
ifneq ($(filter $(UBUNTU_VERSION),$(SUPPORTED_UBUNTU_VERSIONS)),)
else
$(error UBUNTU_VERSION $(UBUNTU_VERSION) is not supported. Supported version is 20.04 & 22.04)
endif

# Check for CPU Architecture
SUPPORTED_CPU_ARCH := x86_64 aarch64 
ifneq ($(filter $(ARCH),$(SUPPORTED_CPU_ARCH)),)
else
$(error ARCH $(ARCH) is not supported. Supported CPU is x86_64 & aarch64)
endif

# Check for Github Owner
SUPPORTED_GH_OWNER := KOMOSYS DEEPX-AI deepx
ifneq ($(filter $(GH_OWNER),$(SUPPORTED_GH_OWNER)),)
else
$(error GH_OWNER $(GH_OWNER) is not supported. Supported github owner is KOMOSYS & DEEPX-AI)
endif

# Define docker run options
DOCKER_RUN_OPTIONS := --rm --name $(APP_NAME)-$(APP_VERSION)

# Add volume options for models
ifneq ($(MODEL_PATH),)
DOCKER_RUN_OPTIONS += -v $(MODEL_PATH):/model
endif

# Run dx_app Docker container
container:
	@echo "Run dx_app Container"
	docker run -it $(DOCKER_RUN_OPTIONS) \
			   -v ./example:/example \
			   --device="/dev/*dma*/" \
			   --privileged \
			   $(APP_NAME):$(APP_VERSION) \
			   /bin/bash

# Run imagenet example demo
imagenet_example:
	@echo "Run imagenet sample demo"
	docker run -i $(DOCKER_RUN_OPTIONS) \
			   -v ./example:/example \
			   --device="/dev/*dma*/" \
			   --privileged \
			   $(APP_NAME):$(APP_VERSION) \
			   /dx_app/bin/run_classifier -c /example/imagenet_example.json

# Run YOLOV5 detector example demo
detector_yolov5s3_example:
	@echo "Run detector sample demo"
	docker run -i $(DOCKER_RUN_OPTIONS) \
			   -v ./example:/example \
			   --device="/dev/*dma*/" \
			   --privileged \
			   $(APP_NAME):$(APP_VERSION) \
			   /dx_app/bin/run_detector -c /example/yolov5s3_example.json

# Run TC Scripts
run_tc:
	@set -e; \
	echo "Run cpp / python TC Scripts"; \
	pip install -r tests/cpp_example/requirements.txt; \
	pip install -r tests/python_example/requirements.txt; \
	echo "Setup dxnn models and videos"; \
	./setup.sh --force-remove-models --internal \
		--internal-path=$(INTERNAL_MODEL_PATH) --internal-video-path=$(INTERNAL_VIDEO_PATH); \
	echo "Run cpp test script (quick version)"; \
	./run_tc.sh --cpp --cli --e2e-quick --loop 2; \
	echo "Run python test script (quick version)"; \
	./run_tc.sh --python --cli --e2e-quick; \
	

# Run TC Scripts for Sonarqube analysis
run_tc_sonar:
	@set -e; \
    echo "Run cpp coverage (recommended: full)"; \
    ./run_tc.sh --cpp --coverage; \
    echo "Generate C++ coverage report (REQUIRED for SonarQube)"; \
    gcovr -r . --sonarqube -o coverage_cpp.xml \
        --gcov-ignore-parse-errors=suspicious_hits.warn \
        -e ".*test.*" -e ".*third_party.*" -e ".*extern.*"; \
    echo "Run python coverage (recommended: full)"; \
    ./run_tc.sh --python --coverage; \
    echo "Generate Python coverage report (REQUIRED for SonarQube)"; \
	mv tests/python_example/.coverage .coverage; \
    coverage xml -o coverage.xml

# Build dx_app Docker Image
app_image:
	@echo "Start Build image for dx_app"
	@if [ -z "$(GH_USER)" ]; then echo "GH_USER argument None. Please insert GitHub User Name" && exit 1; fi
	@if [ -z "$(GH_TOKEN)" ]; then echo "GH_TOKEN argument None. Please insert GitHub User Token" && exit 1; fi
	@echo "Build options : UBUNTU_VERSION=$(UBUNTU_VERSION) APP_VERSION=$(APP_VERSION) GH_OWNER=$(GH_OWNER) GH_BRANCH=$(GH_BRANCH) GH_USER=$(GH_USER) GH_TOKEN=$(GH_TOKEN)"
	docker build --no-cache \
				 --tag $(APP_NAME):$(APP_VERSION) \
	   			 -f ./docker/Dockerfile.app.build \
				 --build-arg "UBUNTU_VERSION=$(UBUNTU_VERSION)" \
				 --build-arg "APP_VERSION=$(APP_VERSION)" \
				 --build-arg "ARCH=$(ARCH)" \
				 --build-arg "GH_OWNER=$(GH_OWNER)" \
				 --build-arg "GH_BRANCH=$(GH_BRANCH)" \
				 --build-arg "GH_USER=$(GH_USER)" \
				 --build-arg "GH_TOKEN=$(GH_TOKEN)" \
				 .
	docker save $(APP_NAME):$(APP_VERSION) | gzip > ./docker/$(APP_NAME)-$(APP_VERSION).tar.gz

# Remove dx_app Docker Image
clean:
	@echo "Remove Build image for dx_app"
	docker rmi $(APP_NAME):$(APP_VERSION) || true
