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

.PHONY: all build_app_image container check_dxrt_service

# Check for Ubuntu Version
SUPPORTED_UBUNTU_VERSIONS := 20.04 22.04
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

# Every NPU test needs the DX-RT service (dxrtd). Without it each one fails
# with "dxrt service is not running", so the test targets check it first
# instead of finding out after the whole run.
check_dxrt_service:
	@if ! command -v dxrt-cli >/dev/null 2>&1; then \
		echo "ERROR: dxrt-cli not found. Install DX-RT (dx_rt) first."; \
		exit 1; \
	fi; \
	if ! out=$$(dxrt-cli -s 2>&1); then \
		echo "$$out"; \
		echo ""; \
		echo "ERROR: 'dxrt-cli -s' failed (output above), so every NPU test would fail."; \
		echo "  Check that the NPU is present and the DX-RT service (dxrtd) is reachable:"; \
		echo "  - Host:      sudo systemctl start dxrt   (check: systemctl status dxrt)"; \
		echo "  - Container: run dxrtd in exactly one place. Share the host's dxrtd"; \
		echo "               (docker run --pid=host --ipc=host --network=host ...), or"; \
		echo "               start dxrtd inside the container after stopping the host's."; \
		exit 1; \
	fi; \
	echo "DX-RT service: OK"

# Run TC Scripts
run_tc: check_dxrt_service
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
	

# Run TC Scripts for Sonarqube analysis. Every step runs even when an earlier
# one fails, so SonarQube still gets both coverage reports; the target fails
# at the end if any step did.
run_tc_sonar: check_dxrt_service
	@failed=""; \
	echo "Run cpp coverage (recommended: full)"; \
	./run_tc.sh --cpp --coverage || failed="$$failed cpp-tests"; \
	echo "Generate C++ coverage report (REQUIRED for SonarQube)"; \
	gcovr -r . --sonarqube -o coverage_cpp.xml \
		--gcov-ignore-parse-errors=suspicious_hits.warn \
		-e ".*test.*" -e ".*third_party.*" -e ".*extern.*" || failed="$$failed cpp-report"; \
	echo "Run python coverage (recommended: full)"; \
	./run_tc.sh --python --coverage || failed="$$failed python-tests"; \
	echo "Generate Python coverage report (REQUIRED for SonarQube)"; \
	if [ -f tests/python_example/.coverage ]; then \
		{ mv tests/python_example/.coverage .coverage && coverage xml -o coverage.xml; } \
			|| failed="$$failed python-report"; \
	else \
		echo "ERROR: tests/python_example/.coverage was not written"; \
		failed="$$failed python-report"; \
	fi; \
	if [ -n "$$failed" ]; then echo "run_tc_sonar failed steps:$$failed"; exit 1; fi

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
