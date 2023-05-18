.PHONY:	show-params

###############################################################################
# GLOBALS                                                                     #
###############################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# PROJECT_NAME := $(shell basename $(subst -,_,$(PROJECT_DIR)))
PROJECT_NAME := $(shell basename $(subst -,_, $(PROJECT_DIR)) | tr '[:upper:]' '[:lower:]')
ENVIRONMENT_NAME = $(PROJECT_NAME)
PYTHON_INTERPRETER = python3
PIP_INTERPRETER = pip
PYTHON_VERSION = 3.9
PIP_VERSION = 22.3

# --- REQUIREMENTS-RELATED
REQUIREMENTS_FILE = $(PROJECT_DIR)/requirements.txt
REQUIREMENTS_FILE_TEMP = $(PROJECT_DIR)/requirements.tmp
REQUIREMENTS_DEV_FILE = $(PROJECT_DIR)/requirements-dev.txt
REQUIREMENTS_DEV_FILE_TEMP = $(PROJECT_DIR)/requirements-dev.tmp

# -- Docker-related
# Variable used for turning on/off Docker Buildkit
DOCKER_BUILDKIT_VALUE=1
LOCAL_DEVELOPMENT_DIR_PATH="$(PROJECT_DIR)/docker"

# ----------------------------- Python-specific -------------------------------
# - Checking what type of python one is using
# Anaconda
ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
# We need to specify the following commands in order to properly activate the
# Anaconda environment.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONDA_DEACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda deactivate ; conda deactivate
endif

# - Pyenv
ifeq (,$(shell which pyenv))
HAS_PYENV=False
else
HAS_PYENV=True
endif

###############################################################################
#  VARIABLES FOR COMMANDS                                                     #
###############################################################################

## Show the set of input parameters
show-params:
	@ printf "\n-------- GENERAL ---------------\n"
	@ echo "PROJECT_DIR:                       $(PROJECT_DIR)"
	@ echo "PROJECT_NAME:                      $(PROJECT_NAME)"
	@ echo "LOCAL_DEVELOPMENT_DIR_PATH:        $(LOCAL_DEVELOPMENT_DIR_PATH)"
	@ echo "ENVIRONMENT_NAME:                  $(ENVIRONMENT_NAME)"
	@ echo "PYTHON_INTERPRETER:                $(PYTHON_INTERPRETER)"
	@ echo "PYTHON_VERSION:                    $(PYTHON_VERSION)"
	@ echo "PIP_VERSION:                       $(PIP_VERSION)"
	@ echo "REQUIREMENTS_FILE:                 $(REQUIREMENTS_FILE)"
	@ echo "REQUIREMENTS_FILE_TEMP:            $(REQUIREMENTS_FILE_TEMP)"
	@ echo "REQUIREMENTS_DEV_FILE:             $(REQUIREMENTS_DEV_FILE)"
	@ echo "REQUIREMENTS_DEV_FILE_TEMP:        $(REQUIREMENTS_DEV_FILE_TEMP)"
	@ printf "\n-------- DOCKER ---------------\n"
	@ echo "DOCKER_BUILDKIT_VALUE:             $(DOCKER_BUILDKIT_VALUE)"
	@ printf "\n-------- PYTHON ---------------\n"
	@ echo "HAS_CONDA:                         $(HAS_CONDA)"
	@ echo "HAS_PYENV:                         $(HAS_PYENV)"
	@ printf "\n-------- LOCAL DEVELOPMENT ---------------\n"
	@ echo "LOCAL_DEV_DOCKER_PROJECT_NAME:     $(LOCAL_DEV_DOCKER_PROJECT_NAME)"
	@ echo "LOCAL_DEV_SERVICE_NAME:            $(LOCAL_DEV_SERVICE_NAME)"
	@ printf "\n-----------------------\n"

## Initialize the repository for code development
init: clean create-envrc delete-environment create-environment
ifeq (True,$(HAS_CONDA))
	@ ($(CONDA_ACTIVATE) $(ENVIRONMENT_NAME) ; $(MAKE) requirements)
	@ printf "\n\n>>> New Conda environment created. Activate with: \n\t: conda activate $(ENVIRONMENT_NAME)"
	@ $(MAKE) show-params
	@ printf "\n\n>>> Project initialized!"
	@ ($(CONDA_ACTIVATE) $(ENVIRONMENT_NAME) ; $(MAKE) pre-commit-install )
	@ ($(CONDA_ACTIVATE) $(ENVIRONMENT_NAME) ; $(MAKE) lint )
else
	@ direnv allow || echo ""
	@ echo ">>> Continuing installation ..."
	@ $(MAKE) requirements
	@ $(MAKE) show-params
	@ printf "\n\n>>> Project initialized!\n"
	@ $(MAKE) pre-commit-install
	@ $(MAKE) lint
endif

## Remove ALL of the artifacts + Python environments
destroy: clean pre-commit-uninstall delete-environment
	@ echo ">>> Deleted all artifacts and environments!"

###############################################################################
# MISCELLANEOUS COMMANDS                                                      #
###############################################################################

# -------------------- Functions for cleaning repository ----------------------

## Removes artifacts from the build stage, and other common Python artifacts.
clean: clean-build clean-pyc clean-test clean-secrets clean-model-files clean-images

## Removes Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

## Remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

## Remove test and coverage artifacts
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

## Remove files related to pre-trained models
clean-model-files:
	find . -name '*.pt' -exec rm -fr {} +

## Clean left-over images
clean-images:
	find . -name '*.png' -exec rm -fr {} +
	find . -name '*.jpg' -exec rm -fr {} +

## Removes secret artifacts - Serverless
clean-secrets:
	find . -name "node_modules" -type d -exec rm -rf {} + || echo ""
	find . -name ".serverless" -type d -exec rm -rf {} + || echo ""

# ---------------------- Functions for local environment ----------------------

## Set up the envrc file for the project.
create-envrc:
	@ echo "cat $(PROJECT_DIR)/template.envrc > $(PROJECT_DIR)/.envrc"
	@ cat $(PROJECT_DIR)/template.envrc > $(PROJECT_DIR)/.envrc

## Delete the local envrc file of the project
delete-envrc:
	@ rm -rf $(PROJECT_DIR)/.envrc || echo ""

# ---------------------- Functions for Python environment ---------------------

## Creates the Python environment
create-environment:
ifeq (True,$(HAS_CONDA))
	@ echo ">>> Detected CONDA ... Creating new conda environment!"
	@ echo ">>> \tCreating environment: \t $(ENVIRONMENT_NAME)"
	@ conda create --name $(ENVIRONMENT_NAME) python=$(PYTHON_VERSION) -y  || echo ""
	@ echo ">>> New conda environment created. Activate with: \n conda activate $(ENVIRONMENT_NAME)"
else ifeq (True,$(HAS_PYENV))
	@ echo ">>> Detected PYENV ... Creating new Pyenv environment!"
	@ echo ">>> \tCreating environment: \t $(ENVIRONMENT_NAME)"
	@ pyenv virtualenv $(PYTHON_VERSION) $(ENVIRONMENT_NAME) || echo ""
	@ pyenv local $(ENVIRONMENT_NAME)
	@ echo ">>> New Pyenv environment created: '$(ENVIRONMENT_NAME)'"
	@ pyenv virtualenvs
	@ echo
endif

## Deletes the Python environment
delete-environment:
ifeq (True,$(HAS_CONDA))
	@ echo ">>> Detected CONDA ... Deleting Conda environment, if applicable!"
	@ echo ">>> Deleting environment:    '$(ENVIRONMENT_NAME)'"
	@ ($(CONDA_DEACTIVATE) ; conda env remove --name $(ENVIRONMENT_NAME) -y) || echo ""
	@ echo ">>> Conda environment deleted: '$(ENVIRONMENT_NAME)'"
else ifeq (True,$(HAS_PYENV))
	@ echo ">>> Detected PYENV ... Deleting Pyenv environment!"
	@ echo ">>> Deleting environment:    '$(ENVIRONMENT_NAME)'"
	@ pyenv uninstall -f $(ENVIRONMENT_NAME) || echo ""
	@ rm $(PROJECT_DIR)/.python-version || echo ""
	@ echo ">>> Pyenv environment deleted: '$(ENVIRONMENT_NAME)'"
	@ pyenv virtualenvs
	@ echo
endif

## Upgrade the version of the 'pip' package
pip-upgrade:
	@ $(PYTHON_INTERPRETER) -m pip install --no-cache-dir -q --upgrade pip==$(PIP_VERSION)

## Sort the project packages requirements file
sort-requirements:
	@ 	sort $(REQUIREMENTS_FILE) | grep "\S" > $(REQUIREMENTS_FILE_TEMP) && \
		mv $(REQUIREMENTS_FILE_TEMP) $(REQUIREMENTS_FILE)
	@ 	sort $(REQUIREMENTS_DEV_FILE) | grep "\S" > $(REQUIREMENTS_DEV_FILE_TEMP) && \
		mv $(REQUIREMENTS_DEV_FILE_TEMP) $(REQUIREMENTS_DEV_FILE)

## Install Python dependencies into the Python environment
requirements: pip-upgrade sort-requirements
	@ $(PYTHON_INTERPRETER) -m pip install --no-cache-dir -q -r $(REQUIREMENTS_DEV_FILE)

# -------------------------- Functions for Code Linting -----------------------

## Installing the pre-commit Git hook
pre-commit-install:
	@ pre-commit install

## Uninstall the pre-commit Git hook
pre-commit-uninstall:
	@ pre-commit uninstall

## Run the 'pre-commit' linting step manually
lint:
	@ pre-commit run -a --hook-stage manual

###############################################################################
# Docker Commands - Local development                                         #
###############################################################################

LOCAL_DEV_DOCKER_PROJECT_NAME="$(PROJECT_NAME)_localdev"
LOCAL_DEV_SERVICE_NAME="local-dev"

## Clean Docker images
docker-prune:
	@	docker system prune -f

## Stops the local development service
all-stop: docker-local-dev-stop
	@	echo "All services are down"

## Starts the local development service
all-start: docker-local-dev-start
	@	echo "All services are up!"

## Build local development image
docker-local-dev-build: docker-prune
	@	cd $(LOCAL_DEVELOPMENT_DIR_PATH) && \
		docker compose \
		--project-name $(LOCAL_DEV_DOCKER_PROJECT_NAME) \
		build $(LOCAL_DEV_SERVICE_NAME)

## Start service for local development
docker-local-dev-start: docker-local-dev-build docker-local-dev-stop
	@	cd $(LOCAL_DEVELOPMENT_DIR_PATH) && \
		docker compose \
		--project-name $(LOCAL_DEV_DOCKER_PROJECT_NAME) \
		up -d $(LOCAL_DEV_SERVICE_NAME)

## Stop service for local development
docker-local-dev-stop:
	@	cd $(LOCAL_DEVELOPMENT_DIR_PATH) && \
		docker compose \
		--project-name $(LOCAL_DEV_DOCKER_PROJECT_NAME) \
		stop $(LOCAL_DEV_SERVICE_NAME)
	@	$(MAKE) docker-prune

## Start a shell session into the docker container
docker-local-dev-login:
	@	cd $(LOCAL_DEVELOPMENT_DIR_PATH) && \
		docker compose \
		--project-name $(LOCAL_DEV_DOCKER_PROJECT_NAME) \
		exec \
		$(LOCAL_DEV_SERVICE_NAME) /bin/zsh

###############################################################################
# Unit Tests and Code checking                                                #
###############################################################################

## Run all Python unit tests with verbose output and logs
test:
	python -m pytest -v -s

# See: https://github.com/google/addlicense for more information
## Add licenses to Python files
add-licenses:
	@	docker run -it \
		-v ${PWD}:/src \
		ghcr.io/google/addlicense \
		-f ./LICENSE.rst  \
		./src/**/*.py

###############################################################################
# Self Documenting Commands                                                   #
###############################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=25 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
