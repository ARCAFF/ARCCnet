# ARCCnet
Active Region Cutout Classification Network

## Contents

- [Setup](#setup)
- [Setup for local code development](#setup-for-local-code-development)
  - [Makefile](#makefile)
  - [Starting up the Docker container and initializing the repository](#starting-up-the-docker-container-and-initializing-the-repository)
- [Tests](#tests)
- [Helpful Commands](#helpful-commands)
- [VS Code Extensions](#vs-code-extensions)
- [Resources](#resources)

## Setup

Ensure you have python and pip installed.

```shell
python --version
pip --version
```

From the root directory run the following command to install the
dependencies: `pip install -r requirements.txt`

## Setup for local code development

There are some steps that need to be done prior to being able to
properly run and develop the code in this repository.

The following is a list of steps that have to happen prior to starting to
work / test the pipelines of this repository:

### Makefile

The project comes with a `Makefile` (**not supported in Windows!**)
that can be used for executing commands to simplify the interaction
with this project. Keep in mind that folders with spaces in their names may cause issues.

The available options can be found by running the following:

```bash
    $: make

    Available rules:

    add-licenses              Add licenses to Python files
    clean                     Removes artifacts from the build stage, and other common Python artifacts.
    clean-build               Remove build artifacts
    clean-images              Clean left-over images
    clean-model-files         Remove files related to pre-trained models
    clean-pyc                 Removes Python file artifacts
    clean-secrets             Removes secret artifacts - Serverless
    clean-test                Remove test and coverage artifacts
    create-environment        Creates the Python environment
    create-envrc              Set up the envrc file for the project.
    delete-environment        Deletes the Python environment
    delete-envrc              Delete the local envrc file of the project
    destroy                   Remove ALL of the artifacts + Python environments
    docker-local-dev-build    Build local development image
    docker-local-dev-login    Start a shell session into the docker container
    docker-local-dev-start    Start service for local development
    docker-local-dev-stop     Stop service for local development
    docker-prune              Clean Docker images
    init                      Initialize the repository for code development
    lint                      Run the 'pre-commit' linting step manually
    pip-upgrade               Upgrade the version of the 'pip' package
    pre-commit-install        Installing the pre-commit Git hook
    pre-commit-uninstall      Uninstall the pre-commit Git hook
    requirements              Install Python dependencies into the Python environment
    show-params               Show the set of input parameters
    sort-requirements         Sort the project packages requirements file
    test                      Run all Python unit tests with verbose output and logs
```

> **NOTE**: If you're using `Windows`, you may have to copy and modify to some
> extents the commands that are part of the `Makefile` for some tasks.

While it is possible to run `make init` directly into a shell window,
the preferred method for development is through Docker:

### Starting up the Docker container and initializing the repository

In order to work on current / new features, one can use *Docker* to
start a new container and start the local development process.

To build the Docker image, one must follow the following steps:

1. Start the Docker daemon. If you're using Mac, one can use the
Docker Desktop App.

2. Go the project's directory and run the following command using the `Makefile`:
```bash
# Go the project's directory
cd /path/to/directory

# Build the Docker image and start a container
make docker-local-dev-start
```

3. Log into the container
```bash
# Log into the container
make docker-local-dev-login
```

4. Before we develop, we want to initialize the repository. This can easily be done
with the `init` command:
```bash
$: make init
```

This will do the following tasks:
- Clean Python files
- Initialize the `.envrc` file used by `direnv`.
- Delete an existing python environment for the project, if it exists.
- Creates a new environment, if applicable
- Apply `direnv allow` to allow for `direnv` modifications.
- Install package requirements via `pip`
- Install `pre-commit` for code-linting and code-checking.

These steps allow for the user to be able to develop new feature within
Docker, which makes it easier for developers to have the exact same set of
tools available.

## Tests

Unit tests can be found under the `src` folder alongside source code.
Test files end with `_test`. The following command will run all of the tests.

```shell
python -m pytest -v -s
```

The `-v` argument is for verbose output. The `-s` argument is for turning
off the capture mode so that print statements are printed to the console.

A Makefile command also exists to run these. See `make test`.

## Helpful Commands

Here is a list of commands that may be helpful when interacting with this project.

### Docker

List all Docker containers:

```shell
docker ps -a
```

## VS Code Extensions

To help facilitate local development you can install the [Visual Studio Code Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension for VS Code. This will allow you to connect to the local development Docker container and more easily develop features.

## Resources

- [direnv](https://github.com/direnv/direnv)
- [Docker](https://docs.docker.com/reference/)
- [Docker Compose](https://docs.docker.com/compose/)
- [flake8](https://flake8.pycqa.org/en/latest/)
- [git](https://git-scm.com/)
- [isort](https://pycqa.github.io/isort/index.html)
- [Makefile](https://www.gnu.org/software/make/manual/make.html)
- [Markdown](https://www.markdownguide.org/)
- [Poetry](https://python-poetry.org/)
- [pre-commit](https://pre-commit.com)
- [Pydantic](https://docs.pydantic.dev/)
- [pytest](https://docs.pytest.org/en/7.2.x/)
- [Python](https://www.python.org/)
