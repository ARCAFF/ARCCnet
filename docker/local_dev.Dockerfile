ARG PYTHON_VERSION="3.9"

FROM --platform=$TARGETPLATFORM python:${PYTHON_VERSION}

# --- SYSTEM ARCHITECTURE
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETVARIANT

RUN printf "I'm building for TARGETPLATFORM=${TARGETPLATFORM}" \
    && printf ", TARGETARCH=${TARGETARCH}" \
    && printf ", TARGETVARIANT=${TARGETVARIANT} \n"

# --- Environment variables
ENV REQUIREMENTS_FILE="requirements.txt"
ENV REQUIREMENTS_FILE_DEV="requirements-dev.txt"
ENV PROJECT_DIR="/opt/program"
ENV DEBIAN_FRONTEND=noninteractive

# --- Dockerfile Metadata
LABEL Maintainer="ARCAFF Team"

# ------------------------- COPYING AND DIRECTORIES ---------------------------
# COPY ./ ${PROJECT_DIR}

# --------------------- INSTALLING EXTRA PACKAGES -----------------------------
# --- Updating packages and installing packages at the system-level

RUN apt-get -y update && \
    apt-get upgrade -y && \
    apt-get clean && \
    # Installing system-level packages
    apt-get install -y \
    git \
    ssh \
    tree \
    direnv \
    bash-completion \
    zsh \
    htop \
    # Installing for Opencv
    ffmpeg \
    libsm6 \
    libxext6 \
    && \
    # Cleaning out
    rm -rf /var/lib/apt/lists/* && \
    # Cleaning installs
    apt-get clean && \
    # Installing ZSH and OhZSH
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    echo "source /etc/profile.d/bash_completion.sh" >> /root/.bashrc && \
    echo "source /etc/profile.d/bash_completion.sh" >> /root/.zshrc && \
    # Install direnv
    echo 'eval "$(direnv hook zsh)"' >> "${OUTDIR}/.zshrc" && \
    echo 'eval "$(direnv hook bash)"' >> "${OUTDIR}/.bash"

# --------------------------- PYTHON-RELATED-LOCAL ----------------------------

# ----------------------------- PYTHON-SPECIFIC -------------------------------

# Set some environment variables. PYTHONUNBUFFERED keeps Python from
# buffering our standard output stream, which means that logs can be
# delivered to the user quickly. PYTHONDONTWRITEBYTECODE keeps Python
# from writing the .pyc files which are unnecessary in this case. We also
# update PATH so that the train and serve programs are found when the
# container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

WORKDIR ${PROJECT_DIR}

CMD ["/bin/sleep", "365d"]
