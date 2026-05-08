ARG ISAACLAB_BASE_IMAGE_ARG

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_LEAP_HAND_PROJECT_PATH_ARG
ENV DOCKER_LEAP_HAND_PROJECT_PATH=${DOCKER_LEAP_HAND_PROJECT_PATH_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_LEAP_HAND_PROJECT_PATH}

# # Install whatever you need as additional dependencies.
RUN bash -i -c "source ${HOME}/.bashrc && \
    cd ${DOCKER_LEAP_HAND_PROJECT_PATH}/source/LEAP_Isaaclab && \
    pip install -e ."

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace
