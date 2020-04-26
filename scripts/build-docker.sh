#!/bin/bash -ex

project="iconn"

script_path=$(realpath $0)
script_dir=$(dirname ${script_path})
root_dir=$(dirname ${script_dir})
docker_config_dir="${root_dir}/docker"
datetime=$(date +"%Y%m%d-%H%M%S")
docker_tag="${project}-${datetime}"

variant=$1

if [ -z ${variant} ];
then
    echo -e "Variant not specified, building Pytorch image"
fi

docker_file="${docker_config_dir}/Dockerfile.${variant}"

if [ ! -f ${docker_file} ]
then
    echo -e "No docker file ${docker_file} found, variant ${variant} invalid"
    exit 1
fi

pushd .
cd ${root_dir}
docker build -t ${docker_tag} -f ${docker_file} .
popd
