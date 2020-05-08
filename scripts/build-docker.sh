#!/bin/bash -ex

# iconn: Interpretable Convolutional Neural Networks
# Copyright (C) 2020 Harish G. Naik <hnaik2@uic.edu>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

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
