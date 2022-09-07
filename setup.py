# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import setuptools

with open("./README.md", "r") as fh:
    long_description = fh.read()

with open("./lamcts/requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name='LA-MCTS',
    version='0.1',
    author="Jia Liu",
    author_email="jialiu@fb.com",
    description="LA-MCTS blackbox optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/LA-MCTS",
    packages=setuptools.find_packages(exclude=['docs', 'example']),
    install_requires=required,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: CC-BY-NC 4.0 license"
    ]
)
