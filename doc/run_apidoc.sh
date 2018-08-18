#!/bin/sh

rm -rf generated/*.rst
SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc -o generated -M -E -e ../parts/
