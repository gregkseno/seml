#!/usr/bin/env bash

set -o errexit

if ! mutmut run --no-progress ${@+"$@"}; then
    exit_code="2"
fi

mutmut junitxml > mutmut.xml

exit "${exit_code-0}"
