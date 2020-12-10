#!/bin/bash

cat $1 | grep -e "Testing net" -e "Test net output #0" | paste - - | awk '{print $6 $20}' | sed 's/,/ /g'
