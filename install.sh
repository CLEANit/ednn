#!/bin/bash

if [ $(uname) == "Darwin" ]
    then
        echo "export PATH=\"$(pwd)""/ml:$""PATH\"" >> ~/.profile
        echo "export PATH=\"$(pwd)""/scripts:$""PATH\"" >> ~/.profile

elif [ $(uname) == "Linux" ]
    then
        echo "export PATH=\"$(pwd)""/ml:$""PATH\"" >> ~/.bashrc
        echo "export PATH=\"$(pwd)""/scripts:$""PATH\"" >> ~/.bashrc
fi

