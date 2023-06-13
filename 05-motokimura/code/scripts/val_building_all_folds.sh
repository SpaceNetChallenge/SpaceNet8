#!/bin/bash
# example usage: ./scripts/this_script.sh 10100 15100

foundation=$1
flood=$2

# XXX: 5-fold
./scripts/val_building.sh $(($foundation+0)) $(($flood+0))
./scripts/val_building.sh $(($foundation+1)) $(($flood+1))
./scripts/val_building.sh $(($foundation+2)) $(($flood+2))
./scripts/val_building.sh $(($foundation+3)) $(($flood+3))
./scripts/val_building.sh $(($foundation+4)) $(($flood+4))
