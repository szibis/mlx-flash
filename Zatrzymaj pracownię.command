#!/bin/bash
cd "$(dirname "$0")" || exit 1
make down
echo "Pracownia została zatrzymana."
read -r -p "Naciśnij Enter, aby zamknąć to okno."
